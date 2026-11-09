"""Evaluate frozen EuroSAT heads on saved held-out features, on CPU."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from channel import los_channel_model, los_symbol_channel
from model import TaskHead
from modulation import modulation, demodulation
from quantization import (
    quantize_features,
    dequantize_features,
    index_to_bits,
    bits_to_index,
)


@torch.inference_mode()
def accuracy(head, features, labels, bits, power, h, sigma, repeats, seed):
    # Keep test features fixed and redraw channel noise for each repetition.
    # Reusing the seed across conditions makes comparisons reproducible.
    torch.manual_seed(seed)
    correct = 0

    for x, target in zip(features.split(128), labels.split(128)):
        indices = quantize_features(
            x,
            bits=bits,
            lower_bound=-4.0,
            upper_bound=4.0,
        )

        symbols = modulation(index_to_bits(indices, bits=bits))

        for _ in range(repeats):
            equalized, _ = los_symbol_channel(
                symbols,
                h=h,
                tx_power=power,
                noise_std_rx=sigma,
            )

            decoded = bits_to_index(
                demodulation(equalized),
                bits=bits,
            )

            received = dequantize_features(
                decoded,
                bits=bits,
                lower_bound=-4.0,
                upper_bound=4.0,
                dtype=x.dtype,
            )

            predictions = head(received).argmax(dim=1)
            correct += (predictions == target).sum().item()

    return 100.0 * correct / (len(labels) * repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=["allocation", "fixed8"],
        default="allocation",
    )
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[-6, -3, 0, 3, 6, 9],
        help="Received SNR points used only in fixed8 mode.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    folder = args.run_dir

    with (folder / "run_config.json").open(encoding="utf-8") as file:
        cfg = json.load(file)

    h = float(
        los_channel_model(
            Gt=cfg["Gt"],
            Gr=cfg["Gr"],
            wavelength_m=cfg["wavelength_m"],
            d_m=cfg["d_m"],
            L=1.0,
        )
    )

    if not math.isfinite(h) or h <= 0:
        raise ValueError("Channel amplitude gain h must be finite and positive.")

    clients = list(range(int(cfg["num_users"])))
    if not clients:
        raise ValueError("The configuration contains no clients.")

    allocations = []
    budget_factor = None
    sigma_fixed = None

    if args.mode == "allocation":
        allocation_path = folder / "resource_allocation_snr_sweep.csv"

        with allocation_path.open(newline="", encoding="utf-8") as file:
            allocations = list(csv.DictReader(file))

        if not allocations:
            raise ValueError("No feasible allocations were saved in the CSV.")

        budget_factors = {
            float(row["budget_factor"]) for row in allocations
        }

        if len(budget_factors) != 1:
            raise ValueError(
                "Use one budget factor for this plot. "
                "Different budgets need separate curves."
            )

        budget_factor = next(iter(budget_factors))

    else:
        sigma_fixed = float(cfg["sigma_signal"])

        if not math.isfinite(sigma_fixed) or sigma_fixed <= 0:
            raise ValueError(
                "Fixed8 mode requires a finite, positive receiver noise std."
            )

    results = []
    fig, ax = plt.subplots()
    tikz_curves = []

    # Evaluate every client separately.
    for cid in clients:
        saved = torch.load(
            folder / f"test_features_client_{cid}.pt",
            map_location="cpu",
            weights_only=True,
        )

        ckpt = torch.load(
            folder / f"client_{cid}_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )

        if (
            saved["split"] != "test"
            or saved["stage"] != "before_quantization"
            or int(saved["client_id"]) != cid
            or int(ckpt["client_id"]) != cid
        ):
            raise ValueError(
                f"Unexpected feature or checkpoint metadata for client {cid}"
            )

        features = saved["features"].float()
        labels = saved["labels"].long()

        if (
            features.ndim != 2
            or labels.shape != (len(features),)
            or not len(labels)
        ):
            raise ValueError(f"Invalid test tensor shapes for client {cid}")

        head = TaskHead(
            input_dim=features.shape[1],
            output_dim=int(ckpt["num_classes"]),
        )
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()

        if args.mode == "allocation":
            # Use the exact power and noise saved by resource allocation.
            points = [
                (
                    float(row["budget_factor"]),
                    float(row["reference_snr_db"]),
                    int(row["bits"]),
                    float(row["tx_power"]),
                    float(row["sigma_rx"]),
                )
                for row in allocations
                if int(row["client_id"]) == cid
            ]

        else:
            # Separate experiment: fixed noise and 8 bits, varying power.
            points = [
                (
                    "",
                    db,
                    8,
                    (sigma_fixed / h) ** 2 * 10 ** (db / 10.0),
                    sigma_fixed,
                )
                for db in args.snrs
            ]

        if not points:
            raise ValueError(f"No evaluation points for client {cid}")

        cache = {}
        curve_points = []

        for budget, reference_db, bits, power, sigma in points:
            if (
                not math.isfinite(power)
                or power <= 0
                or not math.isfinite(sigma)
                or sigma <= 0
                or not math.isfinite(reference_db)
            ):
                raise ValueError(
                    f"Invalid power, noise, or SNR for client {cid}"
                )

            actual_db = 10.0 * math.log10(
                h**2 * power / sigma**2
            )

            # Identical transmission conditions can reuse an evaluation.
            key = (bits, power, sigma)

            if key not in cache:
                cache[key] = accuracy(
                    head,
                    features,
                    labels,
                    bits,
                    power,
                    h,
                    sigma,
                    args.repeats,
                    args.seed + cid,
                )

            pct = cache[key]

            results.append(
                dict(
                    client_id=cid,
                    budget_factor=budget,
                    reference_snr_db=(
                        reference_db if args.mode == "allocation" else ""
                    ),
                    bits=bits,
                    tx_power=power,
                    sigma_rx=sigma,
                    snr_db=actual_db,
                    accuracy_pct=pct,
                    test_images=len(labels),
                    noise_repeats=args.repeats,
                )
            )

            if args.mode == "allocation":
                print(
                    f"Client {cid} | "
                    f"reference SNR {reference_db:.2f} dB | "
                    f"{bits} bits | "
                    f"received SNR {actual_db:.2f} dB | "
                    f"accuracy {pct:.2f}%"
                )
                plot_db = reference_db

            else:
                print(
                    f"Client {cid} | {bits} bits | "
                    f"received SNR {actual_db:.2f} dB | "
                    f"accuracy {pct:.2f}%"
                )
                plot_db = actual_db

            # Preserve every reference SNR point, even if results are cached.
            curve_points.append((plot_db, pct))

        curve_points.sort()
        db_values, accuracies = zip(*curve_points)

        ax.plot(
            db_values,
            accuracies,
            "o-",
            label=f"Client {cid}",
        )

        tikz_curves.append(
            (cid, list(db_values), list(accuracies))
        )

    # Save combined results after all clients have been evaluated.
    if args.mode == "allocation":
        stem = "test_accuracy_reference_snr_sweep"
        xlabel = "Reference SNR (dB)"
        title = (
            f"Resource allocation: budget "
            f"{budget_factor:g} × reference energy"
        )
    else:
        stem = "test_accuracy_fixed8"
        xlabel = "Received SNR (dB)"
        title = "Fixed 8-bit quantization"

    with (folder / f"{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    ax.set(
        xlabel=xlabel,
        ylabel="Test accuracy (%)",
        ylim=(0, 101),
        title=title,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", dpi=200)

    tikz_path = folder / f"{stem}.tikz"

    with tikz_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{tikzpicture}\n")
        file.write("\\begin{axis}[\n")
        file.write(f"  xlabel={{{xlabel}}},\n")
        file.write("  ylabel={Test accuracy (\\%)},\n")
        file.write("  ymin=0, ymax=101,\n")
        file.write("  grid=major,\n")
        file.write("  legend pos=south east,\n")
        file.write("]\n")

        for cid, snrs, accuracies in tikz_curves:
            coordinates = " ".join(
                f"({snr:.8f},{acc:.8f})"
                for snr, acc in zip(snrs, accuracies)
            )

            file.write(
                f"\\addplot+[mark=*] coordinates "
                f"{{{coordinates}}};\n"
            )
            file.write(
                f"\\addlegendentry{{Client {cid}}}\n"
            )

        file.write("\\end{axis}\n")
        file.write("\\end{tikzpicture}\n")

    plt.close(fig)

    print(
        f"Saved {stem}.csv, {stem}.png and {stem}.tikz "
        f"in {folder}"
    )


if __name__ == "__main__":
    main()