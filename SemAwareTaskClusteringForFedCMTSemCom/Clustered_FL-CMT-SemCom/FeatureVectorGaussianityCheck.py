"""Plot class-conditional CNN feature distributions from saved .pt files.

Example (PowerShell / PyCharm terminal):
    python plot_class_gaussianity.py --files "class_features_client_0.pt" "class_features_client_1.pt" "class_features_client_2.pt"

Required packages: torch, numpy, scipy, matplotlib.
Required dictionary keys: features [images, coordinates], labels [images].
Optional key: features_before_normalization, with the same shape and row order.
Optional metadata: client_id, stage, sampling, signal_normalize.

features is plotted as the SAVED feature stage. It is called 'After normalization'
only when signal_normalize=True is recorded. No vectors are normalized, sampled,
quantized, or passed through a model by this script.

Outputs per class:
  *_scatter.png: three coordinate pairs and two PCA components, for each stage.
  *_joint_qq.png: covariance-adjusted distances versus a finite-sample Gaussian
                 reference, calculated with ALL original coordinates.
  gaussianity_summary.csv: sample size, dimension, norm range, Mardia kurtosis,
                          simulation-based two-sided p-value and any errors.

Each row is one saved image sample. PCA/scatter plots are visual diagnostics.
Mardia kurtosis can detect some departures from multivariate normality; a large
p-value does not prove normality or independence. Inference presumes independent
image samples and a fixed model. Augmented/duplicate images can violate this.
No multiple-comparison adjustment is applied to the reported per-group p-values.
"""
from pathlib import Path
import argparse
import csv
import re
import warnings

import numpy as np
import torch
from scipy.linalg import solve_triangular
import matplotlib
matplotlib.use("Agg")  # Save figures without blocking the training IDE.
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_features(path):
    # Fail safely for unsupported pickle objects; do not automatically retry
    # with weights_only=False. The current feature files contain tensor data.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dictionary")
    if "features" not in data or "labels" not in data:
        raise ValueError(f"{path}: needs 'features' and 'labels' keys")
    labels = as_numpy(data["labels"]).reshape(-1)
    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError(f"{path}: labels must be finite numeric class IDs")
    normalized = bool(data.get("signal_normalize", False))
    stages = []
    if "features_before_normalization" in data:
        stages.append(("Before normalization", as_numpy(data["features_before_normalization"])))
    saved_label = "After normalization" if normalized else "Saved features (normalization not confirmed)"
    stages.append((saved_label, as_numpy(data["features"])))
    shape = None
    for name, x in stages:
        if x.ndim != 2 or len(x) != len(labels):
            raise ValueError(f"{path}: {name} must have shape [images, coordinates], matching labels")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: {name} contains NaN/Inf; inspect collection first")
        if shape is not None and x.shape != shape:
            raise ValueError(f"{path}: saved stages have different shapes")
        shape = x.shape
    client = data.get("client_id", path.stem)
    if isinstance(client, torch.Tensor):
        client = client.item()
    return str(client), labels, [(name, x.astype(np.float64)) for name, x in stages]


def squared_distances(x):
    """Fit mean/full covariance (divisor N), return sorted Mahalanobis distances.

    No diagonal approximation or ridge is used: the Gaussian reference must use
    exactly the same operation. Singular groups are reported, not silently fixed.
    """
    n, d = x.shape
    if n <= d:
        raise ValueError(f"need more images than coordinates: N={n}, d={d}")
    centered = x - x.mean(axis=0)
    if np.linalg.matrix_rank(centered) < d:
        raise ValueError("singular covariance (constant/dependent coordinates)")
    covariance = centered.T @ centered / n
    chol = np.linalg.cholesky(covariance)
    white = solve_triangular(chol, centered.T, lower=True)
    return np.sort(np.sum(white * white, axis=0))


def gaussian_reference(n, d, count, seed):
    """Finite-sample null: refit mean and covariance in every simulation."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, d]))
    simulated = np.empty((count, n))
    # Gaussian draws have full rank almost surely when N>d; omit repeated SVDs.
    for b in range(count):
        x = rng.standard_normal((n, d))
        x -= x.mean(0)
        chol = np.linalg.cholesky(x.T @ x / n)
        white = solve_triangular(chol, x.T, lower=True)
        simulated[b] = np.sort(np.sum(white * white, axis=0))
    return (np.median(simulated, axis=0),
            np.quantile(simulated, [0.025, 0.975], axis=0),
            np.mean(simulated ** 2, axis=1))


def gaussian_ellipse(ax, z):
    cov = np.cov(z, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values[0] <= 0:
        return
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * np.sqrt(np.maximum(values, 0) * 5.991464547)
    ax.add_patch(Ellipse(z.mean(0), width, height, angle=angle,
                         fill=False, linestyle="--", color="#344054", linewidth=1))


def plot_scatter(groups, pairs, title, output):
    fig, axes = plt.subplots(len(groups), len(pairs) + 1,
                             figsize=(4 * (len(pairs) + 1), 3.7 * len(groups)), squeeze=False)
    colors = ["#1769aa", "#d35b21"] if len(groups) == 2 else ["#d35b21"]
    for row, ((stage, x), color) in enumerate(zip(groups, colors)):
        centered = x - x.mean(0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        total = np.sum(singular ** 2)
        ratios = singular ** 2 / total if total > 0 else np.zeros_like(singular)
        projections = [(x[:, [a - 1, b - 1]], f"Feature {a}", f"Feature {b}") for a, b in pairs]
        projections.append((centered @ vt[:2].T, f"PC1 ({ratios[0]:.1%})", f"PC2 ({ratios[1]:.1%})"))
        for col, (z, xlabel, ylabel) in enumerate(projections):
            ax = axes[row, col]
            ax.scatter(z[:, 0], z[:, 1], s=9, alpha=0.35, c=color, edgecolors="none")
            gaussian_ellipse(ax, z)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=stage + (" | PCA" if col == len(pairs) else ""))
            ax.grid(alpha=0.15)
    fig.suptitle(title, fontsize=17)
    fig.text(0.02, 0.015, "One dot = one image. Dashed ellipse = fitted Gaussian 95% contour, not a test.\n"
             "Panels have separate axis limits. PCA is fitted separately per class and stage; no whitening.", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("gaussianity_plots"))
    parser.add_argument("--pairs", nargs="+", default=["7,23", "1,2", "64,128"],
                        help="One-based coordinate pairs; change for dimensions below 128")
    parser.add_argument("--simulations", type=int, default=199,
                        help="Gaussian null simulations; 999 gives finer p-value resolution but takes longer")
    parser.add_argument("--seed", type=int, default=2048)
    args = parser.parse_args()
    if args.simulations < 39 or args.seed < 0:
        parser.error("use --simulations >=39 and --seed >=0")
    try:
        pairs = [tuple(map(int, value.split(','))) for value in args.pairs]
        if not pairs or any(len(p) != 2 or min(p) < 1 or p[0] == p[1] for p in pairs):
            raise ValueError()
    except ValueError:
        parser.error('--pairs must contain distinct positive coordinates, e.g. "7,23" "1,2"')
    args.output.mkdir(parents=True, exist_ok=True)
    reference_cache = {}
    rows = []
    for file_number, path in enumerate(args.files):
        client, labels, stages = load_features(path)
        d = stages[0][1].shape[1]
        if d < 2 or max(max(pair) for pair in pairs) > d:
            parser.error(f"{path}: dimension is {d}; specify suitable --pairs")
        for label in np.unique(labels):
            groups = [(stage, x[labels == label]) for stage, x in stages]
            n = len(groups[0][1])
            if n < 3:
                warnings.warn(f"Skipping client {client}, class {label}: fewer than 3 samples")
                continue
            title = f"Client {client} | Class ID {label} | {n} images | {d} coordinates"
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"file{file_number}_client{client}_class{label}")
            print(title, flush=True)
            plot_scatter(groups, pairs, title, args.output / f"{stem}_scatter.png")
            fig, ax = plt.subplots(figsize=(8, 6))
            reference_drawn = False
            for stage, x in groups:
                norms = np.linalg.norm(x, axis=1)
                record = dict(file=str(path), client=client, class_id=label, stage=stage,
                              images=n, coordinates=d, norm_min=norms.min(), norm_max=norms.max(),
                              mardia_kurtosis="", mc_two_sided_p="", simulations=args.simulations, status="")
                try:
                    observed = squared_distances(x)
                    if (n, d) not in reference_cache:
                        print(f"  Simulating Gaussian reference: N={n}, d={d}, B={args.simulations}", flush=True)
                        reference_cache[n, d] = gaussian_reference(n, d, args.simulations, args.seed)
                    expected, band, null_kurtosis = reference_cache[n, d]
                    kurtosis = np.mean(observed ** 2)
                    upper = (1 + np.count_nonzero(null_kurtosis >= kurtosis)) / (args.simulations + 1)
                    lower = (1 + np.count_nonzero(null_kurtosis <= kurtosis)) / (args.simulations + 1)
                    pvalue = min(1.0, 2 * min(upper, lower))
                    record.update(mardia_kurtosis=kurtosis, mc_two_sided_p=pvalue, status="ok")
                    if not reference_drawn:
                        ax.fill_between(expected, band[0], band[1], color="0.85", label="95% pointwise Gaussian reference band")
                        ax.plot(expected, expected, "k--", label="Gaussian reference")
                        reference_drawn = True
                    ax.plot(expected, observed, label=f"{stage} (kurtosis p={pvalue:.4g})")
                except (ValueError, np.linalg.LinAlgError) as error:
                    record["status"] = str(error)
                    warnings.warn(f"{title}, {stage}: joint diagnostic skipped: {error}")
                rows.append(record)
            ax.set(title=title, xlabel="Expected ordered squared distance under Gaussian model",
                   ylabel="Observed squared Mahalanobis distance")
            ax.grid(alpha=0.15)
            if reference_drawn:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "Joint diagnostic unavailable.\nSee gaussianity_summary.csv.", ha="center", transform=ax.transAxes)
            fig.text(0.02, 0.01, "All coordinates used. A matching curve does not prove multivariate normality.\n"
                     "P-values: two-sided Monte Carlo Mardia kurtosis; no multiplicity correction.", fontsize=9)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            fig.savefig(args.output / f"{stem}_joint_qq.png", dpi=160)
            plt.close(fig)
    if rows:
        with (args.output / "gaussianity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Finished. Outputs: {args.output.resolve()}")
    print(f"Two-sided Monte Carlo p-value floor: {2/(args.simulations+1):.4g}. Non-rejection is not proof of Gaussianity.")


if __name__ == "__main__":
    main()
