import argparse
import csv
from itertools import combinations
from pathlib import Path
import torch

from quantization import quantize_features, dequantize_features
run_dir = Path(r".\Results\run_20261108_214919")
all_client_stats = {}

for client_id in range(3):
    path = run_dir / f"class_features_client_{client_id}.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)

    print(data.keys())

    print("Stage:", data["stage"])
    print("Normalization enabled:", data["signal_normalize"])
    print("Features shape", data["features"].shape)
    print("Labels shape", data["labels"].shape)
    print("Labels", data["labels"].unique())

    # Get the count of how many images belong to the same class
    features = data["features"]
    print(features)
    labels = data["labels"]
    class_stats = {}

    class_ids, counts = torch.unique(labels, return_counts=True)
    print(class_ids, counts)
    # Zip combines the two tuples element-wise. .tolist() returns lists,
    # and zip pairs each class ID with its corresponding count.
    for class_id, count in zip(class_ids.tolist(), counts.tolist()):
        print(f"Class {class_id} has {count} images")

    # Select those features for which the corresponding label in data was same as the local_id

    for class_id in class_ids.tolist():
        quantizer_input = features[labels==class_id]
        class_features = quantizer_input.double()
        class_mean = class_features.mean(dim=0)
        class_covar = torch.cov(class_features.T)
        centered_features = class_features - class_mean
        class_stats[class_id] = {
            "features": class_features,
            "mean": class_mean,
            "covariance": class_covar,
            "by_bits": {},
        }

    #     Calculating quantization error with 8 bits as baseline
    #     for loop for bits yet to be implemented
        for bits in [4,6,8,16]:
            indices = quantize_features(quantizer_input, bits=bits, upper_bound=4.0,lower_bound=-4.0)
            dequantized_features = dequantize_features(indices, bits=bits, upper_bound=4.0,lower_bound=-4.0).double()

        # Use double precision for all subsequent statistics.


            quantization_error = dequantized_features - class_features
            print(f"Class {class_id}: features {class_features.shape}, "
                  f"mean {class_mean.shape}, covariance {class_covar.shape}")

            # quantization_error = dequantized_features - class_features
            print("Shape quant error:", quantization_error.shape)
            quant_error_mean = torch.mean(quantization_error, dim=0)
            quant_error_covar = torch.cov(quantization_error.T)

            centered_quant_error = quantization_error - quant_error_mean
            feature_error_crosscov = (1/(class_features.shape[0] -1)) * (centered_features.T @ centered_quant_error)

            dequantized_features_mean = (class_mean + quant_error_mean)

            dequantized_features_covar = (class_covar + quant_error_covar
                                          +feature_error_crosscov + feature_error_crosscov.T)

            covariance_check = dequantized_features_covar.double()
            covariance_check = (
                covariance_check + covariance_check.T
            ) / 2

            eigenvalues = torch.linalg.eigvalsh(covariance_check)
            min_eigenvalue = eigenvalues.min().item()
            max_eigenvalue = eigenvalues.max().item()

            print(
                f"Client {client_id}, class {class_id}, {bits} bits: "
                f"min eigenvalue={min_eigenvalue:.6g}, "
                f"max eigenvalue={max_eigenvalue:.6g}"
            )

            class_stats[class_id]["by_bits"][bits] = {
                "quant_error_mean": quant_error_mean,
                "quant_error_covar": quant_error_covar,
                "feature_error_crosscov": feature_error_crosscov,
                "reconstructed_feature_mean": dequantized_features_mean,
                "reconstructed_feature_covar": dequantized_features_covar,

            }

            torch.testing.assert_close(dequantized_features_mean, dequantized_features.mean(dim=0))
            torch.testing.assert_close(dequantized_features_covar, torch.cov(dequantized_features.T))
            print(f"Client {client_id}, class {class_id}: mean and covariance checks passed.")


    all_client_stats[client_id] = class_stats

# Summary
for saved_client_id, saved_client_classes in all_client_stats.items():
    print(f"Client{saved_client_id} has stored local classes {list(saved_client_classes)}")

client_gains = {}
for client_id, class_stats in all_client_stats.items():
    client_gains[client_id] = {}
    for bits in [4,6,8,16]:
        pairwise_gains = []
        for class_1,class_2 in combinations(class_stats, 2):
            stats_1 = class_stats[class_1]["by_bits"][bits]
            stats_2 = class_stats[class_2]["by_bits"][bits]
            mean_diff = stats_1["reconstructed_feature_mean"] - stats_2["reconstructed_feature_mean"]
            cov_1 = stats_1["reconstructed_feature_covar"]
            cov_2 = stats_2["reconstructed_feature_covar"]

            print(f"Client{client_id} has pairs ({class_1},{class_2})" )
            d = mean_diff.numel()

            mean_term = mean_diff @ (torch.linalg.solve(cov_1,mean_diff) + torch.linalg.solve(cov_2,mean_diff))
            covariance_term = (
                                torch.trace(torch.linalg.solve(cov_2,cov_1))
                               + torch.trace(torch.linalg.solve(cov_1, cov_2))
            )
            pairwise_dg = (1/2) * (mean_term + covariance_term - 2 * d)
            pairwise_gains.append(pairwise_dg.item())

            print(f"Pairwise DG: {pairwise_dg.item()}")
            print(
                f"Client {client_id}, {bits} bits, pair ({class_1}, {class_2}): "
                f"mean contribution={0.5 * mean_term.item():.6f}, "
                f"covariance contribution={0.5 * (covariance_term.item() - 2 * d):.6f}"
            )
        client_gains[client_id][bits] = sum(pairwise_gains) / len(pairwise_gains)
        print(
            f"Client {client_id}, {bits} bits: "
            f"average DG = {client_gains[client_id][bits]:.6f}"
        )

system_gain = sum(
    gains[8] for gains in client_gains.values()
) / len(client_gains)

print(f"System DG at (8, 8, 8): {system_gain:.6f}")
output_path = run_dir / "discriminant_gain_table.csv"

with output_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["client_id", "bits", "gain"])

    for client_id, gains in client_gains.items():
        for bits, gain in gains.items():
            writer.writerow([client_id, bits, gain])

print(f"Saved gain table to {output_path}")