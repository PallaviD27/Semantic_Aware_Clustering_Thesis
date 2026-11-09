import csv
from pathlib import Path
from itertools import product
from channel import los_channel_model
import json
from SNRSweep import sigma_sweep

run_dir = Path(r".\Results\run_20261108_214919")
gain_path = run_dir / "discriminant_gain_table.csv"

client_gains = {}

with gain_path.open(newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        client_id = int(row["client_id"])
        bits = int(row["bits"])
        gain = float(row["gain"])

        if client_id not in client_gains:
            client_gains[client_id] = {}

        client_gains[client_id][bits] = gain

print(client_gains)
print("Client 0 at 8 bits:", client_gains[0][8])

bit_options = [4, 6, 8, 16]
allocations = list(product(bit_options, repeat=3))

print("Number of allocations:", len(allocations))
print("First five:", allocations[:5])

allocation_gains = {}

for allocation in allocations:
    total_gain = 0.0

    for client_id, bits in enumerate(allocation):
        total_gain += client_gains[client_id][bits]

    allocation_gains[allocation] = total_gain / len(allocation)

print("DG for (8, 8, 8):", allocation_gains[(8, 8, 8)])

feature_dimension = 128

slot_seconds = [0.001024, 0.001024, 0.001024]         # 1.024 ms per client for baseline and 8 bits
bandwidth_hz = [1_000_000, 1_000_000, 1_000_000]  # 1 MHz per client

with (run_dir / "run_config.json").open() as file:
    config = json.load(file)

h = float(los_channel_model(
    Gt=config["Gt"],
    Gr=config["Gr"],
    wavelength_m=config["wavelength_m"],
    d_m=config["d_m"],
    L=1.0,
))


import math

# Fixed reference condition: 0 dB with receiver noise std = 0.5.
sigma_ref = 0.5
Pref = sigma_ref**2 / h**2

# Calculate the TOTAL reference energy once, using 8 bits per client.
baseline_energy = 0.0

for cid in client_gains:
    payload_bits = feature_dimension * 8

    required_snr = (
        2 ** (payload_bits / (bandwidth_hz[cid] * slot_seconds[cid]))
        - 1
    )

    reference_power = (sigma_ref**2 / h**2) * required_snr
    baseline_energy += reference_power * slot_seconds[cid]

print("Fixed reference energy:", baseline_energy)

snr_points = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

# Start with ONE fixed energy budget to understand the SNR sweep.
budget_factor = 1.0
# budget_factor = 4.0
energy_budget = budget_factor * baseline_energy

# Separate filename preserves your earlier budget-sweep results.
output_path = run_dir / "resource_allocation_snr_sweep.csv"

with output_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "budget_factor", "energy_budget", "energy_used", "system_dg",
        "client_id", "bits", "tx_power", "snr_db", "client_dg",
        "reference_snr_db", "sigma_rx",
    ])

    for SNRb_db in snr_points:
        sigma_rx = sigma_sweep(Pref, h, SNRb_db)

        noise_to_channel_ratio_w = {
            cid: sigma_rx**2 / h**2
            for cid in client_gains
        }

        # These costs must be recalculated at every reference SNR.
        allocation_energies = {}
        allocation_powers = {}

        for allocation in allocations:
            total_energy = 0.0
            client_powers = []

            for cid, bits in enumerate(allocation):
                payload_bits = feature_dimension * bits
                required_rate = payload_bits / slot_seconds[cid]

                required_snr = (
                    2 ** (required_rate / bandwidth_hz[cid]) - 1
                )

                power = noise_to_channel_ratio_w[cid] * required_snr

                client_powers.append(power)
                total_energy += power * slot_seconds[cid]

            allocation_powers[allocation] = client_powers
            allocation_energies[allocation] = total_energy

        feasible = [
            allocation for allocation in allocations
            if allocation_energies[allocation] <= energy_budget
            or math.isclose(
                allocation_energies[allocation],
                energy_budget,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ]

        if not feasible:
            print(
                f"Reference SNR {SNRb_db:g} dB: "
                "no feasible allocation"
            )
            continue

        best = max(
            feasible,
            key=lambda a: (
                allocation_gains[a],
                -allocation_energies[a],
            ),
        )

        energy_used = allocation_energies[best]
        system_dg = allocation_gains[best]

        print(
            f"Reference SNR {SNRb_db:g} dB | "
            f"sigma={sigma_rx:.6g} | "
            f"bits={best} | system DG={system_dg:.3f}"
        )

        for cid, bits in enumerate(best):
            power = allocation_powers[best][cid]

            # Actual received SNR, distinct from reference SNR.
            snr_linear = power / noise_to_channel_ratio_w[cid]
            snr_db = 10 * math.log10(snr_linear)

            writer.writerow([
                budget_factor, energy_budget, energy_used, system_dg,
                cid, bits, power, snr_db, client_gains[cid][bits],
                SNRb_db, sigma_rx,
            ])

print(f"Saved allocation results to {output_path}")