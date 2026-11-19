
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import lambertw

BITS_PER_FEATURE_VECTOR = 128
MIN_BITS = 3
MAX_BITS = 16

def exact_benchmark(gains, ch: Channel):
    """P1 global discrete benchmark ONLY; never called from solve_sca/recovery."""
    best = None
    feasible = 0
    for combo in product(range(MIN_BITS, MAX_BITS + 1), repeat=gains.shape[0]):
        b = np.asarray(combo, dtype=int)
        total_e, times, energies = min_energy_time_allocation(b, ch)
        if total_e <= ch.total_energy_j * (1 + 1e-9):
            feasible += 1
            val = bits_at(gains, b)
            if best is None or val > best["dg"]:
                best = {"bits": b.tolist(), "dg": val, "energies_j": energies.tolist(),
                        "times_s": times.tolist(), "total_energy_j": total_e}
    return best, feasible


def write_solution(out_dir, name, bits, energies, times, ch, gains):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(len(bits)):
        power = energies[k] / times[k]
        capacity = times[k] * ch.bandwidth_hz[k] * np.log2(1 + ch.gamma[k] * power)
        rows.append({"client_id": k, "bits": int(bits[k]), "dg": float(gains[k, bits[k] - MIN_BITS]),
                     "energy_j": float(energies[k]), "time_s": float(times[k]),
                     "power_w": float(power), "gamma_per_w": float(ch.gamma[k]),
                     "capacity_bits": float(capacity), "payload_bits": int(BITS_PER_FEATURE_VECTOR * bits[k]),
                     "actual_snr_db": float(10 * np.log10(ch.gamma[k] * power))})
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows




# ============================================================================
# SWEEP DRIVER: each reference SNR is a NEW P1/P2/P3/P4 problem with new gamma.
# Energy and time budgets remain FIXED over the entire sweep.
# ============================================================================

def baseline_energy_for_eight_bits(gamma, bandwidth_hz, per_client_time_s):
    """Reference 8-bit total energy used ONLY to set a fixed budget once."""
    exponent = (BITS_PER_FEATURE_VECTOR * 8 * math.log(2)
                / (per_client_time_s * bandwidth_hz))
    return float(np.sum(per_client_time_s / gamma * np.expm1(exponent)))


def select_feasible_seeds(ch, n, first):
    """Deterministic multi-start choices, not an integer grid search."""
    candidates = [first, MIN_BITS, 5, 8, 11, 14]
    result = []
    for v in dict.fromkeys(candidates):
        b = np.full(n, v, dtype=int)
        try:
            if min_energy_time_allocation(b, ch)[0] <= ch.total_energy_j * (1 + 1e-9):
                result.append(b)
        except (ArithmeticError, ValueError):
            continue
    return result

def equal_bits_baseline(gains, ch):
    """OPTIONAL CONTROL: equal integer bitwidths, P4-optimal times/energies.

    Chooses highest feasible common bitwidth (rather than maximizing DG).
    """
    for val in range(MAX_BITS, MIN_BITS - 1, -1):
        b = np.full(len(gains), val)
        total, t, e = min_energy_time_allocation(b, ch)
        if total <= ch.total_energy_j * (1 + 1e-9):
            return {"bits": b, "energies": e, "times": t,
                    "dg": bits_at(gains, b)}
    return None





def make_plots(df, out_dir):
    """Each plot has three client curves per allocation scheme, for each metric."""
    if df.empty:
        return
    plots = [("bits", "Bits per feature"), ("power_w", "Transmit power (W)"),
             ("time_s", "Communication time (s)"), ("energy_j", "Energy (J)"),
             ("actual_snr_db", "Achieved received SNR (dB)")]
    for scheme, group in df.groupby("scheme"):
        safe = scheme.replace("-", "_")
        for field, ylabel in plots:
            fig, ax = plt.subplots(figsize=(7, 4.6))
            for cid, client in group.groupby("client_id"):
                client = client.sort_values("reference_snr_db")
                ax.plot(client.reference_snr_db, client[field], "o-", label=f"Client {cid}")
            ax.set(xlabel="Reference SNR (dB)", ylabel=ylabel,
                   title=f"{scheme}: {ylabel} versus reference SNR")
            ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / f"{safe}_{field}_vs_snr.png", dpi=180)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.6))
    for scheme, group in df.groupby("scheme"):
        curve = group.groupby("reference_snr_db")["system_avg_dg"].first().sort_index()
        ax.plot(curve.index, curve.values, "o-", label=scheme)
    ax.set(xlabel="Reference SNR (dB)", ylabel="Mean system DG", title="DG across allocation schemes")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "comparison_system_dg_vs_snr.png", dpi=180)
    plt.close(fig)



# ============================================================================
# OPTIONAL ACCURACY EVALUATION (not part of P1-P4 optimization).
# Reads saved held-out features and frozen private heads. Never uses test
# accuracy to SELECT allocations; it evaluates schemes AFTER optimization.
# ============================================================================


from RA_Classical import classical_fixed_time_waterfill

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-snrs-db", nargs="+", type=float, default=[-6,-3,0,3,6,9],
                        help="Reference SNR sweep points")
    parser.add_argument("--snr-offsets-db", nargs="+", type=float, default=None,
                        help="One fixed offset per client, e.g. -5 0 5")
    parser.add_argument("--reference-power-w", type=float, default=1.0,
                        help="Reference power for gamma = 10^((SNR+offset)/10)/Pref")
    parser.add_argument("--bandwidth-hz", type=float, default=1e6)
    parser.add_argument("--baseline-client-time-s", type=float, default=0.001024)
    parser.add_argument("--time-budget-s", type=float, default=None)
    parser.add_argument("--energy-budget-j", type=float, default=None,
                        help="Fixed for ALL SNR points; overrides budget-factor")
    parser.add_argument("--budget-factor", type=float, default=1.2,
                        help="If no explicit budget, scale 8-bit reference ENERGY ONCE at anchor SNR")
    parser.add_argument("--budget-anchor-snr-db", type=float, default=0.0,
                        help="SNR used ONLY to calculate the fixed energy budget")
    parser.add_argument("--initial-bits", type=int, default=3)
    parser.add_argument("--local-jumps", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--benchmark", action="store_true", help="P1 exhaustive global optimum, benchmark ONLY")
    parser.add_argument("--skip-sca", action="store_true", help="Diagnostic only: run baselines/benchmark without CVXPY")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--accuracy-mode", choices=["none","ideal","bpsk","both"], default="none",
                        help="Optional held-out accuracy evaluation AFTER allocations")
    parser.add_argument("--accuracy-repeats", type=int, default=5)
    parser.add_argument("--accuracy-seed", type=int, default=42)
    args = parser.parse_args()
    if not args.skip_sca:
        try:
            import cvxpy  # noqa: F401 - preflight check for proposed P3 optimizer
        except ImportError as ex:
            parser.error("SCA requires CVXPY. Install: pip install cvxpy clarabel; "
                         "or use --skip-sca for baseline/benchmark tests only")

    if not args.reference_snrs_db or not all(np.isfinite(args.reference_snrs_db)):
        parser.error("Specify finite reference SNR points")
    gains = load_gain_table(args.run_dir / "discriminant_gain_wa_table.csv")
    n = gains.shape[0]
    offsets = (np.zeros(n) if args.snr_offsets_db is None
               else np.array(args.snr_offsets_db, dtype=float))
    if offsets.shape != (n,) or not np.isfinite(offsets).all():
        parser.error(f"Expected exactly {n} finite SNR offsets")
    if (args.reference_power_w <= 0 or args.bandwidth_hz <= 0 or
        args.baseline_client_time_s <= 0 or args.budget_factor <= 0 or
        args.local_jumps < 1 or args.max_iterations < 1 or args.accuracy_repeats < 1 or
        args.initial_bits not in range(MIN_BITS,MAX_BITS+1)):
        parser.error("Invalid positive scalar parameter, local jumps, iterations, or initial bits")
    bandwidth = np.full(n, args.bandwidth_hz)
    total_time = (args.time_budget_s if args.time_budget_s is not None
                  else n * args.baseline_client_time_s)
    if total_time <= 0:
        parser.error("Time budget must be positive")
    gamma_anchor = np.power(10.0, (args.budget_anchor_snr_db + offsets) / 10) / args.reference_power_w
    reference_energy = baseline_energy_for_eight_bits(gamma_anchor, bandwidth, args.baseline_client_time_s)
    budget = (args.energy_budget_j if args.energy_budget_j is not None
              else args.budget_factor * reference_energy)
    if not np.isfinite(budget) or budget <= 0:
        parser.error("Total energy budget must be finite and positive")

    out_dir = args.run_dir / "RA_WaterFilling"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"FIXED sweep budgets: energy={budget:.10g} J, time={total_time:.10g} s")
    print(f"Offsets: {offsets.tolist()} dB | Reference SNR sweep: {args.reference_snrs_db}")
    rows = []; reports = []; summary = []; diagnostics = []
    for snr in args.reference_snrs_db:
        gamma = np.power(10.0, (snr + offsets) / 10) / args.reference_power_w
        ch = Channel(bandwidth, gamma, total_time, budget)
        case = {"reference_snr_db": float(snr), "client_gamma": gamma.tolist(),
                "energy_budget_j": budget, "time_budget_s": total_time}
        minimum3 = min_energy_time_allocation(np.full(n, MIN_BITS), ch)[0]
        if minimum3 > budget * (1 + 1e-9):
            case["status"] = "infeasible_at_min_bits"
            case["minimum_required_energy_j"] = minimum3
            print(f"SNR {snr:+g} dB: INFEASIBLE even at {MIN_BITS} bits/client; required {minimum3:.6g} J")
            reports.append(case)
            summary.append({"reference_snr_db":snr,"scheme":"all","status":case["status"]})
            continue
        print(f"\nSNR {snr:+g} dB | gamma={np.round(gamma,5).tolist()}")
        solutions = {}
        if args.skip_sca:
            proposed, seeds = None, []
        else:
            proposed, seeds = propose_allocation(gains, ch, args.initial_bits,
                                                 args.local_jumps, args.max_iterations, args.solver)
        case["SCA_seeds"] = seeds
        if proposed is not None:
            solutions["DG_SCA"] = proposed
            diag = continuous_kkt_diagnostic(proposed,gains,ch)
            case["SCA_kkt_diagnostic"] = diag
            for k in range(n):
                diagnostics.append({"reference_snr_db":float(snr), "client_id":k,
                    "beta":float(proposed["beta"][k]),
                    "interior_active":int(k in diag["interior_clients"]),
                    "marginal_energy":diag["marginal_energy"][k],
                    "marginal_time":diag["marginal_time"][k],
                    "energy_relative_spread":diag["relative_energy_spread"],
                    "time_relative_spread":diag["relative_time_spread"],
                    "last_bit_change":diag["converged_bit_change"]})
        elif not args.skip_sca:
            print("  WARNING: all SCA seeds failed; see sweep_report.json")
        rate_wf = classical_fixed_time_waterfill(gains,ch)
        if rate_wf is not None:
            solutions["Classical_fixed_time_WF"] = rate_wf
        equal = equal_bits_baseline(gains,ch)
        if equal is not None:
            solutions["Equal_bits_P4"] = equal
        if args.benchmark:
            benchmark, feasible_count = exact_benchmark(gains,ch)
            case["benchmark_feasible_count"] = feasible_count
            if benchmark is not None:
                solutions["P1_global_benchmark"] = {
                    "bits": np.array(benchmark["bits"],dtype=int),
                    "energies": np.array(benchmark["energies_j"]),
                    "times": np.array(benchmark["times_s"]),
                    "dg": benchmark["dg"]}
        for name, sol in solutions.items():
            verification = assert_solution_valid(sol,ch)
            append_solution_rows(rows,name,sol,gains,ch,snr,offsets,args.reference_power_w)
            summary.append({"reference_snr_db":float(snr),"scheme":name,"status":"ok",
                            "bits":str(sol["bits"].tolist()),"system_sum_dg":sol["dg"],
                            **verification})
            print(f"  {name}: bits={sol['bits'].tolist()} sumDG={sol['dg']:.3f}, "
                  f"energy={verification['energy_utilization']:.2%}")
        if "DG_SCA" in solutions and "P1_global_benchmark" in solutions:
            case["SCA_optimality_gap_pct"] = (100 * (solutions["P1_global_benchmark"]["dg"]
                - solutions["DG_SCA"]["dg"]) / abs(solutions["P1_global_benchmark"]["dg"]))
        case["status"] = ("ok" if proposed is not None else
                          "SCA_skipped" if args.skip_sca else "SCA_failed")
        reports.append(case)

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(["reference_snr_db","scheme","client_id"]).to_csv(out_dir / "allocation_snr_sweep.csv",index=False)
        if not args.no_plots:
            make_plots(df,out_dir)
        if args.accuracy_mode != "none":
            print("Evaluating held-out test accuracy WITHOUT using labels in allocation...")
            adf = accuracy_evaluation(df, args.run_dir, args.accuracy_mode,
                                      args.accuracy_repeats, args.accuracy_seed)
            adf.to_csv(out_dir / "client_accuracy_snr_sweep.csv",index=False)
            if not args.no_plots:
                accuracy_plots(adf,out_dir)
    if diagnostics:
        ddf = pd.DataFrame(diagnostics)
        ddf.to_csv(out_dir / "sca_kkt_diagnostics.csv",index=False)
        if not args.no_plots:
            fig, axes = plt.subplots(2, 1, figsize=(7,7), sharex=True)
            for metric,ax in [("energy_relative_spread",axes[0]),("time_relative_spread",axes[1])]:
                one = ddf.groupby("reference_snr_db")[metric].first().sort_index()
                ax.plot(one.index,one.values,"o-")
                ax.set(ylabel=metric.replace("_"," "),title="Interior-client marginal equalization")
                ax.grid(alpha=0.3)
            axes[1].set_xlabel("Reference SNR (dB)")
            fig.tight_layout(); fig.savefig(out_dir / "sca_kkt_marginal_spread.png",dpi=180)
            plt.close(fig)
    pd.DataFrame(summary).to_csv(out_dir / "summary_snr_sweep.csv",index=False)
    with (out_dir / "sweep_report.json").open("w",encoding="utf-8") as f:
        json.dump({"configuration":vars(args) | {"run_dir":str(args.run_dir)},
                   "fixed_energy_budget_j":budget,"fixed_time_budget_s":total_time,
                   "cases":reports},f,indent=2)
    print(f"\nSaved results to {out_dir.resolve()}")
    print("Accuracy experiments are separate: ideal Shannon capacity is NOT a BER model.")


if __name__ == "__main__":
    main()
