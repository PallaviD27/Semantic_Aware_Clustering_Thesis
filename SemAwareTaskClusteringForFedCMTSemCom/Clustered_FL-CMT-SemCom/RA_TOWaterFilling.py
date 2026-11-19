"""DG-aware joint bit/energy/time allocation for three (or more) clients.

P1: Original integer-bit problem (not solved by the proposed SCA method).
P2: Continuous bitwidth relaxation using PCHIP DG interpolants.
P3: Concave quadratic minorization of P2 solved iteratively using CVXPY.
P4: Minimum-energy time allocation for a fixed integer-bit vector.

The proposed algorithm is P2 -> repeated P3 -> P4-guided local integer recovery.
Optional exhaustive enumeration of P1 is ONLY a benchmark, never part of SCA.

Channel model here is ideal Shannon reliable digital transport, not the
uncoded noisy BPSK model used for some accuracy evaluation experiments.

Dependencies: numpy, pandas, scipy, cvxpy (SCA only).
"""


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


@dataclass
class Channel:
    bandwidth_hz: np.ndarray
    gamma: np.ndarray  # gamma = |h|^2 / sigma_rx^2, 1/W
    total_time_s: float
    total_energy_j: float


def load_gain_table(path: Path):
    """Uses empirical integer DG, not the interpolated DG, for final P1 comparisons."""
    df = pd.read_csv(path)
    required = {"client_id", "bits", "gain"}
    if not required.issubset(df.columns):
        raise ValueError(f"DG CSV must contain columns {sorted(required)}")
    if df.duplicated(["client_id", "bits"]).any():
        raise ValueError("Duplicate client_id/bits rows in DG CSV")
    clients = sorted(int(x) for x in df.client_id.unique())
    if clients != list(range(len(clients))):
        raise ValueError("Client IDs must be contiguous, starting at zero")
    bit_grid = np.arange(MIN_BITS, MAX_BITS + 1)
    gains = np.stack([
        df.loc[df.client_id == k].set_index("bits").reindex(bit_grid)["gain"].to_numpy(dtype=float)
        for k in clients
    ])
    if not np.isfinite(gains).all():
        raise ValueError("DG table missing bits 3..16 for one or more clients")
    return gains


def make_interpolators(gains):
    """P2: differentiable DG representation matching all integer table entries."""
    grid = np.arange(MIN_BITS, MAX_BITS + 1, dtype=float)
    interpolants = [PchipInterpolator(grid, row) for row in gains]
    curvature = []
    for pchip in interpolants:
        # PCHIP is C1, with piecewise-linear second derivative. A bound on
        # |G''| is found at each interval's two one-sided endpoints.
        sample = []
        for b in range(MIN_BITS, MAX_BITS):
            sample.extend([b + 1e-8, b + 1 - 1e-8])
        L = float(np.max(np.abs(pchip.derivative(2)(sample))))
        curvature.append(max(1.05 * L, 1e-8))
    return interpolants, np.asarray(curvature)


def bits_at(gains, b):
    b = np.asarray(b, dtype=int)
    return float(np.sum(gains[np.arange(len(b)), b - MIN_BITS]))


def _q_inverse(x):
    """Invert q(z)=1+(z-1)exp(z) for z>0; q is increasing."""
    x = np.asarray(x, dtype=float)
    z = np.empty_like(x)
    small = x < 1e-7
    # q(z) ~ z^2/2 for z close to zero; avoids branch-point cancellation.
    z[small] = np.sqrt(2 * x[small])
    if np.any(~small):
        arg = (x[~small] - 1) / math.e
        z[~small] = 1 + np.real(lambertw(arg, k=0))
    return np.maximum(z, 1e-14)


def min_energy_time_allocation(b, ch: Channel):
    """P4: globally solve convex minimum-energy time allocation for fixed b.

    Uses scalar dual bisection rather than running a generic optimizer for
    each candidate. All users transmit positive bits, so optimal time uses
    the full shared time budget under this ideal Shannon model.

    Normalized energy e=E/Ebudget and time tau=T/Tmax:
      e_k(tau)=tau/a_k * exp(c_k/tau)-tau/a_k
      a_k=gamma_k*Ebudget/Tmax, c_k=128*b_k*ln(2)/(B_k*Tmax).
    KKT of P4 gives lambda=a_k^-1 [1+(z-1)exp(z)], z=c_k/tau_k.
    """
    b = np.asarray(b, dtype=float)
    if np.any(b < MIN_BITS) or np.any(b > MAX_BITS):
        raise ValueError("Bitwidth outside 3..16")
    a = ch.gamma * ch.total_energy_j / ch.total_time_s
    c = BITS_PER_FEATURE_VECTOR * b * math.log(2) / (ch.bandwidth_hz * ch.total_time_s)
    if np.any(a <= 0) or np.any(c <= 0):
        raise ValueError("Invalid channel parameters or bitwidth")

    def tau_given_dual(lam):
        return c / _q_inverse(lam * a)

    low, high = 1e-22, 1.0
    # At low lambda, sum(tau) should be greater than 1.
    while float(np.sum(tau_given_dual(low))) < 1:
        low /= 100
        if low < 1e-290:
            raise ArithmeticError("Could not bracket P4 dual variable (low)")
    while float(np.sum(tau_given_dual(high))) > 1:
        high *= 10
        if high > 1e290:
            raise ArithmeticError("Could not bracket P4 dual variable (high)")

    lam = brentq(lambda u: float(np.sum(tau_given_dual(math.exp(u))) - 1),
                 math.log(low), math.log(high), xtol=1e-12)
    tau = tau_given_dual(math.exp(lam))
    tau = tau / np.sum(tau)  # clean up minor floating point root residual
    z = c / tau
    if np.max(z) > 700:
        return float("inf"), tau * ch.total_time_s, np.full_like(tau, np.inf)
    energy_normalized = tau * np.expm1(z) / a
    energies = energy_normalized * ch.total_energy_j
    return float(np.sum(energies)), tau * ch.total_time_s, energies


def solve_sca(gains, ch: Channel, seed_bits, max_iterations=100,
              tolerance=1e-6, solver_name="CLARABEL"):
    """P2/P3: solve continuous relaxation by successive convex minorization.

    NO BITWIDTH GRID SEARCH HERE. This is the proposed algorithm's core.
    Each P3 is a convex program; KKT of P3 yields the implicit DG-aware
    water-filling relationship discussed in the mathematical derivation.
    """
    try:
        import cvxpy as cp
    except ImportError as ex:
        raise RuntimeError("Install CVXPY for SCA: pip install cvxpy clarabel") from ex

    n = gains.shape[0]
    interp, L = make_interpolators(gains)
    b0 = np.asarray(seed_bits, dtype=int)
    e0, t0, energies0 = min_energy_time_allocation(b0, ch)
    if e0 > ch.total_energy_j * (1 + 1e-7):
        raise ValueError(f"Initial bits {b0.tolist()} are infeasible")

    beta0 = b0.astype(float)
    energy_norm0 = energies0 / ch.total_energy_j
    time_norm0 = t0 / ch.total_time_s
    a = ch.gamma * ch.total_energy_j / ch.total_time_s
    capacity_factor = ch.total_time_s * ch.bandwidth_hz / math.log(2)
    history = []

    for iteration in range(max_iterations):
        # P3(n): concave quadratic LOWER approximation of P2 objective.
        beta = cp.Variable(n)
        e = cp.Variable(n, nonneg=True)      # E_k / E_budget
        tau = cp.Variable(n, nonneg=True)    # T_k / T_max
        d = np.array([float(p.derivative()(beta0[k])) for k, p in enumerate(interp)])
        obj = cp.sum(cp.multiply(d, beta - beta0)
                     - 0.5 * cp.multiply(L, cp.square(beta - beta0)))

        # Perspective log: -rel_entr(tau, tau+a*e)/ln(2)
        # This is concave in (e,tau) and CVXPY recognizes the DCP form.
        cap = cp.multiply(capacity_factor,
                          -cp.rel_entr(tau, tau + cp.multiply(a, e)))
        constraints = [
            beta >= MIN_BITS, beta <= MAX_BITS,
            cp.sum(e) <= 1,
            cp.sum(tau) <= 1,
            BITS_PER_FEATURE_VECTOR * beta <= cap,
        ]
        problem = cp.Problem(cp.Maximize(obj), constraints)
        try:
            problem.solve(solver=solver_name, warm_start=True,
                          verbose=False)
        except cp.error.SolverError as ex:
            raise RuntimeError(f"P3 solver failed: {ex}") from ex
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"P3 did not solve: {problem.status}")
        beta_next = np.asarray(beta.value, dtype=float)
        e_next = np.asarray(e.value, dtype=float)
        tau_next = np.asarray(tau.value, dtype=float)
        if not np.isfinite(beta_next).all():
            raise RuntimeError("P3 returned nonfinite bitwidths")
            # Numerical check of P3's constraints using original physical capacity.
        tau_safe = np.maximum(tau_next, 1e-14)
        capacity = ch.total_time_s * ch.bandwidth_hz * tau_safe * np.log2(
            1 + a * np.maximum(e_next, 0) / tau_safe)
        violation = max(
            float(np.max(BITS_PER_FEATURE_VECTOR * beta_next - capacity)),
            float(np.sum(e_next) - 1), float(np.sum(tau_next) - 1),
            float(np.max(MIN_BITS - beta_next)),
            float(np.max(beta_next - MAX_BITS)), 0.0)
        if violation > 2e-3:  # capacity tolerance is in transmitted bits
            raise RuntimeError(f"P3 feasibility error: {violation:.3g}")
        objective_actual = sum(float(interp[k](beta_next[k])) for k in range(n))
        history.append({"iteration": iteration + 1,
                        "dg_interpolated": objective_actual,
                        "max_bit_change": float(np.max(np.abs(beta_next - beta0))),
                        "beta": beta_next.tolist()})
        if len(history) > 1 and objective_actual < history[-2]["dg_interpolated"] - 1e-4:
            raise RuntimeError("SCA decreased objective; check solver precision")
        change = float(np.max(np.abs(beta_next - beta0)))
        beta0, energy_norm0, time_norm0 = beta_next, e_next, tau_next
        if change < tolerance:
            break
    return beta0, energy_norm0 * ch.total_energy_j, time_norm0 * ch.total_time_s, history

def integer_recovery(beta, gains, ch: Channel, local_jumps=2):
    """P1 integer recovery guided by P4; NOT exhaustive enumeration.

    Start with floor(beta), guaranteed feasible if beta came from a feasible
    P2 solution. Then accept the best feasible positive-DG local move among
    one-client +1..+local_jumps, and optionally exchange moves (+j,-j).
    This is a LOCAL search (heuristic), not global grid search.
    """
    b = np.clip(np.floor(beta + 1e-9), MIN_BITS, MAX_BITS).astype(int)
    memo = {}

    def p4(x):
        key = tuple(int(v) for v in x)
        if key not in memo:
            memo[key] = min_energy_time_allocation(x, ch)
        return memo[key]

    initial_energy, _, _ = p4(b)
    if initial_energy > ch.total_energy_j * (1 + 1e-6):
        raise RuntimeError("Rounded allocation is not feasible: check SCA numerical tolerance")

    trajectory = []
    while True:
        base_dg = bits_at(gains, b)
        base_energy = p4(b)[0]
        candidates = set()
        n = len(b)
        for k in range(n):
            for step in range(1, local_jumps + 1):
                if b[k] + step <= MAX_BITS:
                    v = b.copy();
                    v[k] += step
                    candidates.add(tuple(v))
                # A bit exchange can escape a local stall while holding or
                # lowering overall bit count. Not an exhaustive search.
                for j in range(n):
                    if j == k or b[k] + step > MAX_BITS or b[j] - step < MIN_BITS:
                        continue
                    v = b.copy();
                    v[k] += step;
                    v[j] -= step
                    candidates.add(tuple(v))
        best = None
        best_score = -float("inf")
        for cand in sorted(candidates):
            cand = np.asarray(cand, dtype=int)
            delta = bits_at(gains, cand) - base_dg
            if delta <= 1e-9:
                continue
            energy, _, _ = p4(cand)
            if energy > ch.total_energy_j * (1 + 1e-8):
                continue
            extra_energy = max(energy - base_energy, 0)
            # Primary goal: higher DG. Efficiency breaks near-equal ties.
            score = delta
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            break
        b = best
        trajectory.append({"bits": b.tolist(), "dg": bits_at(gains, b),
                           "energy_j": p4(b)[0]})
    energy, times, energies = p4(b)
    return b, energies, times, trajectory


def propose_allocation(gains, ch, initial_bits, local_jumps, iterations, solver):
    """P2/P3 SCA and P4 local integer recovery. NEVER calls P1 enumeration."""
    best = None
    records = []
    for seed in select_feasible_seeds(ch, len(gains), initial_bits):
        try:
            beta, continuous_E, continuous_T, history = solve_sca(
                gains, ch, seed, max_iterations=iterations, solver_name=solver)
            b, energies, times, recovery = integer_recovery(
                beta, gains, ch, local_jumps=local_jumps)
            val = bits_at(gains, b)
            records.append({"seed": seed.tolist(), "beta": beta.tolist(),
                            "integer_bits": b.tolist(), "sum_dg": val,
                            "sca_iterations": len(history), "recovery_steps": len(recovery),
                            "status": "ok"})
            if best is None or val > best["dg"]:
                best = {"bits": b, "energies": energies, "times": times,
                        "dg": val, "beta": beta, "continuous_E": continuous_E,
                        "continuous_T": continuous_T, "history": history,
                        "recovery": recovery, "seed": seed.tolist()}
        except (RuntimeError, ValueError, ArithmeticError, OverflowError) as exc:
            records.append({"seed": seed.tolist(), "status": "failed", "error": str(exc)})
    return best, records


def assert_solution_valid(sol, ch):
    """Numerically verify P1 feasibility and physical payload capacity."""
    b = np.asarray(sol["bits"], dtype=int)
    E = np.asarray(sol["energies"], dtype=float)
    T = np.asarray(sol["times"], dtype=float)
    if (np.any((b < MIN_BITS) | (b > MAX_BITS)) or np.any(E <= 0)
            or np.any(T <= 0) or np.any(~np.isfinite(E))
            or np.any(~np.isfinite(T))):
        raise AssertionError("Nonpositive or nonfinite resource allocation")
    capacity = T * ch.bandwidth_hz * np.log2(1 + ch.gamma * E / T)
    violations = np.maximum(128 * b - capacity, 0)
    if (float(sum(E)) > ch.total_energy_j * (1 + 2e-6)
            or float(sum(T)) > ch.total_time_s * (1 + 2e-6)
            or float(np.max(violations)) > 5e-3):
        raise AssertionError("Resource allocation violates P1 constraints")
    return {"max_payload_violation_bits": float(np.max(violations)),
            "energy_utilization": float(sum(E) / ch.total_energy_j),
            "time_utilization": float(sum(T) / ch.total_time_s)}


def continuous_kkt_diagnostic(sol, gains, ch):
    """Stationarity diagnostic for converged P2, NEVER for integer bitwidths.

    Equation from Step 5 (for clients with STRICT interior beta, positive E/T,
    and binding capacity): M_E ~ lambda, M_T ~ nu. A small residual alone
    doesn't prove optimality; check convergence, complementarity, and bounds.
    """
    beta = np.asarray(sol["beta"])
    E = np.asarray(sol["continuous_E"])
    T = np.asarray(sol["continuous_T"])
    interp, _ = make_interpolators(gains)
    deriv = np.array([float(p.derivative()(beta[k])) for k, p in enumerate(interp)])
    snr = ch.gamma * E / np.maximum(T, 1e-30)
    ME = deriv * ch.bandwidth_hz * ch.gamma / (128 * math.log(2) * (1 + snr))
    MT = deriv * ch.bandwidth_hz / (128 * math.log(2)) * (np.log1p(snr) - snr / (1 + snr))
    capacity = T * ch.bandwidth_hz * np.log2(1 + snr)
    interior = (beta > MIN_BITS + 1e-3) & (beta < MAX_BITS - 1e-3) & (np.abs(capacity - 128 * beta) < 0.5)
    indices = np.flatnonzero(interior)
    def spread(arr):
        if len(indices) < 2:
            return None
        values = arr[indices]
        return float((values.max() - values.min()) / max(np.mean(np.abs(values)), 1e-14))
    return {"interior_clients": indices.tolist(),
            "marginal_energy": ME.tolist(), "marginal_time": MT.tolist(),
            "relative_energy_spread": spread(ME),
            "relative_time_spread": spread(MT),
            "converged_bit_change": sol["history"][-1]["max_bit_change"] if sol["history"] else None,
            "note": "Equalization expected only for converged, interior, active-capacity clients. Not applicable to rounded bits."}


def append_solution_rows(all_rows, scheme, sol, gains, ch, snr, offsets, budget_ref):
    for k, b in enumerate(sol["bits"]):
        E = float(sol["energies"][k]); T = float(sol["times"][k]); P = E / T
        all_rows.append({
            "scheme": scheme, "reference_snr_db": float(snr),
            "client_id": int(k), "offset_db": float(offsets[k]),
            "client_reference_snr_db": float(snr + offsets[k]),
            "reference_power_w": float(budget_ref), "gamma_per_w": float(ch.gamma[k]),
            "bits": int(b), "payload_bits": int(128 * b),
            "power_w": P, "energy_j": E, "time_s": T,
            "capacity_bits": float(T * ch.bandwidth_hz[k] * np.log2(1 + ch.gamma[k] * P)),
            "actual_snr_db": float(10 * np.log10(ch.gamma[k] * P)),
            "dg": float(gains[k, int(b) - MIN_BITS]),
            "system_sum_dg": float(sol["dg"]),
            "system_avg_dg": float(sol["dg"] / len(gains)),
            "energy_budget_j": float(ch.total_energy_j),
            "time_budget_s": float(ch.total_time_s),
        })