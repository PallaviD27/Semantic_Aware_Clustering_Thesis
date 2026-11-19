

def classical_fixed_time_waterfill(gains, ch):
    """CHANNEL-ONLY BASELINE (not proposed method): fixed equal T, classical WF.

    Maximize sum transmitted Shannon bits with constant T_k=Tmax/K and
    minimum/maximum allowed bitwidth payloads. DG is NEVER used to optimize.
    After continuous water filling, integer recovery chooses the cheapest
    incremental bit under fixed times. This baseline has a DIFFERENT time
    policy from the joint-time DG-aware method; compare with this caveat.
    """
    n = len(gains)
    t = np.full(n, ch.total_time_s / n)
    gamma, B = ch.gamma, ch.bandwidth_hz
    def energy_at_bits(bits):
        bits = np.asarray(bits, dtype=float)
        exponent = 128 * bits * math.log(2) / (B * t)
        if float(np.max(exponent)) > 700:
            return np.full(n, np.inf)
        return t / gamma * np.expm1(exponent)
    low_e = energy_at_bits(np.full(n, MIN_BITS))
    max_e = energy_at_bits(np.full(n, MAX_BITS))
    if float(sum(low_e)) > ch.total_energy_j * (1 + 1e-9):
        return None
    if sum(max_e) <= ch.total_energy_j:
        continuous_E = max_e.copy()
    else:
        # KKT for max sum T*B*log2(1+gamma*E/T) at equal fixed times.
        # marginal = B*gamma/[ln2*(1+gamma*E/T)]; WF energy level.
        def at_level(level):
            E = t * (B / (level * math.log(2)) - 1 / gamma)
            return np.clip(E, low_e, max_e)
        lower = 1e-16
        upper = float(np.max(B * gamma / math.log(2))) * 1e3
        for _ in range(120):
            mid = math.sqrt(lower * upper)
            if np.sum(at_level(mid)) > ch.total_energy_j:
                lower = mid
            else:
                upper = mid
        continuous_E = at_level(upper)
    continuous_bits = t * B / 128 * np.log2(1 + gamma * continuous_E / t)
    bits = np.clip(np.floor(continuous_bits + 1e-8), MIN_BITS, MAX_BITS).astype(int)
    # Rate-only integer recovery, NOT DG-based and NOT grid search.
    while True:
        base = float(np.sum(energy_at_bits(bits)))
        candidates = []
        for k in range(n):
            if bits[k] == MAX_BITS:
                continue
            step_bits = bits.copy()
            step_bits[k] += 1
            extra = float(np.sum(energy_at_bits(step_bits)) - base)
            if base + extra <= ch.total_energy_j * (1 + 1e-9):
                candidates.append((extra, k))
        if not candidates:
            break
        _, k = min(candidates)
        bits[k] += 1
    return {"bits": bits, "energies": energy_at_bits(bits),
            "times": t, "dg": bits_at(gains, bits),
            "continuous_bits": continuous_bits.tolist(),
            "note": "Fixed equal communication times; maximizes Shannon sum-rate, not DG"}