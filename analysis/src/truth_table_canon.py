#!/usr/bin/env python3
"""ABC-faithful truth-table canonicalization.

Implements the canonical form ABC's mapper uses for supergate lookup:
the minimum truth-table integer over all 2^n input-phase assignments
(Map_CanonComputeSlow / Map_CanonComputePhase in mapperCanon.c).
Inputs are negated; inputs are NOT permuted and the output is NOT
complemented, so groups are keyed by (n_vars, canonical_tt).

Run `python3 truth_table_canon.py` to self-test against a brute-force
reference implementation.
"""

_VAR_MASKS_CACHE = {}


def var_masks(n_vars):
    """ABC uTruths[v]: positions i in [0, 2^n) whose bit `v` is set."""
    if n_vars not in _VAR_MASKS_CACHE:
        _VAR_MASKS_CACHE[n_vars] = [
            sum((1 << i) for i in range(1 << n_vars) if (i >> v) & 1)
            for v in range(n_vars)
        ]
    return _VAR_MASKS_CACHE[n_vars]


def phase_transform(tt, n_vars, phase, masks=None):
    """Apply input-phase assignment `phase` (bit v set = negate input v).

    Mirrors the C bit-twiddle in Map_CanonComputePhase: positions where
    bit v is set shift right by 2^v, positions where it is clear shift
    left by 2^v (an exact swap of minterms i and i ^ (1 << v)).
    """
    if masks is None:
        masks = var_masks(n_vars)
    for v in range(n_vars):
        if (phase >> v) & 1:
            shift = 1 << v
            mask = masks[v]
            tt = ((tt & ~mask) << shift) | ((tt & mask) >> shift)
    return tt


def canonicalize(tt, n_vars):
    """Minimum TT integer over all input-phase assignments."""
    masks = var_masks(n_vars)
    best = tt
    for phase in range(1, 1 << n_vars):
        t = phase_transform(tt, n_vars, phase, masks)
        if t < best:
            best = t
    return best


class Canonicalizer:
    """Caching canonicalizer; groups keyed by (n_vars, canonical_tt)."""

    def __init__(self):
        self._cache = {}

    def key(self, tt, n_vars):
        """Return the (n_vars, canonical_tt) group key for a raw TT."""
        cache_key = (tt, n_vars)
        entry = self._cache.get(cache_key)
        if entry is None:
            entry = (n_vars, canonicalize(tt, n_vars))
            self._cache[cache_key] = entry
        return entry


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _build_tt(n_vars, on_minterms):
    tt = 0
    for m in on_minterms:
        tt |= 1 << m
    return tt


def _reference_phase(tt, n_vars, phase):
    """Brute-force reference: negating input v maps minterm i -> i ^ (1<<v)."""
    out = 0
    for i in range(1 << n_vars):
        j = i
        for v in range(n_vars):
            if (phase >> v) & 1:
                j ^= 1 << v
        if (tt >> j) & 1:
            out |= 1 << i
    return out


def _test_phase_transform_matches_reference():
    for n in range(2, 6):
        tests = [_build_tt(n, [0]), _build_tt(n, [1, 2]), _build_tt(n, range(1 << n))]
        for tt in tests:
            for phase in range(1 << n):
                assert phase_transform(tt, n, phase) == _reference_phase(tt, n, phase), (n, hex(tt), phase)


def _test_canonicalize_is_min_over_phases():
    # minterms 3,7 => f = x0 & x1 (x2 is a don't-care)
    tt = _build_tt(3, [3, 7])
    expected = min(_reference_phase(tt, 3, p) for p in range(8))
    assert canonicalize(tt, 3) == expected
    for phase in range(8):
        variant = _reference_phase(tt, 3, phase)
        assert canonicalize(variant, 3) == expected


def _test_symmetric_xor():
    xor3 = _build_tt(3, [1, 2, 4, 7])
    xnor3 = _build_tt(3, [0, 3, 5, 6])
    expected = min(xor3, xnor3)
    assert canonicalize(xor3, 3) == expected
    assert canonicalize(xnor3, 3) == expected


def _test_and2_canonical_is_one():
    # f = x0&x1 has TT 0x8; all phase variants are single-minterm functions,
    # so the minimum is 0x1 (the constant-on-one-minterm 0).
    assert canonicalize(0x8, 2) == 0x1


def _test_same_integer_different_arity():
    # 2-input AND and the 3-input function x0&x1&~x2 both have TT 0x8,
    # but canonicalize to different groups because arity differs.
    assert canonicalize(0x8, 2) == 0x1
    assert canonicalize(0x8, 3) == 0x1
    c = Canonicalizer()
    assert c.key(0x8, 2) != c.key(0x8, 3)


def _test_canonicalizer_cache_consistency():
    c = Canonicalizer()
    tt = _build_tt(5, [1, 5, 9, 13])
    assert c.key(tt, 5) == c.key(tt, 5)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
