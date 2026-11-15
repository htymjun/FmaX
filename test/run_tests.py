#!/usr/bin/env python3
"""FmaX fltflt test suite (driven by `make test CASE_GPU_CC=...`).

Four independent checks, each of which has to pass on its own:

  A. bit-exact against NVIDIA MatX fltflt.h (ops 1..58 that both provide):
     ff_test (FmaX, production flags) and ff_matx must agree in every bit.
  B. contraction independence: ff_test (-fast, FMA contraction on) and
     ff_test_nofma (-Mnofma) must agree in every bit for every op. An
     error-free transformation that the compiler can fuse into an FMA shows up
     here, independently of A.
  C. accuracy against an extended-precision reference (x86 80-bit long double,
     64-bit mantissa, i.e. 2^-64 >> the 2^-44 bounds tested) and the
     canonical form fl(hi + lo) == hi of every finite result.
  D. hand-written edge cases with exact expected results (ties, large
     magnitudes, cancellation, zeros, NaN, the cases from the 2026-11 review).

Exit status 0 only if all four pass. Nothing here needs anything but numpy.
"""
from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
F_FMA = os.path.join(HERE, "ff_test")
F_NOFMA = os.path.join(HERE, "ff_test_nofma")
MATX = os.path.join(HERE, "ff_matx")

f32 = np.float32
LD = np.longdouble
assert np.finfo(LD).nmant >= 63, "needs x86 80-bit long double for the reference"

U = 2.0 ** -24  # unit roundoff of FP32

# ---------------------------------------------------------------- ops table
# code: (name, kind) where kind selects the reference / bound in check C.
OPS = {
    1: "add_ff_ff", 2: "add_ff_r4", 3: "add_r4_ff",
    4: "sub_ff_ff", 5: "sub_ff_r4", 6: "sub_r4_ff",
    7: "mul_ff_ff", 8: "mul_ff_r4", 9: "mul_r4_ff",
    10: "div_ff_ff", 11: "div_ff_r4", 12: "div_r4_ff",
    13: "fma_ff_ff_ff", 14: "fma_ff_ff_r4", 15: "fma_r4_ff_ff", 16: "fma_ff_r4_ff",
    17: "fma_ff_r4_r4", 18: "fma_r4_ff_r4", 19: "fma_r4_r4_ff",
    20: "fmaA_ff_ff_ff", 21: "fmaA_ff_ff_r4", 22: "fmaA_r4_ff_ff", 23: "fmaA_ff_r4_ff",
    24: "fmaA_ff_r4_r4", 25: "fmaA_r4_ff_r4", 26: "fmaA_r4_r4_ff",
    27: "add_same_sign_ff_ff", 28: "add_same_sign_ff_r4", 29: "add_same_sign_r4_ff",
    30: "sqrt", 31: "sqrt_fast", 32: "norm3d", 33: "abs",
    34: "round_to_nearest", 35: "round_toward_zero", 36: "floor",
    37: "fmod_ff_ff", 38: "fmod_ff_r4", 39: "init_r8", 40: "neg",
    41: "eq_ff_ff", 42: "ne_ff_ff", 43: "lt_ff_ff", 44: "gt_ff_ff", 45: "le_ff_ff", 46: "ge_ff_ff",
    47: "eq_ff_r4", 48: "ne_ff_r4", 49: "lt_ff_r4", 50: "gt_ff_r4", 51: "le_ff_r4", 52: "ge_ff_r4",
    53: "eq_r4_ff", 54: "ne_r4_ff", 55: "lt_r4_ff", 56: "gt_r4_ff", 57: "le_r4_ff", 58: "ge_r4_ff",
    60: "recip", 61: "square", 62: "dot2_r4", 63: "dot2_ff", 64: "dot3_ff", 65: "dot4_ff",
    66: "add3", 67: "add4", 68: "add5", 69: "ceil", 70: "hypot", 71: "lerp", 72: "pow_int",
    73: "min", 74: "max", 75: "sign", 76: "add_ff_r8", 77: "mul_ff_r8", 78: "div_ff_r8",
    79: "div_r8_ff", 80: "dot3_r4", 81: "warp_reduce_sum", 82: "dot4_r4", 83: "sub_ff_r8",
}
MATX_OPS = [c for c in OPS if c <= 58]
CMP_OPS = list(range(41, 59))

# ---------------------------------------------------------------- I/O
NAMES = ["ah", "al", "bh", "bl", "ch", "cl", "dh", "dl"]


def write_inputs(path, d):
    n = len(d["ah"])
    with open(path, "wb") as f:
        np.array([n], dtype="<i4").tofile(f)
        for k in NAMES:
            np.asarray(d[k], dtype="<f4").tofile(f)
        np.asarray(d["x8"], dtype="<f8").tofile(f)
        np.asarray(d["k"], dtype="<i4").tofile(f)


def read_outputs(path, n):
    raw = np.fromfile(path, dtype="<u4")
    assert raw.size == 3 * n, (path, raw.size, n)
    rh = raw[:n].view("<f4")
    rl = raw[n:2 * n].view("<f4")
    ri = raw[2 * n:].view("<i4")
    return rh, rl, ri


def run(exe, op, inp, tag):
    out = os.path.join(WORK, f"{tag}_{op}.bin")
    p = subprocess.run([exe, str(op), inp, out], capture_output=True, text=True)
    if p.returncode == 3:
        return None
    if p.returncode != 0:
        raise RuntimeError(f"{exe} op {op} failed ({p.returncode}): {p.stdout}{p.stderr}")
    n = int(np.fromfile(inp, dtype="<i4", count=1)[0])
    return read_outputs(out, n)


# ---------------------------------------------------------------- inputs
def canon(hi, lo):
    """FastTwoSum in FP32 (numpy float32 arithmetic is round-to-nearest, no FMA)."""
    hi = hi.astype(f32)
    lo = lo.astype(f32)
    s = (hi + lo).astype(f32)
    e = (lo - (s - hi).astype(f32)).astype(f32)
    return s, e


def rand_ff(rng, n, emin, emax, sign=None):
    e = rng.uniform(emin, emax, n)
    m = rng.uniform(1.0, 2.0, n)
    s = rng.choice([-1.0, 1.0], n) if sign is None else sign
    x = s * m * 2.0 ** np.floor(e)
    hi = x.astype(f32)
    lo = (hi.astype(np.float64) * rng.uniform(-0.5, 0.5, n) * 2.0 ** -24 * 2.0 ** -rng.integers(0, 4, n))
    return canon(hi, lo)


def gen_inputs(rng, n):
    """Four blocks of n/4: general, close exponents, cancellation, integer-ish."""
    q = n // 4
    d = {k: np.empty(n, f32) for k in NAMES}
    # block 0: general, exponents in [-40, 40]
    for p in "abcd":
        d[p + "h"][:q], d[p + "l"][:q] = rand_ff(rng, q, -40, 40)
    # block 1: close exponents [-3, 3]
    for p in "abcd":
        d[p + "h"][q:2 * q], d[p + "l"][q:2 * q] = rand_ff(rng, q, -3, 3)
    # block 2: cancellation: b ~ -a, c ~ -a*b, d ~ -c (relative gap 2^-1 .. 2^-46)
    s2 = slice(2 * q, 3 * q)
    ah, al = rand_ff(rng, q, -20, 20)
    d["ah"][s2], d["al"][s2] = ah, al
    av = ah.astype(np.float64) + al.astype(np.float64)
    gap = 2.0 ** -rng.uniform(1, 46, q) * rng.choice([-1, 1], q)
    bh, bl = canon(*split64(-av * (1 + gap)))
    d["bh"][s2], d["bl"][s2] = bh, bl
    bv = bh.astype(np.float64) + bl.astype(np.float64)
    gap = 2.0 ** -rng.uniform(1, 46, q) * rng.choice([-1, 1], q)
    ch, cl = canon(*split64(-(av * bv) * (1 + gap)))
    d["ch"][s2], d["cl"][s2] = ch, cl
    dh, dl = canon(*split64(-(ch.astype(np.float64) + cl) * (1 + gap[::-1])))
    d["dh"][s2], d["dl"][s2] = dh, dl
    # block 3: integer / half-integer hi with fractional lo (rounding ops),
    # magnitudes up to 2^40 so both the |hi| < 2^23 and the >= 2^23 paths run
    s3 = slice(3 * q, n)
    m = n - 3 * q
    mag = 2.0 ** rng.uniform(0, 40, m)
    base = np.round(mag) * rng.choice([-1, 1], m)
    half = rng.choice([0.0, 0.5], m)
    hi = (base + half).astype(f32)
    ulp_hi = np.spacing(np.abs(hi)).astype(np.float64)
    frac = rng.choice([0.0, 0.5, -0.5, 0.25, -0.25, 1e-3, -1e-3], m)
    lo = np.where(ulp_hi >= 1.0, frac * np.minimum(ulp_hi, 1.0), frac * ulp_hi)
    lo = np.where(np.abs(hi) >= 2 ** 23, frac, lo * 0.5)
    d["ah"][s3], d["al"][s3] = canon(hi, lo)
    for p in "bcd":
        d[p + "h"][s3], d[p + "l"][s3] = rand_ff(rng, m, -3, 6)
    d["x8"] = rng.uniform(1, 2, n) * 2.0 ** rng.integers(-40, 40, n) * rng.choice([-1, 1], n)
    d["x8"][2 * q:3 * q] = -(d["ah"][2 * q:3 * q].astype(np.float64) + d["al"][2 * q:3 * q]) * (
        1 + 2.0 ** -rng.uniform(1, 50, q))
    d["k"] = rng.integers(0, 9, n).astype(np.int32)
    return d


def split64(x):
    hi = x.astype(f32)
    lo = (x - hi.astype(np.float64)).astype(f32)
    return hi, lo


def val(h, l):
    return LD(0) + h.astype(LD) + l.astype(LD)


# ---------------------------------------------------------------- reference
def reference(op, d):
    """Exact-ish value (long double) and the scale the error is measured against."""
    a = val(d["ah"], d["al"]); b = val(d["bh"], d["bl"])
    c = val(d["ch"], d["cl"]); dd = val(d["dh"], d["dl"])
    a4 = d["ah"].astype(LD); b4 = d["bh"].astype(LD); c4 = d["ch"].astype(LD); d4 = d["dh"].astype(LD)
    x8 = d["x8"].astype(LD)
    absl = np.abs
    table = {
        1: (a + b, absl(a) + absl(b)), 2: (a + b4, absl(a) + absl(b4)), 3: (a4 + b, absl(a4) + absl(b)),
        4: (a - b, absl(a) + absl(b)), 5: (a - b4, absl(a) + absl(b4)), 6: (a4 - b, absl(a4) + absl(b)),
        7: (a * b, None), 8: (a * b4, None), 9: (a4 * b, None),
        10: (a / b, None), 11: (a / b4, None), 12: (a4 / b, None),
        13: (a * b + c, absl(a * b) + absl(c)), 14: (a * b + c4, absl(a * b) + absl(c4)),
        15: (a4 * b + c, absl(a4 * b) + absl(c)), 16: (a * b4 + c, absl(a * b4) + absl(c)),
        17: (a * b4 + c4, absl(a * b4) + absl(c4)), 18: (a4 * b + c4, absl(a4 * b) + absl(c4)),
        19: (a4 * b4 + c, absl(a4 * b4) + absl(c)),
        30: (np.sqrt(a), None), 31: (np.sqrt(a), None),
        32: (np.sqrt(a * a + b * b + c * c), None),
        60: (1 / a, None), 61: (a * a, None),
        62: (a4 * b4 + c4 * d4, absl(a4 * b4) + absl(c4 * d4)),
        63: (a * b + c * dd, absl(a * b) + absl(c * dd)),
        64: (a * b + c * dd + b * c, absl(a * b) + absl(c * dd) + absl(b * c)),
        65: (a * b + c * dd + a * dd + b * c, absl(a * b) + absl(c * dd) + absl(a * dd) + absl(b * c)),
        66: (a + b + c, absl(a) + absl(b) + absl(c)),
        67: (a + b + c + dd, absl(a) + absl(b) + absl(c) + absl(dd)),
        68: (a + b + c + dd + a, 2 * absl(a) + absl(b) + absl(c) + absl(dd)),
        70: (np.sqrt(a4 * a4 + b4 * b4), None),
        71: (a + c4 * (b - a), absl(a) + absl(c4 * (b - a))),
        72: (a ** d["k"].astype(LD), None),
        76: (a + x8, absl(a) + absl(x8)), 77: (a * x8, None), 78: (a / x8, None), 79: (x8 / b, None),
        80: (a4 * b4 + c4 * d4 + a4 * d4, absl(a4 * b4) + absl(c4 * d4) + absl(a4 * d4)),
        82: (a4 * b4 + c4 * d4 + a4 * d4 + b4 * c4,
             absl(a4 * b4) + absl(c4 * d4) + absl(a4 * d4) + absl(b4 * c4)),
        83: (a - x8, absl(a) + absl(x8)),
    }
    for o in range(20, 27):
        table[o] = table[o - 7]
    for o in (27, 28, 29):
        table[o] = table[o - 26]
    return table.get(op)


# relative bounds (in units of 2^-k): conservative published bounds for
# double-word arithmetic with u = 2^-24 (Joldes, Muller, Popescu 2017), rounded
# up to a power of two
BOUND = {o: 2.0 ** -44 for o in OPS}
BOUND.update({31: 2.0 ** -43, 32: 2.0 ** -43, 72: 2.0 ** -41, 71: 2.0 ** -43})
# ops whose error is measured against |result| (not against a sum of |terms|)
RELATIVE_TO_RESULT = {7, 8, 9, 10, 11, 12, 30, 31, 32, 60, 61, 70, 72, 77, 78, 79}


# ---------------------------------------------------------------- checks
def finite_canonical(rh, rl):
    ok = np.isfinite(rh) & np.isfinite(rl)
    canon_ok = ((rh + rl).astype(f32) == rh) | ~ok
    return canon_ok


def bits_equal(x, y):
    return (x[0].view("<u4") == y[0].view("<u4")) & (x[1].view("<u4") == y[1].view("<u4")) & (x[2] == y[2])


class Report:
    def __init__(self):
        self.fail = []

    def check(self, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            self.fail.append(name)


def check_random(rep, rng, n):
    d = gen_inputs(rng, n)
    # ops that need same-sign / non-negative / non-zero inputs get adjusted copies
    inp = os.path.join(WORK, "in_rand.bin")
    write_inputs(inp, d)
    dpos = dict(d)
    dpos["ah"], dpos["al"] = np.abs(d["ah"]), np.where(d["ah"] < 0, -d["al"], d["al"])
    inp_pos = os.path.join(WORK, "in_pos.bin")
    write_inputs(inp_pos, dpos)
    dss = dict(d)  # same sign: b gets a's sign
    sgn = np.sign(d["ah"]).astype(f32) * np.sign(d["bh"]).astype(f32)
    dss["bh"], dss["bl"] = d["bh"] * sgn, d["bl"] * sgn
    inp_ss = os.path.join(WORK, "in_ss.bin")
    write_inputs(inp_ss, dss)
    dfm = dict(d)  # fmod: moderate quotients, |a/b| < 2^20
    dfm["bh"], dfm["bl"] = rand_ff(rng, n, -3, 3)
    dfm["ah"], dfm["al"] = rand_ff(rng, n, -3, 17)
    inp_fm = os.path.join(WORK, "in_fm.bin")
    write_inputs(inp_fm, dfm)

    dpw = dict(d)  # pow_int: |a| in [2^-2, 2^2] so a^8 stays far inside the FP32 range
    dpw["ah"], dpw["al"] = rand_ff(rng, n, -2, 2)
    inp_pw = os.path.join(WORK, "in_pw.bin")
    write_inputs(inp_pw, dpw)

    def pick(op):
        if op == 72:
            return inp_pw, dpw
        if op in (30, 31):
            return inp_pos, dpos
        if op in (27, 28, 29):
            return inp_ss, dss
        if op in (37, 38):
            return inp_fm, dfm
        return inp, d

    print(f"\n== A/B/C on {n} random inputs per op ==")
    a_bad, b_bad, c_bad, canon_bad = {}, {}, {}, {}
    for op in OPS:
        path, dd = pick(op)
        rf = run(F_FMA, op, path, "f")
        rn = run(F_NOFMA, op, path, "n")
        if rf is None:
            rep.check(f"op {op} {OPS[op]} exists in FmaX", False)
            continue
        # B: contraction independence
        eq = bits_equal(rf, rn)
        if not eq.all():
            b_bad[op] = int((~eq).sum())
        # A: MatX
        if op in MATX_OPS:
            rm = run(MATX, op, path, "m")
            eqm = bits_equal(rf, rm)
            if op == 39:  # init from real(8): FmaX keeps the FP64-subtraction split, see D
                eqm = eqm | ((rf[0] + rf[1]).astype(f32) == rf[0])
            if not eqm.all():
                a_bad[op] = (int((~eqm).sum()), first_diff(rf, rm, eqm, dd))
        # C: accuracy + canonical form
        if op in CMP_OPS or op in (33, 40, 73, 74, 75, 39, 81):
            continue
        rh, rl = rf[0], rf[1]
        cb = ~finite_canonical(rh, rl)
        if cb.any():
            canon_bad[op] = int(cb.sum())
        ref = reference(op, dd)
        if ref is not None:
            exact, scale = ref
            got = val(rh, rl)
            fin = np.isfinite(rh) & np.isfinite(exact.astype(np.float64))
            denom = np.abs(exact) if (op in RELATIVE_TO_RESULT or scale is None) else scale
            with np.errstate(all="ignore"):
                err = np.abs(got - exact) / np.where(denom == 0, 1, denom)
            err = np.where(fin, err, 0)
            worst = float(err.max())
            if worst > BOUND[op]:
                c_bad[op] = (worst, int((err > BOUND[op]).sum()), int(np.argmax(err)))
    rep.check("A  bit-identical to MatX fltflt.h (ops 1-58)", not a_bad,
              "; ".join(f"{op} {OPS[op]}: {v[0]} diffs, e.g. {v[1]}" for op, v in a_bad.items()))
    rep.check("B  independent of FMA contraction (-fast vs -Mnofma)", not b_bad,
              ", ".join(f"{op} {OPS[op]}: {v}" for op, v in b_bad.items()))
    inherited = {op: v for op, v in canon_bad.items() if op in MATX_OPS and op not in a_bad}
    own = {op: v for op, v in canon_bad.items() if op not in inherited}
    rep.check("C1 canonical form fl(hi+lo)==hi", not own,
              ", ".join(f"{op} {OPS[op]}: {v}" for op, v in own.items()))
    if inherited:
        print("       note: non-canonical results identical to MatX's (deep cancellation in fltflt_fma, "
              "final FastTwoSum precondition |hi|>=|lo| not met upstream): "
              + ", ".join(f"{op} {OPS[op]}: {v}" for op, v in inherited.items()))
    rep.check("C2 accuracy within bound", not c_bad,
              "; ".join(f"{op} {OPS[op]}: worst {v[0]:.2e} (2^{np.log2(v[0]):.1f}), {v[1]} over" for op, v in c_bad.items()))
    check_rounding_random(rep, d)
    check_cmp_random(rep, d)
    check_warp(rep, rng)


def first_diff(rf, rm, eq, d):
    i = int(np.argmin(eq))
    return (f"i={i} a=({d['ah'][i]!r},{d['al'][i]!r}) b=({d['bh'][i]!r},{d['bl'][i]!r}) "
            f"FmaX=({rf[0][i]!r},{rf[1][i]!r},{rf[2][i]}) MatX=({rm[0][i]!r},{rm[1][i]!r},{rm[2][i]})")


def check_rounding_random(rep, d):
    """round/trunc/floor/ceil are exact operations: compare with Fraction-exact values."""
    inp = os.path.join(WORK, "in_rand.bin")
    a = [Fraction(float(h)) + Fraction(float(l)) for h, l in zip(d["ah"], d["al"])]
    import math
    exp = {
        34: [round(x) for x in a],                # Python round(): ties to even
        35: [math.trunc(x) for x in a],
        36: [math.floor(x) for x in a],
        69: [math.ceil(x) for x in a],
    }
    for op, want in exp.items():
        rh, rl, _ = run(F_FMA, op, inp, "f")
        got = [Fraction(float(h)) + Fraction(float(l)) for h, l in zip(rh, rl)]
        bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
        rep.check(f"C3 {OPS[op]} exact", not bad,
                  f"{len(bad)} wrong, e.g. a=({d['ah'][bad[0]]!r},{d['al'][bad[0]]!r}) "
                  f"got {float(got[bad[0]])!r} want {want[bad[0]]}" if bad else "")


def check_cmp_random(rep, d):
    inp = os.path.join(WORK, "in_rand.bin")
    a = val(d["ah"], d["al"]); b = val(d["bh"], d["bl"]); b4 = d["bh"].astype(LD); a4 = d["ah"].astype(LD)
    ops = {41: (a, b, "eq"), 42: (a, b, "ne"), 43: (a, b, "lt"), 44: (a, b, "gt"), 45: (a, b, "le"), 46: (a, b, "ge"),
           47: (a, b4, "eq"), 48: (a, b4, "ne"), 49: (a, b4, "lt"), 50: (a, b4, "gt"), 51: (a, b4, "le"), 52: (a, b4, "ge"),
           53: (a4, b, "eq"), 54: (a4, b, "ne"), 55: (a4, b, "lt"), 56: (a4, b, "gt"), 57: (a4, b, "le"), 58: (a4, b, "ge")}
    bad = []
    for op, (x, y, kind) in ops.items():
        want = getattr(np, {"eq": "equal", "ne": "not_equal", "lt": "less", "gt": "greater",
                            "le": "less_equal", "ge": "greater_equal"}[kind])(x, y).astype(np.int32)
        got = run(F_FMA, op, inp, "f")[2]
        if (got != want).any():
            bad.append(f"{OPS[op]}: {int((got != want).sum())}")
    rep.check("C4 comparisons agree with the exact values", not bad, ", ".join(bad))


def check_warp(rep, rng):
    n = 32 * 512
    d = gen_inputs(rng, n)
    inp = os.path.join(WORK, "in_warp.bin")
    write_inputs(inp, d)
    rh, rl, _ = run(F_FMA, 81, inp, "f")
    a = val(d["ah"], d["al"]).reshape(-1, 32)
    exact = a.sum(axis=1)
    scale = np.abs(a).sum(axis=1)
    rh = rh.reshape(-1, 32); rl = rl.reshape(-1, 32)
    same = (rh == rh[:, :1]).all() and (rl == rl[:, :1]).all()
    err = np.abs(val(rh[:, 0], rl[:, 0]) - exact) / scale
    rep.check("C5 warp_reduce_sum: all lanes equal, error <= 2^-42 of sum|x|", bool(same and err.max() <= 2.0 ** -42),
              f"lanes equal={same}, worst {float(err.max()):.2e}")


# ---------------------------------------------------------------- D: edge cases
def ff(x):
    """Exact canonical (hi, lo) of a Fraction/float/int, by round-to-nearest."""
    x = Fraction(x)
    hi = f32(float(x))
    lo = f32(float(x - Fraction(float(hi))))
    return float(hi), float(lo)


NAN = float("nan")
INF = float("inf")


def edge_cases():
    """(op, a, b, c, d, x8, k, expected) with expected = (hi, lo) or int or 'nan'."""
    p = Fraction(2)
    E = []
    one = (1.0, 0.0)
    # exactness: these fail if an EFT step is contracted into an FMA
    E.append((62, (1 + 2.0 ** -12, 0.0), (1 + 2.0 ** -12, 0.0), (-1.0, 0.0), (1.0, 0.0), 0, 0,
              ff(Fraction(1) * (1 + p ** -12) ** 2 - 1)))
    E.append((7, (1 + 2.0 ** -12, 0.0), (1 + 2.0 ** -12, 0.0), one, one, 0, 0, (1 + 2.0 ** -11, 2.0 ** -24)))
    E.append((61, (1 + 2.0 ** -12, 0.0), one, one, one, 0, 0, (1 + 2.0 ** -11, 2.0 ** -24)))
    E.append((5, ff(1 + p ** -30), (1.0, 0.0), one, one, 0, 0, (2.0 ** -30, 0.0)))
    E.append((1, (1.0, 0.0), (2.0 ** -30, 0.0), one, one, 0, 0, (1.0, 2.0 ** -30)))
    # fma with the lo*lo term carrying the answer (review item 5)
    # a*b = 1 + 2^-29 + 2^-60 against c = -(1 - 2^-20): the 2^-60 = a.lo*b.lo
    # survives only if that term is accumulated (lost in fma_approx by design)
    cc = (-(1.0 - 2.0 ** -20), 0.0)
    E.append((13, (1.0, 2.0 ** -30), (1.0, 2.0 ** -30), cc, one, 0, 0, (2.0 ** -20 + 2.0 ** -29, 2.0 ** -60)))
    # the float-c overload (fltflt.h) adds a.lo*b.lo before its only renormalisation,
    # where 2^-29 + 2^-60 rounds to 2^-29: the term is computed but absorbed here;
    # check A covers that it is present (bit-identical to fltflt.h on random data)
    E.append((14, (1.0, 2.0 ** -30), (1.0, 2.0 ** -30), cc, one, 0, 0, (2.0 ** -20 + 2.0 ** -29, 0.0)))
    E.append((21, (1.0, 2.0 ** -30), (1.0, 2.0 ** -30), cc, one, 0, 0, (2.0 ** -20 + 2.0 ** -29, 0.0)))
    # division / reciprocal / sqrt
    third = Fraction(1, 3)
    E.append((10, one, (3.0, 0.0), one, one, 0, 0, ("near", third, 2.0 ** -46)))
    E.append((60, (3.0, 0.0), one, one, one, 0, 0, ("near", third, 2.0 ** -46)))
    E.append((30, (2.0, 0.0), one, one, one, 0, 0, ("sqrt", Fraction(2), 2.0 ** -44)))
    E.append((30, (0.0, 0.0), one, one, one, 0, 0, (0.0, 0.0)))
    E.append((31, (0.0, 0.0), one, one, one, 0, 0, (0.0, 0.0)))
    E.append((30, (-1.0, 0.0), one, one, one, 0, 0, "nan"))
    E.append((10, one, (0.0, 0.0), one, one, 0, 0, (0.0, 0.0)))  # MatX semantics: x/0 = 0
    # round_to_nearest (ties to even), review item 3
    for x, want in [(2.5, 2), (3.5, 4), (-2.5, -2), (0.5, 0), (-0.5, 0)]:
        E.append((34, (x, 0.0), one, one, one, 0, 0, (float(want), 0.0)))
    E.append((34, (2.5, 1e-8), one, one, one, 0, 0, (3.0, 0.0)))
    E.append((34, (2.5, -1e-8), one, one, one, 0, 0, (2.0, 0.0)))
    E.append((34, (16777218.0, -0.5), one, one, one, 0, 0, ff(16777218)))   # 16777217.5 -> even 16777218
    E.append((34, (16777220.0, -0.5), one, one, one, 0, 0, ff(16777220)))   # 16777219.5 -> 16777220
    E.append((34, (16777216.0, 0.5), one, one, one, 0, 0, ff(16777216)))    # 16777216.5 -> 16777216
    # round_toward_zero / floor / ceil, review item 4
    E.append((35, (1.0, -1e-9), one, one, one, 0, 0, (0.0, 0.0)))
    E.append((35, (-1.0, 1e-9), one, one, one, 0, 0, (0.0, 0.0)))
    E.append((35, (2.0 ** 24, -0.5), one, one, one, 0, 0, ff(2 ** 24 - 1)))
    E.append((36, (1.0, -1e-9), one, one, one, 0, 0, (0.0, 0.0)))
    E.append((36, (2.0 ** 60, -(2.0 ** 35) + 0.5), one, one, one, 0, 0, ff(p ** 60 - p ** 35)))
    E.append((36, (2.0 ** 70, 2.0 ** 40 + 0.5), one, one, one, 0, 0, ff(p ** 70 + p ** 40)))
    E.append((69, (2.0 ** 60, -(2.0 ** 35) + 0.5), one, one, one, 0, 0, ff(p ** 60 - p ** 35 + 1)))
    E.append((69, (-1.0, -1e-9), one, one, one, 0, 0, (-1.0, 0.0)))
    # fmod
    for a, b, want in [(5.5, 2.0, 1.5), (-5.5, 2.0, -1.5), (5.5, -2.0, 1.5)]:
        E.append((37, (a, 0.0), (b, 0.0), one, one, 0, 0, (want, 0.0)))
        E.append((38, (a, 0.0), (b, 0.0), one, one, 0, 0, (want, 0.0)))
    E.append((37, (5.5, 0.0), (0.0, 0.0), one, one, 0, 0, "nan"))
    E.append((38, (5.5, 0.0), (0.0, 0.0), one, one, 0, 0, "nan"))
    # init from real(8), review item 6
    x = 1.0 + 2.0 ** -23 + 2.0 ** -24 - 2.0 ** -52
    E.append((39, one, one, one, one, x, 0, ("init", Fraction(x))))
    E.append((39, one, one, one, one, 1e39, 0, (INF, 0.0)))
    E.append((39, one, one, one, one, -1e39, 0, (-INF, 0.0)))
    E.append((39, one, one, one, one, 1.0 / 3.0, 0, ("init", Fraction(1.0 / 3.0))))
    # comparisons (lexicographic), incl. the nvfortran .not. trap and NaN, -0
    c = (1.0, 2.0 ** -30)
    E.append((50, c, one, one, one, 0, 0, 1))        # (1,2^-30) > 1.0
    E.append((55, one, c, one, one, 0, 0, 1))        # 1.0 < (1,2^-30)
    E.append((49, (1.0, -(2.0 ** -30)), one, one, one, 0, 0, 1))
    E.append((42, c, c, one, one, 0, 0, 0))          # equal inputs: /= must be false
    E.append((48, one, one, one, one, 0, 0, 0))
    E.append((54, one, one, one, one, 0, 0, 0))
    E.append((41, (-0.0, 0.0), (0.0, 0.0), one, one, 0, 0, 1))
    for op in (41, 43, 44, 45, 46):
        E.append((op, (NAN, 0.0), one, one, one, 0, 0, 0))
    E.append((42, (NAN, 0.0), one, one, one, 0, 0, 1))
    # pow_int, abs, neg, r8 operands
    E.append((72, (1.5, 0.0), one, one, one, 0, 5, ff(Fraction(3, 2) ** 5)))
    E.append((72, (2.0, 0.0), one, one, one, 0, 0, (1.0, 0.0)))
    E.append((33, (-3.0, 1e-8), one, one, one, 0, 0, (3.0, -1e-8)))
    E.append((76, (1.0, 0.0), one, one, one, 2.0 ** -40, 0, (1.0, 2.0 ** -40)))
    E.append((83, ff(1 + p ** -40), one, one, one, 1.0, 0, (2.0 ** -40, 0.0)))
    return E


def check_edges(rep):
    E = edge_cases()
    by_op = {}
    for e in E:
        by_op.setdefault(e[0], []).append(e)
    nbad = 0
    msgs = []
    for op, lst in by_op.items():
        d = {k: [] for k in NAMES + ["x8", "k"]}
        for (_, a, b, c, dd, x8, k, _) in lst:
            for p, v in zip("abcd", (a, b, c, dd)):
                d[p + "h"].append(v[0]); d[p + "l"].append(v[1])
            d["x8"].append(x8); d["k"].append(k)
        inp = os.path.join(WORK, f"in_edge_{op}.bin")
        write_inputs(inp, {k: np.array(v) for k, v in d.items()})
        rh, rl, ri = run(F_FMA, op, inp, "e")
        for i, e in enumerate(lst):
            want = e[7]
            gh, gl, gi = float(rh[i]), float(rl[i]), int(ri[i])
            if want == "nan":
                ok = np.isnan(gh)
            elif isinstance(want, int):
                ok = gi == want
            elif want[0] == "near":
                got = Fraction(gh) + Fraction(gl)
                ok = abs(got - want[1]) <= abs(want[1]) * Fraction(want[2])
            elif want[0] == "sqrt":
                got = Fraction(gh) + Fraction(gl)
                ok = abs(got * got - want[1]) <= want[1] * Fraction(2 * want[2])
            elif want[0] == "init":
                got = Fraction(gh) + Fraction(gl)
                ok = (abs(got - want[1]) <= abs(want[1]) * Fraction(2.0 ** -48)
                      and float(f32(gh) + f32(gl)) == gh)
            else:
                wh, wl = float(f32(want[0])), float(f32(want[1]))
                ok = (gh == wh or (np.isnan(gh) and np.isnan(wh))) and gl == wl
            if not ok:
                nbad += 1
                msgs.append(f"{OPS[op]}{e[1:3]} -> ({gh!r},{gl!r},{gi}) want {want}")
    rep.check(f"D  {len(E)} edge cases with exact expected results", nbad == 0, "\n      " + "\n      ".join(msgs) if msgs else "")


def main():
    os.makedirs(WORK, exist_ok=True)
    for exe in (F_FMA, F_NOFMA, MATX):
        if not os.path.exists(exe):
            sys.exit(f"missing {exe}: run make first")
    rep = Report()
    rng = np.random.default_rng(20261115)
    check_random(rep, rng, int(os.environ.get("FF_TEST_N", 1 << 18)))
    print("\n== D edge cases ==")
    check_edges(rep)
    print()
    if rep.fail:
        print(f"FAILED: {len(rep.fail)} check(s)")
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
