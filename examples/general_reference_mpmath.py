"""50-digit reference check for ``examples/general_reference.rs``.

Recomputes each dumped point as ``M = D0^-1 * prod_j expm(-i k0 d_j Delta_j)
* D_N[:, forward]`` with mpmath at 50 significant digits. Interior layers enter
only through matrix exponentials of their Berreman matrices, so this check
shares no eigenmode solving, sorting, labeling, or gauge choice with the engine
under test. Usage::

    cargo run --profile perf --no-default-features --example general_reference > ref.txt
    uv run --with mpmath python examples/general_reference_mpmath.py ref.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50
CHANNELS = ("pp", "ss", "sp", "ps")


def parse(path: Path) -> tuple[mp.mpf, dict, dict, dict]:
    """Wavenumber, per-stack layers ``(d, chi)``, q rows, and stack names."""
    k0 = mp.mpf(0)
    layers: dict[int, list] = {}
    rows: dict[int, list] = {}
    names: dict[int, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        tag, rest = parts[0], parts[1:]
        if tag == "K":
            k0 = mp.mpf(rest[0])
        elif tag == "N":
            names[int(rest[0])] = " ".join(rest[1:])
        elif tag == "L":
            values = [mp.mpf(x) for x in rest[2:]]
            chi = mp.matrix(3, 3)
            for i in range(3):
                for j in range(3):
                    k = 2 * (3 * i + j)
                    chi[i, j] = mp.mpc(values[k], values[k + 1])
            layers.setdefault(int(rest[0]), []).append((mp.mpf(rest[1]), chi))
        elif tag == "Q":
            q_row = [mp.mpf(rest[1])] + [float(x) for x in rest[2:]]
            rows.setdefault(int(rest[0]), []).append(q_row)
    return k0, layers, rows, names


def berreman(chi: mp.matrix, zeta: mp.mpf) -> mp.matrix:
    """Berreman matrix for ``psi = [Ex, Hy, Ey, Hx]`` (same form as the engine)."""

    def eps(i: int, j: int) -> mp.mpc:
        return (1 if i == j else 0) + chi[i, j]

    ezz = eps(2, 2)
    a_x, a_y, a_h = -eps(2, 0) / ezz, -eps(2, 1) / ezz, -zeta / ezz
    return mp.matrix(
        [
            [zeta * a_x, 1 + zeta * a_h, zeta * a_y, 0],
            [
                eps(0, 0) + eps(0, 2) * a_x,
                eps(0, 2) * a_h,
                eps(0, 1) + eps(0, 2) * a_y,
                0,
            ],
            [0, 0, 0, -1],
            [
                -(eps(1, 0) + eps(1, 2) * a_x),
                -(eps(1, 2) * a_h),
                zeta**2 - eps(1, 1) - eps(1, 2) * a_y,
                0,
            ],
        ]
    )


def reflectance(q: mp.mpf, k0: mp.mpf, stack: list) -> list[float]:
    """``[R_pp, R_ss, R_sp, R_ps]`` from the transfer-matrix product."""
    q0 = q / (2 * k0)
    zeta = mp.sqrt(1 - q0**2)
    d0 = mp.matrix([[q0, -q0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -q0, q0]])
    transfer = mp.eye(4)
    for thickness, chi in stack[1:-1]:
        transfer = transfer * mp.expm(-1j * k0 * thickness * berreman(chi, zeta))
    eigvals, eigvecs = mp.eig(berreman(stack[-1][1], zeta))
    forward = sorted(range(4), key=lambda i: -mp.im(eigvals[i]))[:2]
    d_n = mp.matrix(4, 2)
    for col, i in enumerate(forward):
        for r in range(4):
            d_n[r, col] = eigvecs[r, i]
    m = mp.inverse(d0) * transfer * d_n
    down = mp.matrix([[m[0, 0], m[0, 1]], [m[2, 0], m[2, 1]]])
    up = mp.matrix([[m[1, 0], m[1, 1]], [m[3, 0], m[3, 1]]])
    r = up * mp.inverse(down)
    return [float(abs(r[i, j]) ** 2) for i, j in ((0, 0), (1, 1), (1, 0), (0, 1))]


def max_error(ref: list, rows: list, column: int, channel: int) -> tuple[float, int]:
    """Worst relative error (floored at 1e-6 of the channel maximum) and NaN count."""
    peak = max(x[channel] for x in ref)
    worst, failures = 0.0, 0
    for exact, row in zip(ref, rows, strict=True):
        value = row[column]
        if value != value:
            failures += 1
            continue
        worst = max(
            worst, abs(value - exact[channel]) / max(exact[channel], 1e-6 * peak)
        )
    return worst, failures


def main() -> None:
    k0, layers, rows, names = parse(
        Path(sys.argv[1] if len(sys.argv) > 1 else "ref.txt")
    )
    print(  # stdout is the report
        f"{'stack':<42} {'chan':>4} {'max R':>9} "
        f"{'f64 err':>9} {'f32 err':>9} {'f32 NaN':>7}"
    )
    for si in sorted(rows):
        ref = [reflectance(row[0], k0, layers[si]) for row in rows[si]]
        for c, name in enumerate(CHANNELS):
            if max(x[c] for x in ref) == 0.0:
                continue
            e64, _ = max_error(ref, rows[si], 1 + c, c)
            e32, fails = max_error(ref, rows[si], 5 + c, c)
            peak = max(x[c] for x in ref)
            print(
                f"{names[si][:42]:<42} {name:>4} {peak:>9.2e} "
                f"{e64:>9.1e} {e32:>9.1e} {fails:>7d}"
            )


if __name__ == "__main__":
    main()
