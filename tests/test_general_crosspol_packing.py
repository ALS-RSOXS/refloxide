"""Lock general_reflectivity cross-pol index layout.

Kernel packing is ``[[R_pp, R_sp], [R_ps, R_ss]]`` with
``R_sp`` = s reflected from p incident and ``R_ps`` = p reflected from s
incident (see ``solve_point_general`` / ``reflect_chain``).
"""

from __future__ import annotations

import numpy as np

from refloxide.tmm import general_reflectivity

_ENERGY_EV = 284.4
_Q = np.linspace(0.02, 0.22, 64)


def _diag(nx: complex, ny: complex, nz: complex) -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.complex128)
    m[0, 0], m[1, 1], m[2, 2] = nx, ny, nz
    return m


def _stack(
    t_lab: np.ndarray, thickness: float = 200.0
) -> tuple[np.ndarray, np.ndarray]:
    n_si = 7.5e-4 + 1.2e-4j
    layers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [
                thickness,
                0.5 * (t_lab[0, 0].real + t_lab[1, 1].real),
                0.5 * (t_lab[0, 0].imag + t_lab[1, 1].imag),
                2.0,
            ],
            [0.0, n_si.real, n_si.imag, 3.0],
        ],
        dtype=np.float64,
    )
    tensor = np.stack(
        [
            _diag(0j, 0j, 0j),
            t_lab,
            _diag(n_si, n_si, n_si),
        ]
    )
    return layers, tensor


def test_lab_diagonal_biaxial_has_zero_cross_pol() -> None:
    """xz incidence on a lab-diagonal biaxial film keeps s/p decoupled."""
    t_lab = _diag(0.80e-3 + 2.0e-4j, 2.20e-3 + 7.0e-4j, 1.50e-3 + 4.0e-4j)
    layers, tensor = _stack(t_lab)
    refl = general_reflectivity(_Q, layers, tensor, _ENERGY_EV, parallel=False)
    np.testing.assert_array_equal(refl[:, 0, 1], 0.0)
    np.testing.assert_array_equal(refl[:, 1, 0], 0.0)
    assert float(refl[:, 0, 0].max()) > 0.0
    assert float(refl[:, 1, 1].max()) > 0.0


def test_rotated_biaxial_cross_pol_indices_match_kernel_convention() -> None:
    """``[:,0,1]`` is ``R_sp`` and ``[:,1,0]`` is ``R_ps`` after in-plane rotation.

    A 45 degree z-rotation mixes the in-plane principal axes into the
    incidence plane so both cross-pol power channels are nonzero. Their
    assignment follows ``reflect_chain``:
    ``result[0][1] = refl(s_up, p_down)`` and
    ``result[1][0] = refl(p_up, s_down)``.
    """
    bi_xx = 0.80e-3 + 2.0e-4j
    bi_yy = 2.20e-3 + 7.0e-4j
    bi_zz = 1.50e-3 + 4.0e-4j
    phi = np.deg2rad(45.0)
    c, s = float(np.cos(phi)), float(np.sin(phi))
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t_lab = rot_z @ _diag(bi_xx, bi_yy, bi_zz) @ rot_z.T
    layers, tensor = _stack(t_lab)
    refl = general_reflectivity(_Q, layers, tensor, _ENERGY_EV, parallel=False)

    r_pp = refl[:, 0, 0]
    r_sp = refl[:, 0, 1]
    r_ps = refl[:, 1, 0]
    r_ss = refl[:, 1, 1]

    assert float(r_sp.max()) > 1e-8
    assert float(r_ps.max()) > 1e-8
    assert float(r_pp.max()) > float(r_sp.max())
    assert float(r_ss.max()) > float(r_sp.max())
    # Reciprocal absorbing biaxial: cross-pol channels stay close (not identical).
    rel = np.max(np.abs(r_sp - r_ps)) / max(float(r_sp.max()), 1e-30)
    assert rel < 0.02

    # Index contract from public stubs: [0,1]=R_sp, [1,0]=R_ps (not swapped
    # relative to [[R_pp, R_sp], [R_ps, R_ss]]).
    np.testing.assert_array_equal(r_sp, refl[:, 0, 1])
    np.testing.assert_array_equal(r_ps, refl[:, 1, 0])
