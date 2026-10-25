"""Peak-sum optical sources: lineshape peaks summed into a uniaxial molecular tensor.

`Peak` holds one labeled lineshape (Gaussian, Lorentzian, Voigt or pseudo-Voigt)
with a height and a transition-dipole (TDM) polar angle. `PeakSumSLD` sums a set
of peaks on top of an optional tabulated baseline and projects the result to the
laboratory frame, the same way `UniTensorSLD` does for a table alone. Line shapes
are evaluated by the Rust `peak_sum_molecular` kernel (`lineshapes` crate).

A peak parameter that is a single number (or a `Parameter`) is constant over its
slab and is evaluated once per energy. Only a parameter bound to a depth field is
evaluated per microslab inside a `DepthProfile`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from refnx.analysis import Parameter, Parameters, possibly_create_parameter

from refloxide import optics
from refloxide.data import OpticalConstants
from refloxide.mixing import pack_lab_diagonal
from refloxide.model import (
    Scatterer,
    UniTensorSLD,
    _depth_field_scalar,
    _is_depth_field,
)
from refloxide.profiles import _eval_field
from refloxide.sources import lab_diagonal_from_cos2

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from refloxide.profiles import _FieldBase

MAGIC_ANGLE: float = float(np.arccos(1.0 / np.sqrt(3.0)))
"""TDM polar angle (radians) at which both fractions equal 1, `arccos(1/sqrt(3))`."""

_SHAPE_PARAMS: dict[str, tuple[str, ...]] = {
    "gaussian": ("energy", "width"),
    "lorentzian": ("energy", "gamma"),
    "voigt": ("energy", "sigma", "gamma"),
    "pseudo_voigt": ("energy", "fwhm", "eta"),
}


def tdm_fractions(theta: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Molecular `(f_xx, f_zz)` projections for a TDM polar angle.

    Parameters
    ----------
    theta : ArrayLike
        TDM polar angle in radians, measured from the molecular normal.
        `0` is along the normal, `pi/2` lies in the molecular plane.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        `(1.5*sin(theta)**2, 3*cos(theta)**2)`. Always satisfies
        `(2*f_xx + f_zz)/3 == 1`.
    """
    angle = np.asarray(theta, dtype=np.float64)
    f_xx = 1.5 * np.sin(angle) ** 2
    f_zz = 3.0 * np.cos(angle) ** 2
    return f_xx, f_zz


def _require_finite(values: NDArray[np.float64], channel: str) -> None:
    """Raise `ValueError` naming `channel` if any value is non-finite."""
    finite = np.isfinite(values)
    if not np.all(finite):
        bad = float(np.asarray(values, dtype=np.float64)[~finite].ravel()[0])
        msg = f"{channel} must be finite, got {bad!r}"
        raise ValueError(msg)


class Peak:
    """One labeled lineshape peak with height and TDM polar angle.

    Every value may be a number (stored as a frozen `Parameter`), a
    `Parameter`, or a depth field such as `SecondOrderTransition`.

    Parameters
    ----------
    kind : str, optional
        Lineshape family: ``"gaussian"`` (``energy``, ``width``),
        ``"lorentzian"`` (``energy``, ``gamma``), ``"voigt"`` (``energy``,
        ``sigma``, ``gamma``) or ``"pseudo_voigt"`` (``energy``, ``fwhm``,
        ``eta``). ``width`` is the 1/e half-width of the Gaussian and
        ``gamma`` the HWHM of the Lorentzian.
    label : str
        Peak name, used in parameter names (``f"{name}_{label}_{param}"``
        once the peak belongs to a `PeakSumSLD`) and depth-field addresses.
    height : float, Parameter, or depth field
        Peak strength. Multiplies the peak's unit-amplitude line shape.
    tdm_tilt : float, Parameter, or depth field, optional
        TDM polar angle in radians from the molecular normal. Defaults to
        `MAGIC_ANGLE`, which gives `f_xx = f_zz = 1`.
    **shape_params : float, Parameter, or depth field
        Shape parameters for `kind`, as listed above.

    Raises
    ------
    ValueError
        If `kind` is not a known lineshape.
    TypeError
        If the shape parameters do not match `kind`.
    """

    def __init__(
        self,
        kind: str = "gaussian",
        *,
        label: str,
        height: float | Parameter | Any,
        tdm_tilt: float | Parameter | Any = MAGIC_ANGLE,
        **shape_params: float | Parameter | Any,
    ) -> None:
        if kind not in _SHAPE_PARAMS:
            msg = f"unknown peak kind {kind!r}; expected one of {sorted(_SHAPE_PARAMS)}"
            raise ValueError(msg)
        expected = _SHAPE_PARAMS[kind]
        missing = [p for p in expected if p not in shape_params]
        unexpected = sorted(set(shape_params) - set(expected))
        if missing or unexpected:
            msg = (
                f"{kind} peak {label!r} takes {list(expected)}; "
                f"missing {missing}, unexpected {unexpected}"
            )
            raise TypeError(msg)
        self.kind = kind
        self.label = label
        self._owned: set[str] = set()
        self._owner: PeakSumSLD | None = None
        values: dict[str, Any] = {p: shape_params[p] for p in expected}
        values["height"] = height
        values["tdm_tilt"] = tdm_tilt
        self._items: dict[str, Any] = {
            pname: self._store(value, pname) for pname, value in values.items()
        }

    def _store(self, value: Any, pname: str) -> Any:
        if _is_depth_field(value) or isinstance(value, Parameter):
            return value
        self._owned.add(pname)
        return possibly_create_parameter(
            value, name=f"{self.label}_{pname}", vary=False
        )

    def parameter_items(self) -> dict[str, Parameter | _FieldBase]:
        """Stored parameter values for this peak, in shape-then-height-then-TDM order.

        Returns
        -------
        dict[str, Parameter | _FieldBase]
            Map from parameter name (for example ``"width"``, ``"height"``,
            ``"tdm_tilt"``) to the stored `Parameter` or depth field.
        """
        return dict(self._items)

    def __repr__(self) -> str:
        return f"Peak({self.kind!r}, label={self.label!r})"


def _channel_values(
    value: Any, z: NDArray[np.float64] | None, thickness: float
) -> NDArray[np.float64]:
    """Length-1 array for a constant, or depth-field values at `z` (length n_z).

    With `z=None` (no depth) a depth field collapses to its bulk pin.
    """
    if _is_depth_field(value):
        if z is None:
            return np.array([_depth_field_scalar(value)], dtype=np.float64)
        return np.asarray(_eval_field(value, z, thickness=thickness), dtype=np.float64)
    if isinstance(value, Parameter):
        return np.array([float(value.value or 0.0)], dtype=np.float64)
    return np.array([float(value)], dtype=np.float64)


def _shape_rows(cols: list[NDArray[np.float64]], n_s: int) -> NDArray[np.float64]:
    """Flatten one peak's shape columns for the kernel.

    All-constant columns give one broadcast row. Otherwise every column is
    expanded to `n_s` slices and flattened in (slice, parameter) order.
    """
    if all(col.shape[0] == 1 for col in cols):
        return np.array([col[0] for col in cols], dtype=np.float64)
    expanded = [np.broadcast_to(col, (n_s,)) for col in cols]
    return np.stack(expanded, axis=1).reshape(-1)


class PeakSumSLD(Scatterer):
    """Uniaxial material index from a sum of lineshape peaks over an optional baseline.

    The molecular indices are `density * (baseline + sum_k peak_k)`, with
    `peak_k = height_k * f_k * line_k(E)` and `(f_xx, f_zz)` from the TDM angle.
    They are then projected to the laboratory frame with `rotation`, as
    `UniTensorSLD` does. Energy is resolved fresh on every call.

    `density`, `rotation` and `energy_offset` use the same storage and defaults
    as `UniTensorSLD`. `tensor_at` evaluates a depth-field parameter at its bulk
    pin. `lab_channels_at_depth` evaluates it at each microslab depth.

    Parameters
    ----------
    peaks : sequence of Peak
        Peaks to sum. Labels must be unique.
    baseline : OpticalConstants, polars.DataFrame, str, or pathlib.Path, optional
        Tabulated molecular constants added under the peaks. Omit for a
        peak-only material.
    density : float, Parameter, or depth field, optional
        Mass-density scale applied to the molecular indices (g/cm^3).
    rotation : float, Parameter, or depth field, optional
        Polar rotation of the molecular frame in radians (`0` is face-on,
        `pi/2` edge-on).
    energy_offset : float or Parameter, optional
        Energy offset (eV) added to the query energy before the peaks and
        baseline are evaluated. Frozen at 0 by default.
    name : str, optional
        Prefix for parameter names (``f"{name}_{label}_{param}"``).

    Raises
    ------
    ValueError
        If two peaks share a label.

    Notes
    -----
    Evaluating a sum of peaks costs more than interpolating a table, so a
    `PeakSumSLD` may be slower than the tabulated uniaxial tensor
    (`UniTensorSLD`). In a benchmark of one uniform slab with five scalar peaks
    over 200 energies it took about 1.2 times as long as the tabulated slab. A
    peak parameter that is a single number is evaluated once per energy.
    Parameters bound to a depth field are evaluated per microslab, and a
    depth-varying width costs more than a depth-varying height.
    """

    def __init__(
        self,
        peaks: Sequence[Peak],
        *,
        baseline: OpticalConstants | Any | None = None,
        density: float | Parameter | Any = 1.0,
        rotation: float | Parameter | Any = 0.0,
        energy_offset: float | Parameter = 0.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.peaks: list[Peak] = list(peaks)
        labels = [peak.label for peak in self.peaks]
        if len(set(labels)) != len(labels):
            msg = f"peak labels must be unique, got {labels}"
            raise ValueError(msg)
        for peak in self.peaks:
            if peak._owner is not None:
                msg = (
                    f"Peak {peak.label!r} is already attached to PeakSumSLD "
                    f"{peak._owner.name!r}; create a separate Peak per source"
                )
                raise ValueError(msg)
        self.baseline = (
            None if baseline is None else OpticalConstants.from_source(baseline)
        )
        self.density = UniTensorSLD._store_channel(
            density, name=f"{name}_density", kind="density"
        )
        self.rotation = UniTensorSLD._store_channel(
            rotation, name=f"{name}_rotation", kind="rotation"
        )
        self.energy_offset = possibly_create_parameter(
            energy_offset,
            name=f"{name}_energy_offset",
            vary=False,
            bounds=(-0.01, 0.01),
        )
        self._parameters = Parameters(name=name)
        self._parameters.extend(self._collect_parameters())
        # Commit peak ownership and renames only after every fallible step above
        # has succeeded, so a failed constructor leaves the Peaks untouched.
        for peak in self.peaks:
            peak._owner = self
            for pname in peak._owned:
                peak._items[pname].name = f"{name}_{peak.label}_{pname}"

    def _collect_parameters(self) -> list[Parameter]:
        """Unique parameters: peak values, density/rotation channels, energy offset."""
        found: list[Parameter] = []
        seen: set[int] = set()

        def add(param: Parameter) -> None:
            if id(param) not in seen:
                seen.add(id(param))
                found.append(param)

        for peak in self.peaks:
            for value in peak._items.values():
                if _is_depth_field(value):
                    for param in value.parameters:
                        add(param)
                else:
                    add(value)
        for channel in (self.density, self.rotation):
            if _is_depth_field(channel):
                for param in channel.parameters:
                    add(param)
            else:
                add(channel)
        add(self.energy_offset)
        return found

    def depth_fields(self) -> dict[str, _FieldBase]:
        """Depth-field-bound peak parameters, keyed by ``"{name}.{label}.{param}"``.

        Returns
        -------
        dict[str, _FieldBase]
            One entry per peak parameter (including ``height`` and
            ``tdm_tilt``) that is bound to a depth field. Empty when every
            peak parameter is a number.
        """
        fields: dict[str, _FieldBase] = {}
        for peak in self.peaks:
            for pname, value in peak._items.items():
                if _is_depth_field(value):
                    fields[f"{self.name}.{peak.label}.{pname}"] = value
        return fields

    def _peak_sums(
        self,
        eff_ev: NDArray[np.float64],
        z: NDArray[np.float64] | None,
        thickness: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Molecular peak sums `(n_xx, n_zz)`, each `(n_s, n_E)`.

        `n_s = 1` when `z` is None (no depth).
        """
        n_s = 1 if z is None else int(z.shape[0])
        if not self.peaks:
            zeros = np.zeros((n_s, eff_ev.shape[0]), dtype=np.complex128)
            return zeros, zeros.copy()
        kinds: list[str] = []
        shapes: list[NDArray[np.float64]] = []
        heights: list[NDArray[np.float64]] = []
        f_xx: list[NDArray[np.float64]] = []
        f_zz: list[NDArray[np.float64]] = []
        for peak in self.peaks:
            items = peak._items
            cols = [
                _channel_values(items[pname], z, thickness)
                for pname in _SHAPE_PARAMS[peak.kind]
            ]
            kinds.append(peak.kind)
            shapes.append(_shape_rows(cols, n_s))
            heights.append(_channel_values(items["height"], z, thickness))
            tilt = _channel_values(items["tdm_tilt"], z, thickness)
            _require_finite(tilt, f"peak {peak.label!r} tdm_tilt")
            fxx, fzz = tdm_fractions(tilt)
            f_xx.append(fxx)
            f_zz.append(fzz)
        return optics.peak_sum_molecular(
            kinds, shapes, heights, f_xx, f_zz, eff_ev, n_s
        )

    def _lab_diagonals(
        self,
        eff_ev: NDArray[np.float64],
        z: NDArray[np.float64] | None,
        thickness: float,
        density: NDArray[np.float64] | None,
        gamma: NDArray[np.float64] | None,
        cos2: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Lab `(n_o, n_e)`, each `(n_s, n_E)`; `n_s = 1` without depth.

        Channels passed as `None` are read from this source (per slice when
        `z` is given). `cos2` overrides `cos(gamma)**2` when supplied.
        """
        if density is None:
            density = _channel_values(self.density, z, thickness)
        rho = np.asarray(density, dtype=np.float64).reshape(-1, 1)
        _require_finite(rho, "density")
        if cos2 is None:
            if gamma is None:
                gamma = _channel_values(self.rotation, z, thickness)
            gamma_arr = np.asarray(gamma, dtype=np.float64)
            _require_finite(gamma_arr, "rotation (gamma)")
            cos2 = np.cos(gamma_arr) ** 2
        cos2_arr = np.asarray(cos2, dtype=np.float64)
        _require_finite(cos2_arr, "cos2")
        cos2_col = cos2_arr.reshape(-1, 1)
        if self.baseline is None:
            base_xx = np.zeros(eff_ev.shape[0], dtype=np.complex128)
            base_zz = np.zeros(eff_ev.shape[0], dtype=np.complex128)
        else:
            base_xx, base_zz = self.baseline.molecular_index_at_many(eff_ev, 1.0)
        peak_xx, peak_zz = self._peak_sums(eff_ev, z, thickness)
        n_xx = rho * (base_xx[np.newaxis, :] + peak_xx)
        n_zz = rho * (base_zz[np.newaxis, :] + peak_zz)
        n_o, n_e = lab_diagonal_from_cos2(n_xx, n_zz, cos2_col)
        return (
            np.asarray(n_o, dtype=np.complex128),
            np.asarray(n_e, dtype=np.complex128),
        )

    def _shifted(self, energies_ev: NDArray[np.float64]) -> NDArray[np.float64]:
        """Query energies plus the energy offset, after checking both are finite."""
        energies = np.asarray(energies_ev, dtype=np.float64)
        _require_finite(energies, "energy_ev")
        offset = np.array([float(self.energy_offset.value or 0.0)])
        _require_finite(offset, "energy_offset")
        return energies + offset[0]

    def tensor_at(self, energy_ev: float) -> NDArray[np.complex128]:
        """Laboratory `(3, 3)` tensor at `energy_ev`, depth fields at their bulk pin."""
        return self.tensor_at_many(np.array([float(energy_ev)], dtype=np.float64))[0]

    def tensor_at_many(
        self, energies_ev: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Laboratory tensors at many energies, shape `(n_E, 3, 3)`.

        Depth-field parameters are evaluated at their bulk pin.
        """
        eff_ev = self._shifted(energies_ev)
        n_o, n_e = self._lab_diagonals(eff_ev, None, 0.0, None, None, None)
        return pack_lab_diagonal(n_o[0], n_e[0])

    def lab_channels_at_depth(
        self,
        energy_ev: float,
        z: NDArray[np.float64],
        thickness: float,
        *,
        density: NDArray[np.float64] | None,
        gamma: NDArray[np.float64] | None,
        cos2: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Lab `(n_o, n_e)` at one energy for every microslab depth `z`.

        Parameters
        ----------
        energy_ev : float
            Photon energy in eV.
        z : NDArray[np.float64]
            Slice depths, shape `(n_z,)`. Depth fields are evaluated here.
        thickness : float
            Thickness of the depth profile (passed to the depth fields).
        density, gamma, cos2 : NDArray or None
            Per-slice overrides (length 1 or `n_z`) for the mass-density scale,
            the molecular tilt in radians, and `cos^2(gamma)`. `None` reads the
            source's own channel (per slice where it is a depth field).

        Returns
        -------
        tuple[NDArray[np.complex128], NDArray[np.complex128]]
            `(n_o, n_e)`, each shape `(n_z,)`.
        """
        eff_ev = self._shifted(np.array([float(energy_ev)], dtype=np.float64))
        z_arr = np.asarray(z, dtype=np.float64)
        n_o, n_e = self._lab_diagonals(
            eff_ev, z_arr, float(thickness), density, gamma, cos2
        )
        return n_o[:, 0], n_e[:, 0]

    @property
    def parameters(self) -> Parameters:
        self._parameters.name = self.name
        return self._parameters

    def effective_density(self) -> float | None:
        return UniTensorSLD._channel_scalar(self.density)

    def effective_rotation(self) -> float | None:
        return UniTensorSLD._channel_scalar(self.rotation)

    def __repr__(self) -> str:
        labels = [peak.label for peak in self.peaks]
        return f"PeakSumSLD(name={self.name!r}, peaks={labels!r})"
