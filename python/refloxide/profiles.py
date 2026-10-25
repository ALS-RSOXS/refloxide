"""Depth field forms and `DepthProfile` stack segments.

Owns inspectable scalar fields (`Constant`, `SecondOrderTransition`,
`Diffusion`, `Polynomial`, `Spline`, `CallableField`) and the multi-row
`DepthProfile` / `CallableDepthProfile` components that sample them onto a
microslab mesh for the uniaxial TMM. Does not own tabular OOC loading or the
reflectivity kernel.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeGuard, cast

import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from scipy.interpolate import CubicSpline
from scipy.special import erf

from refloxide import optics
from refloxide.mixing import pack_lab_diagonal
from refloxide.model import Mix, MultiRowComponent, Scatterer, UniTensorSLD
from refloxide.pxr.energy.bookended import adaptive_microslab_thicknesses
from refloxide.sources import lab_diagonal_from_cos2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

DiffusionKind = Literal["couple", "exponential", "finite"]
MeshKind = Literal["adaptive", "uniform"]

__all__ = [
    "CallableDepthProfile",
    "CallableField",
    "Constant",
    "DepthProfile",
    "Diffusion",
    "FieldContext",
    "Polynomial",
    "SecondOrderTransition",
    "Spline",
    "lab_tensor_from_order_parameter",
]


class FieldContext(Protocol):
    """Injected kwargs every field callable may rely on."""

    thickness: float
    energy_ev: float


def lab_tensor_from_order_parameter(
    n_o: complex,
    n_e: complex,
    order: float | NDArray[np.float64],
    *,
    kind: Literal["gamma", "cos2_gamma"] = "gamma",
) -> NDArray[np.complex128]:
    """Build lab-frame ``(3,3)`` or ``(n,3,3)`` tensors from an order parameter.

    Parameters
    ----------
    n_o, n_e
        Laboratory ordinary/extraordinary ``delta+i*beta`` when `kind` is
        already a lab diagonal; for ``gamma``/``cos2_gamma`` with molecular
        inputs, pass molecular ``n_xx``/``n_zz`` instead and this function
        reprojects.
    order
        Tilt angle (rad) or ``<cos^2 gamma>`` depending on `kind`.
    kind
        ``gamma`` uses ``uniaxial_lab_tensor``; ``cos2_gamma`` uses
        `lab_diagonal_from_cos2`.
    """
    match kind:
        case "gamma":
            order_arr = np.asarray(order, dtype=np.float64)
            if order_arr.ndim == 0:
                return np.asarray(
                    optics.uniaxial_lab_tensor(n_o, n_e, float(order_arr)),
                    dtype=np.complex128,
                )
            tensors = np.empty((order_arr.size, 3, 3), dtype=np.complex128)
            for i, gamma in enumerate(order_arr.ravel()):
                tensors[i] = np.asarray(
                    optics.uniaxial_lab_tensor(n_o, n_e, float(gamma)),
                    dtype=np.complex128,
                )
            return tensors.reshape((*order_arr.shape, 3, 3))
        case "cos2_gamma":
            n_o_lab, n_e_lab = lab_diagonal_from_cos2(n_o, n_e, order)
            return pack_lab_diagonal(n_o_lab, n_e_lab)
        case _:
            msg = f"unknown order-parameter kind {kind!r}"
            raise ValueError(msg)


def _param(value: float | Parameter, name: str, **kwargs: Any) -> Parameter:
    return cast("Parameter", possibly_create_parameter(value, name=name, **kwargs))


class _FieldBase:
    """Shared Parameter bag + call/plot helpers for scalar depth fields."""

    name: str

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        raise NotImplementedError

    def __call__(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        return self.evaluate(z, **kwargs)

    @property
    def parameters(self) -> Parameters:
        raise NotImplementedError

    def plot(self, ax: Axes | None = None, *, n: int = 400, **kwargs: Any) -> Axes:
        """Plot this field vs depth on `[0, thickness]` when thickness is known."""
        thickness = float(kwargs.get("thickness", getattr(self, "thickness", 1.0)))
        if isinstance(thickness, Parameter):
            thickness = float(thickness.value or 1.0)
        z = np.linspace(0.0, thickness, n)
        y = np.asarray(
            self.evaluate(z, thickness=thickness, **kwargs), dtype=np.float64
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(z, y, **{k: v for k, v in kwargs.items() if k not in {"thickness"}})
        ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
        ax.set_ylabel(self.name or "field")
        return ax


class Constant(_FieldBase):
    """Flat channel value across depth."""

    def __init__(self, value: float | Parameter, *, name: str = "constant") -> None:
        self.name = name
        self.value = _param(value, f"{name}_value")

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        del kwargs
        val = float(self.value.value or 0.0)
        if isinstance(z, np.ndarray):
            return np.full(z.shape, val, dtype=np.float64)
        return val

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.append(self.value)
        return params


class SecondOrderTransition(_FieldBase):
    """Two-sided exponential relaxation into a bulk plateau (2nd-order form).

    ``f(z) = bulk + (top - bulk) exp(-z/tau_top)
                 + (bottom - bulk) exp(-(T - z)/tau_bottom)``

    This is the physics formerly labeled "bookended".
    """

    def __init__(
        self,
        thickness: float | Parameter,
        bulk: float | Parameter,
        top: float | Parameter,
        bottom: float | Parameter,
        tau_top: float | Parameter,
        tau_bottom: float | Parameter,
        *,
        name: str = "second_order",
        # Deprecated aliases
        vac: float | Parameter | None = None,
        sub: float | Parameter | None = None,
        tau_vac: float | Parameter | None = None,
        tau_si: float | Parameter | None = None,
    ) -> None:
        if (
            vac is not None
            or sub is not None
            or tau_vac is not None
            or tau_si is not None
        ):
            warnings.warn(
                "vac/sub/tau_vac/tau_si are deprecated; "
                "use top/bottom/tau_top/tau_bottom",
                DeprecationWarning,
                stacklevel=2,
            )
            if vac is not None:
                top = vac
            if sub is not None:
                bottom = sub
            if tau_vac is not None:
                tau_top = tau_vac
            if tau_si is not None:
                tau_bottom = tau_si
        self.name = name
        self.thickness = _param(thickness, f"{name}_thickness", vary=False)
        self.bulk = _param(bulk, f"{name}_bulk")
        self.top = _param(top, f"{name}_top")
        self.bottom = _param(bottom, f"{name}_bottom")
        self.tau_top = _param(tau_top, f"{name}_tau_top", bounds=(1e-6, None))
        self.tau_bottom = _param(tau_bottom, f"{name}_tau_bottom", bounds=(1e-6, None))

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        thickness = float(kwargs.get("thickness", self.thickness.value or 0.0))
        bulk = float(self.bulk.value or 0.0)
        top = float(self.top.value or 0.0)
        bottom = float(self.bottom.value or 0.0)
        tau_top = max(float(self.tau_top.value or 1e-6), 1e-6)
        tau_bottom = max(float(self.tau_bottom.value or 1e-6), 1e-6)
        if isinstance(z, np.ndarray):
            zz = np.asarray(z, dtype=np.float64)
            return (
                bulk
                + (top - bulk) * np.exp(-zz / tau_top)
                + (bottom - bulk) * np.exp(-(thickness - zz) / tau_bottom)
            )
        return float(
            bulk
            + (top - bulk) * np.exp(-float(z) / tau_top)
            + (bottom - bulk) * np.exp(-(thickness - float(z)) / tau_bottom)
        )

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend(
            [
                self.thickness,
                self.bulk,
                self.top,
                self.bottom,
                self.tau_top,
                self.tau_bottom,
            ]
        )
        return params


class Diffusion(_FieldBase):
    """Depth diffusion solutions; `kind` selects the closed form.

    - ``couple`` (default): planar Fick couple / step IC,
      ``0.5*(left+right) + 0.5*(right-left)*erf((z - T/2)/length)``
      with ``length = 2 sqrt(D t)``.
    - ``exponential``: ``base + (edge - base)*exp(-z/length)``.
    - ``finite``: reserved; raises ``NotImplementedError``.
    """

    def __init__(
        self,
        thickness: float | Parameter,
        *,
        kind: DiffusionKind = "couple",
        left: float | Parameter = 1.0,
        right: float | Parameter = 0.0,
        length: float | Parameter = 6.0,
        edge: float | Parameter = 1.0,
        base: float | Parameter = 0.0,
        name: str = "diffusion",
    ) -> None:
        self.name = name
        self.kind = kind
        self.thickness = _param(thickness, f"{name}_thickness", vary=False)
        self.left = _param(left, f"{name}_left")
        self.right = _param(right, f"{name}_right")
        self.length = _param(length, f"{name}_length", bounds=(1e-6, None))
        self.edge = _param(edge, f"{name}_edge")
        self.base = _param(base, f"{name}_base")

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        thickness = float(kwargs.get("thickness", self.thickness.value or 0.0))
        length = max(float(self.length.value or 1e-6), 1e-6)
        zz = np.asarray(z, dtype=np.float64)
        match self.kind:
            case "couple":
                left = float(self.left.value or 0.0)
                right = float(self.right.value or 0.0)
                mid = 0.5 * thickness
                out = 0.5 * (left + right) + 0.5 * (right - left) * erf(
                    (zz - mid) / length
                )
            case "exponential":
                edge = float(self.edge.value or 0.0)
                base = float(self.base.value or 0.0)
                out = base + (edge - base) * np.exp(-zz / length)
            case "finite":
                msg = 'Diffusion kind="finite" is not implemented yet'
                raise NotImplementedError(msg)
            case _:
                msg = f"unknown Diffusion kind {self.kind!r}"
                raise ValueError(msg)
        if isinstance(z, np.ndarray):
            return np.asarray(out, dtype=np.float64)
        return float(np.asarray(out))

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend(
            [self.thickness, self.left, self.right, self.length, self.edge, self.base]
        )
        return params


class Polynomial(_FieldBase):
    """Polynomial ``sum_k c_k z^k`` vs depth."""

    def __init__(
        self, coeffs: Sequence[float | Parameter], *, name: str = "polynomial"
    ) -> None:
        self.name = name
        self.coeffs = [_param(c, f"{name}_c{i}") for i, c in enumerate(coeffs)]

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        del kwargs
        zz = np.asarray(z, dtype=np.float64)
        out = np.zeros_like(zz, dtype=np.float64)
        for k, coeff in enumerate(self.coeffs):
            out = out + float(coeff.value or 0.0) * zz**k
        if isinstance(z, np.ndarray):
            return out
        return float(out)

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend(self.coeffs)
        return params


class Spline(_FieldBase):
    """Cubic spline through `(knots, values)`."""

    def __init__(
        self,
        knots: Sequence[float],
        values: Sequence[float | Parameter],
        *,
        name: str = "spline",
    ) -> None:
        if len(knots) != len(values):
            msg = "knots and values must have the same length"
            raise ValueError(msg)
        self.name = name
        self.knots = np.asarray(knots, dtype=np.float64)
        self.values = [_param(v, f"{name}_v{i}") for i, v in enumerate(values)]

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        del kwargs
        y = np.array([float(v.value or 0.0) for v in self.values], dtype=np.float64)
        spline = CubicSpline(self.knots, y, bc_type="natural")
        zz = np.asarray(z, dtype=np.float64)
        out = np.asarray(spline(zz), dtype=np.float64)
        if isinstance(z, np.ndarray):
            return out
        return float(out)

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend(self.values)
        return params


class CallableField(_FieldBase):
    """Typed Python escape hatch: ``fn(z, **resolved_params, thickness=...)``."""

    def __init__(
        self,
        fn: Callable[..., float | NDArray[np.float64]],
        params: Mapping[str, Parameter | float] | None = None,
        *,
        name: str = "callable_field",
    ) -> None:
        self.name = name
        self.fn = fn
        self._params = {
            key: _param(value, f"{name}_{key}")
            if not isinstance(value, Parameter)
            else value
            for key, value in (params or {}).items()
        }

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        resolved = {
            key: float(param.value or 0.0) for key, param in self._params.items()
        }
        resolved.update({k: v for k, v in kwargs.items() if k not in resolved})
        return self.fn(z, **resolved)

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend(list(self._params.values()))
        return params


def _mesh_thicknesses(
    thickness: float,
    n_slabs: int,
    mesh: MeshKind,
    mesh_constant: float,
) -> NDArray[np.float64]:
    if mesh == "uniform":
        return np.full(n_slabs, thickness / n_slabs, dtype=np.float64)
    if mesh == "adaptive":
        return adaptive_microslab_thicknesses(thickness, n_slabs, mesh_constant)
    msg = f"unknown mesh {mesh!r}"
    raise ValueError(msg)


def _midpoints(thicknesses: NDArray[np.float64]) -> NDArray[np.float64]:
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    return 0.5 * (edges[:-1] + edges[1:])


def _eval_field(
    field: _FieldBase | float | Parameter | None,
    z: NDArray[np.float64],
    *,
    thickness: float,
) -> NDArray[np.float64] | None:
    if field is None:
        return None
    if isinstance(field, (int, float, Parameter)):
        val = float(field.value if isinstance(field, Parameter) else field)
        return np.full(z.shape, val, dtype=np.float64)
    return np.asarray(field.evaluate(z, thickness=thickness), dtype=np.float64)


def _is_component_sequence(
    value: object,
) -> TypeGuard[list[object] | tuple[object, ...]]:
    return isinstance(value, (list, tuple))


def _as_channel(value: object) -> _FieldBase | float | Parameter | None:
    if value is None or isinstance(value, (Parameter, _FieldBase)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    msg = (
        "profile channel must be a field, number, or Parameter, "
        f"got {type(value).__name__}"
    )
    raise TypeError(msg)


def _eval_shared_channel(
    value: object,
    z: NDArray[np.float64],
    *,
    thickness: float,
) -> NDArray[np.float64] | None:
    if _is_component_sequence(value):
        return None
    return _eval_field(_as_channel(value), z, thickness=thickness)


def _per_component(
    value: object,
    n: int,
    label: str,
) -> list[_FieldBase | float | Parameter | None]:
    """Broadcast one channel, or align a per-component sequence to `n` materials."""
    if value is None:
        return [None] * n
    if _is_component_sequence(value):
        items = [_as_channel(item) for item in value]
        if len(items) != n:
            msg = f"{label} sequence length {len(items)} does not match {n} materials"
            raise ValueError(msg)
        return items
    return [_as_channel(value)] * n


def _profile_parameter_sources(value: object) -> Iterator[tuple[Parameter, bool]]:
    """Parameters of one channel as `(parameter, via_field)` pairs.

    `via_field` is True for Parameters reached through a `_FieldBase`, the only
    source whose shared shape parameters are deduplicated by the caller.
    """
    if value is None:
        return
    if isinstance(value, _FieldBase):
        for param in value.parameters:
            yield param, True
        return
    if isinstance(value, Parameter):
        yield value, False
        return
    if _is_component_sequence(value):
        for item in value:
            yield from _profile_parameter_sources(item)


def _carried_field(value: object) -> _FieldBase | float | Parameter | None:
    """Depth field stored on a scatterer channel, ignoring scalar parameters."""
    if value is None or isinstance(value, (int, float, Parameter)):
        return None
    if isinstance(value, _FieldBase):
        return value
    return None


def _scatterer_depth_fields(material: object) -> dict[str, _FieldBase]:
    """Depth-field-bound peak parameters a scatterer exposes via `depth_fields()`."""
    method = getattr(material, "depth_fields", None)
    if not callable(method):
        return {}
    return dict(method())


def _lab_diagonals(
    n_xx: complex | NDArray[np.complex128],
    n_zz: complex | NDArray[np.complex128],
    cos2: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    cos2_arr = np.asarray(cos2, dtype=np.float64)
    sin2 = 1.0 - cos2_arr
    n_xx_arr = np.asarray(n_xx, dtype=np.complex128)
    n_zz_arr = np.asarray(n_zz, dtype=np.complex128)
    n_o = (n_xx_arr * (1.0 + cos2_arr) + n_zz_arr * sin2) / 2.0
    n_e = n_xx_arr * sin2 + n_zz_arr * cos2_arr
    return n_o, n_e


def _component_lab_channels(
    material: Scatterer,
    energy_ev: float,
    z: NDArray[np.float64],
    thickness: float,
    density_field: _FieldBase | float | Parameter | None,
    gamma_field: _FieldBase | float | Parameter | None,
    cos2_field: _FieldBase | float | Parameter | None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Laboratory diagonals for one mix component at depths `z`.

    A depth field replaces that component's scalar density or rotation.
    With no field, the scatterer's own `tensor_at` keeps its scalars.
    """
    density = _eval_field(density_field, z, thickness=thickness)
    gamma = _eval_field(gamma_field, z, thickness=thickness)
    cos2 = _eval_field(cos2_field, z, thickness=thickness)
    hook = getattr(material, "lab_channels_at_depth", None)
    if callable(hook):
        return cast(
            "tuple[NDArray[np.complex128], NDArray[np.complex128]]",
            hook(energy_ev, z, thickness, density=density, gamma=gamma, cos2=cos2),
        )
    if isinstance(material, UniTensorSLD) and (
        density is not None or gamma is not None or cos2 is not None
    ):
        offset = float(material.energy_offset.value or 0.0)
        n_xx_unit, n_zz_unit = material.ooc.molecular_index_at(energy_ev + offset, 1.0)
        if density is None:
            density = np.full(z.shape, float(material.density.value or 0.0))
        if cos2 is None:
            if gamma is None:
                gamma = np.full(z.shape, float(material.rotation.value or 0.0))
            cos2 = np.cos(np.asarray(gamma, dtype=np.float64)) ** 2
        n_xx = np.asarray(n_xx_unit * density, dtype=np.complex128)
        n_zz = np.asarray(n_zz_unit * density, dtype=np.complex128)
        return _lab_diagonals(n_xx, n_zz, np.asarray(cos2, dtype=np.float64))
    base = material.tensor_at(energy_ev)
    n_o = np.full(z.shape, complex(base[0, 0]), dtype=np.complex128)
    n_e = np.full(z.shape, complex(base[2, 2]), dtype=np.complex128)
    if density is not None:
        n_o = n_o * density
        n_e = n_e * density
    if gamma is not None or cos2 is not None:
        if cos2 is None:
            cos2 = np.cos(np.asarray(gamma, dtype=np.float64)) ** 2
        n_o, n_e = _lab_diagonals(n_o, n_e, cos2)
    return n_o, n_e


class DepthProfile(MultiRowComponent):
    """Microslab segment driven by scalar depth fields on a material handle.

    Channel kwargs select which observable each field drives: `gamma`,
    `cos2_gamma`, `density`, `phi`, and free optical `delta_o`/`delta_e`/
    `beta_o`/`beta_e`. A channel accepts one field, one number, one
    `Parameter`, or a sequence aligned with `Mix.materials`.

    On a `Mix`, `phi` is applied together with each component's own
    `density` and `rotation`. Those channels live on the `UniTensorSLD`:
    pass a `SecondOrderTransition` (or any other depth field) as
    `density=` or `rotation=` and this profile evaluates it at every
    microslab before mixing. A single `phi` field is the volume fraction
    of `materials[0]` (exactly two components). `phi` as a sequence is one
    weight profile per component, clipped at 0 and renormalized to sum to 1.
    Attaching `phi` freezes `Mix.vf`, because those homogeneous fractions
    are not the evaluated composition.

    `density_scale` multiplies the final tensors of every slice after mixing
    (the mix-level total density). A scatterer with `lab_channels_at_depth`
    (such as `PeakSumSLD`) receives this profile's `density`, `gamma` and
    `cos2_gamma` per slice, and its depth-field peak parameters are evaluated
    at each slice.
    """

    def __init__(
        self,
        material: Scatterer,
        thickness: float | Parameter,
        roughness: float | Parameter = 0.0,
        *,
        n_slabs: int = 40,
        mesh: MeshKind = "adaptive",
        mesh_constant: float = 1.5,
        gamma: _FieldBase
        | float
        | Parameter
        | Sequence[_FieldBase | float | Parameter]
        | None = None,
        cos2_gamma: _FieldBase
        | float
        | Parameter
        | Sequence[_FieldBase | float | Parameter]
        | None = None,
        density: _FieldBase
        | float
        | Parameter
        | Sequence[_FieldBase | float | Parameter]
        | None = None,
        phi: _FieldBase
        | float
        | Parameter
        | Sequence[_FieldBase | float | Parameter]
        | None = None,
        delta_o: _FieldBase | float | Parameter | None = None,
        delta_e: _FieldBase | float | Parameter | None = None,
        beta_o: _FieldBase | float | Parameter | None = None,
        beta_e: _FieldBase | float | Parameter | None = None,
        density_scale: _FieldBase | float | Parameter | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or getattr(material, "name", "depth_profile"))
        self.material = material
        self.thick = _param(
            thickness, f"{self.name}_thick", vary=True, bounds=(0.0, None)
        )
        self.rough = _param(
            roughness, f"{self.name}_rough", vary=True, bounds=(0.0, None)
        )
        self.n_slabs = int(n_slabs)
        self.mesh = mesh
        self.mesh_constant = float(mesh_constant)
        self.gamma = gamma
        self.cos2_gamma = cos2_gamma
        self.density = density
        self.density_scale = _as_channel(density_scale)
        self.phi = phi
        self.delta_o = delta_o
        self.delta_e = delta_e
        self.beta_o = beta_o
        self.beta_e = beta_e
        if gamma is not None and cos2_gamma is not None:
            msg = "DepthProfile accepts at most one of gamma and cos2_gamma"
            raise ValueError(msg)
        if self.phi is not None and isinstance(self.material, Mix):
            for fraction in self.material.vf:
                fraction.vary = False

    def geometric_thickness(self) -> float:
        return float(self.thick.value or 0.0)

    def channel_profile(
        self, channel: str, *, n: int = 400
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate a named channel on a uniform depth grid for inspection."""
        thickness = self.geometric_thickness()
        z = np.linspace(0.0, thickness, n)
        field = getattr(self, channel, None)
        values = _eval_field(field, z, thickness=thickness)
        if values is None:
            msg = f"DepthProfile has no field for channel {channel!r}"
            raise ValueError(msg)
        return z, values

    def named_values_at(self, z: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Evaluate every configured channel at local depths `z` (Å from top)."""
        thickness = self.geometric_thickness()
        out: dict[str, NDArray[np.float64]] = {}
        mapping = (
            ("density", self.density),
            ("density_scale", self.density_scale),
            ("orientation", self.gamma),
            ("cos2_gamma", self.cos2_gamma),
            ("phi", self.phi),
            ("delta_o", self.delta_o),
            ("delta_e", self.delta_e),
            ("beta_o", self.beta_o),
            ("beta_e", self.beta_e),
        )
        for key, field in mapping:
            if _is_component_sequence(field):
                for index, item in enumerate(field):
                    values = _eval_field(_as_channel(item), z, thickness=thickness)
                    if values is not None:
                        out[f"{key}_{index}"] = values
                continue
            values = _eval_shared_channel(field, z, thickness=thickness)
            if values is not None:
                out[key] = values
        materials = getattr(self.material, "materials", ())
        for index, material in enumerate(materials):
            for key, attr in (("density", "density"), ("orientation", "rotation")):
                label = f"{key}_{index}"
                if label in out:
                    continue
                carried = _carried_field(getattr(material, attr, None))
                if carried is None:
                    continue
                values = _eval_field(carried, z, thickness=thickness)
                if values is not None:
                    out[label] = values
        depth_owner: dict[str, object] = {}
        for scatterer in (self.material, *materials):
            for key, field in _scatterer_depth_fields(scatterer).items():
                if key in depth_owner and depth_owner[key] is not field:
                    msg = (
                        f"two depth fields share the address {key!r}; give each "
                        "scatterer (for example each PeakSumSLD) a unique name"
                    )
                    raise ValueError(msg)
                depth_owner[key] = field
                values = _eval_field(field, z, thickness=thickness)
                if values is not None:
                    out[key] = values
        return out

    def _thicknesses(self) -> NDArray[np.float64]:
        return _mesh_thicknesses(
            self.geometric_thickness(), self.n_slabs, self.mesh, self.mesh_constant
        )

    def _components_carry_depth_fields(self) -> bool:
        materials = getattr(self.material, "materials", ())
        if any(_scatterer_depth_fields(material) for material in materials):
            return True
        return any(
            _carried_field(getattr(material, attr, None)) is not None
            for material in materials
            for attr in ("density", "rotation")
        )

    def _mixes_component_channels(self) -> bool:
        channels = (self.phi, self.density, self.gamma, self.cos2_gamma)
        if any(_is_component_sequence(channel) for channel in channels):
            return True
        if self._components_carry_depth_fields():
            return True
        return any(
            channel is not None
            for channel in (self.density, self.gamma, self.cos2_gamma)
        )

    def _mixed_component_tensors(
        self,
        energy_ev: float,
        z: NDArray[np.float64],
        thickness: float,
    ) -> NDArray[np.complex128]:
        from refloxide.mixing import mix_channels, pack_lab_diagonal

        mix = cast("Mix", self.material)
        materials = mix.materials
        n_materials = len(materials)
        densities = _per_component(self.density, n_materials, "density")
        gammas = _per_component(self.gamma, n_materials, "gamma")
        cos2s = _per_component(self.cos2_gamma, n_materials, "cos2_gamma")
        if _is_component_sequence(self.phi):
            raw: list[NDArray[np.float64]] = []
            for field in self.phi:
                values = _eval_field(
                    cast("_FieldBase | float | Parameter | None", field),
                    z,
                    thickness=thickness,
                )
                if values is None:
                    msg = "each phi entry must be a field, number, or Parameter"
                    raise ValueError(msg)
                raw.append(np.clip(values, 0.0, None))
            stacked = np.stack(raw, axis=0)
            total = stacked.sum(axis=0)
            total = np.where(total == 0.0, 1.0, total)
            fractions = [stacked[i] / total for i in range(n_materials)]
        else:
            phi = _eval_shared_channel(self.phi, z, thickness=thickness)
            if phi is not None:
                if n_materials != 2:
                    msg = "a single phi field requires exactly two Mix materials"
                    raise ValueError(msg)
                fractions = [phi, 1.0 - phi]
            else:
                fractions = [
                    np.full(z.shape, float(fraction.value or 0.0))
                    for fraction in mix.vf
                ]
        n_o_parts: list[NDArray[np.complex128]] = []
        n_e_parts: list[NDArray[np.complex128]] = []
        for material, density, gamma, cos2 in zip(
            materials, densities, gammas, cos2s, strict=True
        ):
            if density is None:
                density = _carried_field(getattr(material, "density", None))
            if gamma is None:
                gamma = _carried_field(getattr(material, "rotation", None))
            n_o, n_e = _component_lab_channels(
                material, energy_ev, z, thickness, density, gamma, cos2
            )
            n_o_parts.append(n_o)
            n_e_parts.append(n_e)
        return pack_lab_diagonal(
            mix_channels(
                n_o_parts, fractions, rule=mix.rule, host_index=mix.host_index
            ),
            mix_channels(
                n_e_parts, fractions, rule=mix.rule, host_index=mix.host_index
            ),
        )

    def rows_and_tensors_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        thicknesses = self._thicknesses()
        z = _midpoints(thicknesses)
        thickness = self.geometric_thickness()
        tensors = self._tensors_at_z(energy_ev, z, thickness)
        n_avg = (tensors[:, 0, 0] + tensors[:, 1, 1] + tensors[:, 2, 2]) / 3.0
        rows = np.empty((tensors.shape[0], 4), dtype=np.float64)
        rows[:, 0] = thicknesses
        rows[:, 1] = n_avg.real
        rows[:, 2] = n_avg.imag
        rows[:, 3] = 0.0
        rows[0, 3] = float(self.rough.value or 0.0)
        return rows, tensors

    def _tensors_at_z(
        self,
        energy_ev: float,
        z: NDArray[np.float64],
        thickness: float,
    ) -> NDArray[np.complex128]:
        tensors = self._tensors_at_z_unscaled(energy_ev, z, thickness)
        scale = _eval_shared_channel(self.density_scale, z, thickness=thickness)
        if scale is None:
            return tensors
        return tensors * scale[:, None, None]

    def _tensors_at_z_unscaled(
        self,
        energy_ev: float,
        z: NDArray[np.float64],
        thickness: float,
    ) -> NDArray[np.complex128]:
        gamma = _eval_shared_channel(self.gamma, z, thickness=thickness)
        cos2 = _eval_shared_channel(self.cos2_gamma, z, thickness=thickness)
        density = _eval_shared_channel(self.density, z, thickness=thickness)
        phi = _eval_shared_channel(self.phi, z, thickness=thickness)
        delta_o = _eval_field(self.delta_o, z, thickness=thickness)
        delta_e = _eval_field(self.delta_e, z, thickness=thickness)
        beta_o = _eval_field(self.beta_o, z, thickness=thickness)
        beta_e = _eval_field(self.beta_e, z, thickness=thickness)
        free_optical = delta_o is not None or delta_e is not None
        if isinstance(self.material, Mix) and self._mixes_component_channels():
            if free_optical:
                msg = (
                    "DepthProfile cannot combine free optical diagonals "
                    "with a Mix profile"
                )
                raise ValueError(msg)
            return self._mixed_component_tensors(energy_ev, z, thickness)

        if free_optical and callable(
            getattr(self.material, "lab_channels_at_depth", None)
        ):
            msg = (
                "DepthProfile cannot combine free optical diagonals "
                "with a peak-sum source"
            )
            raise ValueError(msg)

        # Free optical polynomial/spline path
        if free_optical:
            base = self.material.tensor_at(energy_ev)
            n_o0 = complex(base[0, 0])
            n_e0 = complex(base[2, 2])
            n_o = (
                delta_o.astype(np.complex128)
                + 1j * (beta_o if beta_o is not None else np.full_like(z, n_o0.imag))
                if delta_o is not None
                else np.full(z.shape, n_o0, dtype=np.complex128)
            )
            n_e = (
                delta_e.astype(np.complex128)
                + 1j * (beta_e if beta_e is not None else np.full_like(z, n_e0.imag))
                if delta_e is not None
                else np.full(z.shape, n_e0, dtype=np.complex128)
            )
            if density is not None:
                n_o = n_o * density
                n_e = n_e * density
            if gamma is not None or cos2 is not None:
                if cos2 is None:
                    cos2 = np.cos(np.asarray(gamma, dtype=np.float64)) ** 2
                n_o, n_e = _lab_diagonals(n_o, n_e, cos2)
            return pack_lab_diagonal(n_o, n_e)

        mix_fn = getattr(self.material, "mix_tensors_at", None)
        if phi is not None and callable(mix_fn):
            return cast("NDArray[np.complex128]", mix_fn(energy_ev, phi))

        hook = getattr(self.material, "lab_channels_at_depth", None)
        if callable(hook):
            n_o, n_e = hook(
                energy_ev, z, thickness, density=density, gamma=gamma, cos2=cos2
            )
            return pack_lab_diagonal(n_o, n_e)

        if isinstance(self.material, UniTensorSLD):
            uni = self.material
            if density is None:
                carried_density = _carried_field(uni.density)
                if carried_density is not None:
                    density = _eval_field(carried_density, z, thickness=thickness)
            if gamma is None and cos2 is None:
                carried_rotation = _carried_field(uni.rotation)
                if carried_rotation is not None:
                    gamma = _eval_field(carried_rotation, z, thickness=thickness)
            if density is None:
                dens = np.full(z.shape, uni._channel_scalar(uni.density))
            else:
                dens = density
            eo = float(uni.energy_offset.value or 0.0)
            eff = energy_ev + eo
            tensors = np.empty((z.size, 3, 3), dtype=np.complex128)
            for i in range(z.size):
                n_xx, n_zz = uni.ooc.molecular_index_at(eff, float(dens[i]))
                if cos2 is not None:
                    tensors[i] = lab_tensor_from_order_parameter(
                        n_xx, n_zz, float(cos2[i]), kind="cos2_gamma"
                    )
                else:
                    g = (
                        float(gamma[i])
                        if gamma is not None
                        else uni._channel_scalar(uni.rotation)
                    )
                    tensors[i] = np.asarray(
                        optics.uniaxial_lab_tensor(n_xx, n_zz, g),
                        dtype=np.complex128,
                    )
            return tensors

        base = self.material.tensor_at(energy_ev)
        n_o = complex(base[0, 0])
        n_e = complex(base[2, 2])
        if density is not None:
            n_o_arr = np.asarray(n_o * density, dtype=np.complex128)
            n_e_arr = np.asarray(n_e * density, dtype=np.complex128)
        else:
            n_o_arr = np.full(z.shape, n_o, dtype=np.complex128)
            n_e_arr = np.full(z.shape, n_e, dtype=np.complex128)
        if cos2 is not None:
            n_o_lab, n_e_lab = _lab_diagonals(n_o_arr, n_e_arr, cos2)
            return pack_lab_diagonal(n_o_lab, n_e_lab)
        if gamma is not None:
            tensors = np.empty((z.size, 3, 3), dtype=np.complex128)
            for i in range(z.size):
                tensors[i] = np.asarray(
                    optics.uniaxial_lab_tensor(n_o_arr[i], n_e_arr[i], float(gamma[i])),
                    dtype=np.complex128,
                )
            return tensors
        return pack_lab_diagonal(n_o_arr, n_e_arr)

    def slab_rows_at(self, energy_ev: float) -> NDArray[np.float64]:
        return self.rows_and_tensors_at(energy_ev)[0]

    def tensor_rows_at(self, energy_ev: float) -> NDArray[np.complex128]:
        return self.rows_and_tensors_at(energy_ev)[1]

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend([self.thick, self.rough])
        params.extend(list(self.material.parameters))
        # Shared shapes (one TransitionShape under several LevelFields) yield
        # the same Parameter objects through several fields; list those once.
        # Directly supplied Parameters and material parameters are never deduped.
        seen_via_field: set[int] = set()
        for field in (
            self.gamma,
            self.cos2_gamma,
            self.density,
            self.density_scale,
            self.phi,
            self.delta_o,
            self.delta_e,
            self.beta_o,
            self.beta_e,
        ):
            for param, via_field in _profile_parameter_sources(field):
                if via_field:
                    if id(param) in seen_via_field:
                        continue
                    seen_via_field.add(id(param))
                params.extend([param])
        return params

    def __repr__(self) -> str:
        return f"DepthProfile(material={self.material!r}, name={self.name!r})"


class CallableDepthProfile(MultiRowComponent):
    """User tensor profile: ``fn(z, material=..., energy_ev=..., **params)``."""

    def __init__(
        self,
        fn: Callable[..., NDArray[np.complex128]],
        *,
        material: Scatterer,
        params: Mapping[str, Parameter | float] | None = None,
        thickness: float | Parameter,
        roughness: float | Parameter = 0.0,
        n_slabs: int = 60,
        mesh: MeshKind = "uniform",
        mesh_constant: float = 1.5,
        name: str = "",
    ) -> None:
        super().__init__(name=name or getattr(material, "name", "callable_depth"))
        self.fn = fn
        self.material = material
        self.thick = _param(
            thickness, f"{self.name}_thick", vary=True, bounds=(0.0, None)
        )
        self.rough = _param(
            roughness, f"{self.name}_rough", vary=True, bounds=(0.0, None)
        )
        self.n_slabs = int(n_slabs)
        self.mesh = mesh
        self.mesh_constant = float(mesh_constant)
        self._params = {
            key: _param(value, f"{self.name}_{key}")
            if not isinstance(value, Parameter)
            else value
            for key, value in (params or {}).items()
        }

    def geometric_thickness(self) -> float:
        return float(self.thick.value or 0.0)

    def rows_and_tensors_at(
        self, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        thicknesses = _mesh_thicknesses(
            self.geometric_thickness(), self.n_slabs, self.mesh, self.mesh_constant
        )
        z = _midpoints(thicknesses)
        resolved = {
            key: float(param.value or 0.0) for key, param in self._params.items()
        }
        tensors = np.asarray(
            self.fn(
                z,
                material=self.material,
                energy_ev=energy_ev,
                thickness=self.geometric_thickness(),
                **resolved,
            ),
            dtype=np.complex128,
        )
        if tensors.ndim == 2:
            tensors = np.broadcast_to(tensors, (z.size, 3, 3)).copy()
        n_avg = (tensors[:, 0, 0] + tensors[:, 1, 1] + tensors[:, 2, 2]) / 3.0
        rows = np.empty((tensors.shape[0], 4), dtype=np.float64)
        rows[:, 0] = thicknesses
        rows[:, 1] = n_avg.real
        rows[:, 2] = n_avg.imag
        rows[:, 3] = 0.0
        rows[0, 3] = float(self.rough.value or 0.0)
        return rows, tensors

    def slab_rows_at(self, energy_ev: float) -> NDArray[np.float64]:
        return self.rows_and_tensors_at(energy_ev)[0]

    def tensor_rows_at(self, energy_ev: float) -> NDArray[np.complex128]:
        return self.rows_and_tensors_at(energy_ev)[1]

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend([self.thick, self.rough])
        params.extend(list(self.material.parameters))
        params.extend(list(self._params.values()))
        return params
