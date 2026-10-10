"""Depth field forms and `DepthProfile` stack segments.

Owns inspectable scalar fields (`Constant`, `SecondOrderTransition`,
`Diffusion`, `Polynomial`, `Spline`, `CallableField`) and the multi-row
`DepthProfile` / `CallableDepthProfile` components that sample them onto a
microslab mesh for the uniaxial TMM. Does not own tabular OOC loading or the
reflectivity kernel.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from scipy.interpolate import CubicSpline
from scipy.special import erf

from refloxide import optics
from refloxide.mixing import pack_lab_diagonal
from refloxide.model import MultiRowComponent, Scatterer, UniTensorSLD
from refloxide.pxr.energy.bookended import adaptive_microslab_thicknesses
from refloxide.sources import lab_diagonal_from_cos2

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

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


class DepthProfile(MultiRowComponent):
    """Microslab segment driven by scalar depth fields on a material handle.

    Channel kwargs select which observable each field drives: `gamma`,
    `cos2_gamma`, `density`, `phi`, and free optical `delta_o`/`delta_e`/
    `beta_o`/`beta_e`.
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
        gamma: _FieldBase | float | Parameter | None = None,
        cos2_gamma: _FieldBase | float | Parameter | None = None,
        density: _FieldBase | float | Parameter | None = None,
        phi: _FieldBase | float | Parameter | None = None,
        delta_o: _FieldBase | float | Parameter | None = None,
        delta_e: _FieldBase | float | Parameter | None = None,
        beta_o: _FieldBase | float | Parameter | None = None,
        beta_e: _FieldBase | float | Parameter | None = None,
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
        self.phi = phi
        self.delta_o = delta_o
        self.delta_e = delta_e
        self.beta_o = beta_o
        self.beta_e = beta_e
        if gamma is not None and cos2_gamma is not None:
            msg = "DepthProfile accepts at most one of gamma and cos2_gamma"
            raise ValueError(msg)

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
            ("orientation", self.gamma),
            ("cos2_gamma", self.cos2_gamma),
            ("phi", self.phi),
            ("delta_o", self.delta_o),
            ("delta_e", self.delta_e),
            ("beta_o", self.beta_o),
            ("beta_e", self.beta_e),
        )
        for key, field in mapping:
            values = _eval_field(field, z, thickness=thickness)
            if values is not None:
                out[key] = values
        return out

    def _thicknesses(self) -> NDArray[np.float64]:
        return _mesh_thicknesses(
            self.geometric_thickness(), self.n_slabs, self.mesh, self.mesh_constant
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
        gamma = _eval_field(self.gamma, z, thickness=thickness)
        cos2 = _eval_field(self.cos2_gamma, z, thickness=thickness)
        density = _eval_field(self.density, z, thickness=thickness)
        phi = _eval_field(self.phi, z, thickness=thickness)
        delta_o = _eval_field(self.delta_o, z, thickness=thickness)
        delta_e = _eval_field(self.delta_e, z, thickness=thickness)
        beta_o = _eval_field(self.beta_o, z, thickness=thickness)
        beta_e = _eval_field(self.beta_e, z, thickness=thickness)

        # Free optical polynomial/spline path
        if delta_o is not None or delta_e is not None:
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
            return pack_lab_diagonal(n_o, n_e)

        mix_fn = getattr(self.material, "mix_tensors_at", None)
        if phi is not None and callable(mix_fn):
            return cast("NDArray[np.complex128]", mix_fn(energy_ev, phi))

        if isinstance(self.material, UniTensorSLD):
            uni = self.material
            dens0 = float(uni.density.value or 1.0)
            dens = density if density is not None else np.full(z.shape, dens0)
            rot0 = float(uni.rotation.value or 0.0)
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
                    g = float(gamma[i]) if gamma is not None else rot0
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
            return lab_tensor_from_order_parameter(
                complex(np.mean(n_o_arr)),
                complex(np.mean(n_e_arr)),
                cos2,
                kind="cos2_gamma",
            )
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
        for field in (
            self.gamma,
            self.cos2_gamma,
            self.density,
            self.phi,
            self.delta_o,
            self.delta_e,
            self.beta_o,
            self.beta_e,
        ):
            if isinstance(field, _FieldBase):
                params.extend(list(field.parameters))
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
