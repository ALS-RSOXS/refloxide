"""Shape-plus-level depth fields, shared shapes, and derived channels.

A `LevelField` pairs a `Shape` (nonlinear shape parameters such as taus,
knot-free family settings, wavelength, or width) with its own per-channel
level `Parameter`s. Several level fields may share one `Shape` object, in
which case the shape parameters are one set in the refnx `Parameters` tree.

Built-in shapes are linear in their levels, so each one exposes a `basis`
(``(n_levels, n_z)`` weights) that is cached on the shape parameter values,
thickness and depth array. A shape without a basis overrides `evaluate`.
`Derived` combines channels pointwise with a user function. The existing
scalar fields in `refloxide.profiles` are unchanged and remain valid
channels.
"""

from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Parameter, Parameters
from scipy.interpolate import CubicSpline

from refloxide.profiles import _FieldBase, _param

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "Derived",
    "LevelField",
    "LocalizedShape",
    "PolynomialShape",
    "Shape",
    "SinusoidShape",
    "SplineShape",
    "TransitionShape",
    "cos2_of",
]

_BASIS_CACHE_SIZE = 8
_MIN_LENGTH = 1e-6


def _require_thickness(kwargs: dict[str, Any]) -> float:
    if "thickness" not in kwargs or kwargs["thickness"] is None:
        msg = "this field needs thickness=... (DepthProfile supplies it)"
        raise TypeError(msg)
    thickness = kwargs["thickness"]
    if isinstance(thickness, Parameter):
        return float(thickness.value or 0.0)
    return float(thickness)


class Shape(ABC):
    """Depth shape shared by one or more `LevelField`s.

    Subclasses set `level_names`, return their shape parameters from
    `_shape_parameters`, and override `_build_basis` (linear families) or
    `evaluate` (nonlinear families).
    """

    name: str
    level_names: tuple[str, ...]

    def __init__(self, *, name: str) -> None:
        self.name = name
        self._basis_cache: OrderedDict[tuple[Any, ...], NDArray[np.float64]] = (
            OrderedDict()
        )

    def _shape_parameters(self) -> list[Parameter]:
        return []

    @property
    def parameters(self) -> Parameters:
        """Shape parameters only (level parameters belong to each `LevelField`)."""
        params = Parameters(name=self.name)
        params.extend(self._shape_parameters())
        return params

    def _build_basis(
        self, z: NDArray[np.float64], thickness: float
    ) -> NDArray[np.float64] | None:
        del z, thickness
        return None

    def basis(
        self, z: NDArray[np.float64], *, thickness: float
    ) -> NDArray[np.float64] | None:
        """Return ``(n_levels, n_z)`` weights with ``value(z) = levels @ basis``.

        Returns None when the shape is not linear in its levels. Results are
        cached on the current shape-parameter values, `thickness`, and the
        depth array, so a changed tau, width or depth array is never stale.
        The returned array is read-only.
        """
        zz = np.asarray(z, dtype=np.float64).ravel()
        thick = float(thickness)
        key = (
            tuple(float(p.value or 0.0) for p in self._shape_parameters()),
            thick,
            zz.tobytes(),
        )
        cached = self._basis_cache.get(key)
        if cached is not None:
            self._basis_cache.move_to_end(key)
            return cached
        built = self._build_basis(zz, thick)
        if built is None:
            return None
        built = np.asarray(built, dtype=np.float64)
        built.setflags(write=False)
        self._basis_cache[key] = built
        if len(self._basis_cache) > _BASIS_CACHE_SIZE:
            self._basis_cache.popitem(last=False)
        return built

    def evaluate(
        self,
        z: NDArray[np.float64],
        levels: NDArray[np.float64],
        *,
        thickness: float,
    ) -> NDArray[np.float64]:
        """Evaluate the shape at `z` for the given level values.

        The default is ``levels @ basis(z)``. Shapes without a basis must
        override this method.
        """
        zz = np.asarray(z, dtype=np.float64)
        weights = self.basis(zz, thickness=thickness)
        if weights is None:
            msg = f"{type(self).__name__} has no basis; override evaluate"
            raise NotImplementedError(msg)
        values = np.asarray(levels, dtype=np.float64) @ weights
        return values.reshape(zz.shape)


class LevelField(_FieldBase):
    """A `Shape` with this channel's own level values.

    Parameters
    ----------
    shape
        Shape object; may be shared with other `LevelField`s.
    levels
        One value or `Parameter` per shape level, in `shape.level_names` order.
    name
        Prefix for the level `Parameter` names.
    """

    def __init__(
        self,
        shape: Shape,
        levels: Sequence[float | Parameter],
        *,
        name: str,
    ) -> None:
        if len(levels) != len(shape.level_names):
            msg = (
                f"{type(shape).__name__} takes {len(shape.level_names)} levels "
                f"{shape.level_names}, got {len(levels)}"
            )
            raise ValueError(msg)
        self.name = name
        self.shape = shape
        self.levels = [
            _param(value, f"{name}_{level}")
            for value, level in zip(levels, shape.level_names, strict=True)
        ]

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        thickness = _require_thickness(kwargs)
        zz = np.asarray(z, dtype=np.float64)
        values = np.array(
            [float(level.value or 0.0) for level in self.levels], dtype=np.float64
        )
        out = self.shape.evaluate(zz, values, thickness=thickness)
        if zz.ndim == 0:
            return float(out)
        return out

    def plot(
        self,
        ax: Axes | None = None,
        *,
        n: int = 400,
        thickness: float | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot vs depth on `[0, thickness]`; `thickness` must be given.

        Remaining keyword arguments go to `Axes.plot`.
        """
        return _plot_depth_field(self, ax, n=n, thickness=thickness, **kwargs)

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        params.extend(self.levels)
        params.extend(self.shape.parameters.flattened())
        return params


def _plot_depth_field(
    field: _FieldBase,
    ax: Axes | None,
    *,
    n: int,
    thickness: float | None,
    **kwargs: Any,
) -> Axes:
    """Plot `field` on `[0, thickness]`; `thickness` is required (no silent default)."""
    if thickness is None:
        msg = f"{type(field).__name__}.plot needs thickness=... (the depth range)"
        raise ValueError(msg)
    if isinstance(thickness, Parameter):
        thickness = float(thickness.value or 0.0)
    depth = float(thickness)
    z = np.linspace(0.0, depth, n)
    y = np.asarray(field.evaluate(z, thickness=depth), dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(z, y, **kwargs)
    ax.set_xlabel(r"depth $z$ ($\mathrm{\AA}$)")
    ax.set_ylabel(field.name or "field")
    return ax


def _channel_values(
    channel: _FieldBase | float | Parameter,
    z: NDArray[np.float64],
    kwargs: dict[str, Any],
) -> NDArray[np.float64]:
    if isinstance(channel, _FieldBase):
        return np.asarray(channel.evaluate(z, **kwargs), dtype=np.float64)
    value = (
        float(channel.value or 0.0)
        if isinstance(channel, Parameter)
        else float(channel)
    )
    return np.full(z.shape, value, dtype=np.float64)


class Derived(_FieldBase):
    """Pointwise function of other channels, evaluated at each depth.

    Parameters
    ----------
    fn
        Called as ``fn(**arrays)`` with one array per input, each shaped like
        the depth array.
    name
        Field name used in plots and parameter trees.
    **inputs
        Channels: a `_FieldBase`, a number, or a `Parameter`.
    """

    def __init__(
        self,
        fn: Callable[..., NDArray[np.float64]],
        /,
        *,
        name: str = "derived",
        **inputs: _FieldBase | float | Parameter,
    ) -> None:
        for key, channel in inputs.items():
            if not isinstance(
                channel, (_FieldBase, Parameter, int, float)
            ) or isinstance(channel, bool):
                msg = (
                    f"Derived input {key!r} must be a field, number, or Parameter, "
                    f"got {type(channel).__name__}"
                )
                raise TypeError(msg)
        self.name = name
        self.fn = fn
        self.inputs = dict(inputs)

    def evaluate(
        self, z: float | NDArray[np.float64], **kwargs: Any
    ) -> float | NDArray[np.float64]:
        zz = np.asarray(z, dtype=np.float64)
        arrays = {
            key: _channel_values(channel, zz, kwargs)
            for key, channel in self.inputs.items()
        }
        out = np.asarray(self.fn(**arrays))
        if zz.ndim == 0:
            return out.reshape(()).item()
        return out.reshape(zz.shape)

    def plot(
        self,
        ax: Axes | None = None,
        *,
        n: int = 400,
        thickness: float | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot vs depth on `[0, thickness]`; `thickness` must be given.

        Remaining keyword arguments go to `Axes.plot`.
        """
        return _plot_depth_field(self, ax, n=n, thickness=thickness, **kwargs)

    @property
    def parameters(self) -> Parameters:
        params = Parameters(name=self.name)
        seen: set[int] = set()
        unique: list[Parameter] = []
        for channel in self.inputs.values():
            if isinstance(channel, _FieldBase):
                items = channel.parameters.flattened()
            elif isinstance(channel, Parameter):
                items = [channel]
            else:
                items = []
            for item in items:
                if id(item) not in seen:
                    seen.add(id(item))
                    unique.append(item)
        params.extend(unique)
        return params


def cos2_of(gamma: _FieldBase | float | Parameter) -> Derived:
    """Derived ``<cos^2 gamma>``-style channel: ``cos(gamma)**2`` (gamma in rad)."""
    return Derived(
        lambda gamma: np.cos(gamma) ** 2,
        name="cos2_gamma",
        gamma=gamma,
    )


class TransitionShape(Shape):
    """Two-sided exponential relaxation with `top`, `bulk`, `bottom` levels.

    ``value(z) = top*e_top + bulk*(1 - e_top - e_bot) + bottom*e_bot`` with
    ``e_top = exp(-z/tau_top)`` and ``e_bot = exp(-(T - z)/tau_bottom)``.
    Same function as `SecondOrderTransition`. Taus are clamped to >= 1e-6 Å.
    """

    level_names = ("top", "bulk", "bottom")

    def __init__(
        self,
        tau_top: float | Parameter,
        tau_bottom: float | Parameter,
        *,
        name: str = "transition",
    ) -> None:
        super().__init__(name=name)
        self.tau_top = _param(tau_top, f"{name}_tau_top", bounds=(1e-6, None))
        self.tau_bottom = _param(tau_bottom, f"{name}_tau_bottom", bounds=(1e-6, None))

    def _shape_parameters(self) -> list[Parameter]:
        return [self.tau_top, self.tau_bottom]

    def _build_basis(
        self, z: NDArray[np.float64], thickness: float
    ) -> NDArray[np.float64]:
        tau_top = max(float(self.tau_top.value or _MIN_LENGTH), _MIN_LENGTH)
        tau_bottom = max(float(self.tau_bottom.value or _MIN_LENGTH), _MIN_LENGTH)
        e_top = np.exp(-z / tau_top)
        e_bottom = np.exp(-(thickness - z) / tau_bottom)
        return np.stack([e_top, 1.0 - e_top - e_bottom, e_bottom])


class SplineShape(Shape):
    """Natural cubic spline through fixed `knots` (depths in Å).

    Levels are the knot values.
    """

    def __init__(self, knots: Sequence[float], *, name: str = "spline") -> None:
        super().__init__(name=name)
        # Copy: a caller's array mutated later must not change this shape.
        knot_arr = np.array(knots, dtype=np.float64)
        if knot_arr.ndim != 1 or knot_arr.size < 2:
            msg = "SplineShape needs at least two knots"
            raise ValueError(msg)
        if not np.all(np.diff(knot_arr) > 0.0):
            msg = "SplineShape knots must be strictly increasing"
            raise ValueError(msg)
        self.knots = knot_arr
        self.level_names = tuple(f"knot_{i}" for i in range(knot_arr.size))

    def _build_basis(
        self, z: NDArray[np.float64], thickness: float
    ) -> NDArray[np.float64]:
        del thickness
        k = self.knots.size
        spline = CubicSpline(self.knots, np.eye(k), axis=0, bc_type="natural")
        return np.asarray(spline(z), dtype=np.float64).T


class PolynomialShape(Shape):
    """Polynomial of `degree` through values at `positions` (fractions of thickness).

    Levels are the values at the anchors; the basis is ``V(u) @ inv(V(anchors))``
    with ``u = z / thickness`` and ``V`` the Vandermonde matrix. At thickness 0
    every depth maps to ``u = 0`` so the values stay finite.
    """

    def __init__(
        self,
        degree: int,
        positions: Sequence[float] | None = None,
        *,
        name: str = "polynomial",
    ) -> None:
        super().__init__(name=name)
        if degree < 0:
            msg = "PolynomialShape degree must be >= 0"
            raise ValueError(msg)
        pos = (
            np.linspace(0.0, 1.0, degree + 1)
            if positions is None
            else np.array(positions, dtype=np.float64)
        )
        if pos.shape != (degree + 1,):
            msg = f"PolynomialShape of degree {degree} needs {degree + 1} positions"
            raise ValueError(msg)
        if len(set(pos.tolist())) != pos.size:
            msg = "PolynomialShape positions must be distinct"
            raise ValueError(msg)
        self.degree = int(degree)
        self.positions = pos
        self.level_names = tuple(f"value_{i}" for i in range(degree + 1))
        self._anchor_inverse = np.linalg.inv(
            np.vander(pos, degree + 1, increasing=True)
        )

    def _build_basis(
        self, z: NDArray[np.float64], thickness: float
    ) -> NDArray[np.float64]:
        u = z / thickness if thickness > 0.0 else np.zeros_like(z)
        vander = np.vander(u, self.degree + 1, increasing=True)
        return (vander @ self._anchor_inverse).T


class SinusoidShape(Shape):
    """Sinusoid ``offset + amplitude*sin(2*pi*z/wavelength + phase)``.

    Levels are (offset, amplitude).
    """

    level_names = ("offset", "amplitude")

    def __init__(
        self,
        wavelength: float | Parameter,
        phase: float | Parameter = 0.0,
        *,
        name: str = "sinusoid",
    ) -> None:
        super().__init__(name=name)
        self.wavelength = _param(wavelength, f"{name}_wavelength", bounds=(1e-6, None))
        self.phase = _param(phase, f"{name}_phase")

    def _shape_parameters(self) -> list[Parameter]:
        return [self.wavelength, self.phase]

    def _build_basis(
        self, z: NDArray[np.float64], thickness: float
    ) -> NDArray[np.float64]:
        del thickness
        wavelength = max(float(self.wavelength.value or _MIN_LENGTH), _MIN_LENGTH)
        phase = float(self.phase.value or 0.0)
        return np.stack([np.ones_like(z), np.sin(2.0 * np.pi * z / wavelength + phase)])


class LocalizedShape(Shape):
    """Gaussian bump ``base + amplitude*exp(-((z - center)/width)**2 / 2)``.

    Levels are (base, amplitude). `center = 0` gives a layer at the free surface.
    """

    level_names = ("base", "amplitude")

    def __init__(
        self,
        center: float | Parameter,
        width: float | Parameter,
        *,
        name: str = "localized",
    ) -> None:
        super().__init__(name=name)
        self.center = _param(center, f"{name}_center")
        self.width = _param(width, f"{name}_width", bounds=(1e-6, None))

    def _shape_parameters(self) -> list[Parameter]:
        return [self.center, self.width]

    def _build_basis(
        self, z: NDArray[np.float64], thickness: float
    ) -> NDArray[np.float64]:
        del thickness
        center = float(self.center.value or 0.0)
        width = max(float(self.width.value or _MIN_LENGTH), _MIN_LENGTH)
        bump = np.exp(-0.5 * ((z - center) / width) ** 2)
        return np.stack([np.ones_like(z), bump])
