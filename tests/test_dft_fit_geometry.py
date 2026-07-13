"""Regression tests for DFT fit geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

from refnx.analysis import Parameter

from refloxide.pxr.plugin.dft_fit import apply_shared_slab_geometry_from_reference
from refloxide.pxr.plugin.structure import Scatterer


class _DummyScatterer(Scatterer):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)


@dataclass
class _SlabGeom:
    thick: Parameter
    rough: Parameter


@dataclass
class _Stack:
    components: list[_SlabGeom | None]

    def __getitem__(self, index: int) -> _SlabGeom:
        slab = self.components[index]
        if slab is None:
            msg = f"missing slab at index {index}"
            raise IndexError(msg)
        return slab


def _slab_geom(
    thick: float,
    thick_bounds: tuple[float, float],
    rough: float,
    rough_bounds: tuple[float, float],
) -> _SlabGeom:
    return _SlabGeom(
        thick=Parameter(thick, bounds=thick_bounds),
        rough=Parameter(rough, bounds=rough_bounds),
    )


def test_apply_shared_slab_geometry_copies_bounds_not_only_values() -> None:
    ref = _Stack(
        [
            None,
            _slab_geom(180.0, (150.0, 210.0), 12.0, (2.0, 16.0)),
            _slab_geom(8.8, (5.0, 12.0), 5.0, (0.0, 8.0)),
        ]
    )
    built = _Stack(
        [
            None,
            _slab_geom(2.0, (0.0, 4.0), 6.0, (0.0, 12.0)),
            _slab_geom(1.5, (0.0, 3.0), 10.0, (0.0, 20.0)),
        ]
    )
    apply_shared_slab_geometry_from_reference(
        built,
        ref,
        film_indices=(1,),
        oxide_index=2,
        substrate_index=2,
        fix_substrate=False,
    )

    znpc = built[1]
    assert znpc.thick.value == 180.0
    assert znpc.thick.bounds is not None
    assert float(znpc.thick.bounds.lb) == 150.0
    assert float(znpc.thick.bounds.ub) == 210.0
    assert znpc.rough.bounds is not None
    assert float(znpc.rough.bounds.lb) == 2.0
    assert float(znpc.rough.bounds.ub) == 16.0

    oxide = built[2]
    assert oxide.thick.value == 8.8
    assert oxide.thick.bounds is not None
    assert float(oxide.thick.bounds.lb) == 5.0
    assert float(oxide.thick.bounds.ub) == 12.0


def test_scatterer_call_sets_double_thickness_bounds() -> None:
    slab = _DummyScatterer("film")(2.97, 6.78)
    assert slab.thick.bounds is not None
    assert float(slab.thick.bounds.ub) == 5.94
    assert float(slab.rough.bounds.ub) == 13.56
