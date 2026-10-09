# Functional forms

Depth laws attached to `DepthProfile`. Any `_FieldBase` form (`Spline`,
`Polynomial`, `Diffusion`, `SecondOrderTransition`, `CallableField`, …) can
drive **any** supported channel — not only Mix volume fraction $\phi$ or
orientation $\gamma$ / $\langle\cos^2\gamma\rangle$.

## Channels

| Channel kwargs | Typical use |
|----------------|-------------|
| `phi` | Volume fraction of `materials[0]` in a `Mix` |
| `gamma`, `cos2_gamma` | Orientation / order parameter on a uniaxial material |
| `density` | Density scale on the material |
| `delta_o`, `delta_e`, `beta_o`, `beta_e` | Free laboratory optical diagonals (no Mix required) |

Attach the same form class to different kwargs:

```python
# Mix composition
DepthProfile(material=Mix([mat_a, mat_b], rule="linear"), thickness=200.0, phi=field)

# Free optical diagonals (δ + iβ packing)
DepthProfile(
    material=mat,
    thickness=240.0,
    delta_o=field_o,
    delta_e=field_e,
    beta_o=4e-4,
    beta_e=6e-4,
)
```

Depth panels annotate the controlling constants. Figure builders live in
[`build_figures.py`](../builtin_slabs/build_figures.py); conventions are on
the [Building structures overview](index.md).

For interfacial orientation / composition transitions see
[Phase transitions](phase_transitions.md). For Fick couple / exponential
profiles see [Diffusion](diffusion.md).

## Pages

| Page | Role |
|------|------|
| [Splines](splines.md) | Knotted $\phi(z)$ (order / knot count) and free $\delta_o$/$\delta_e$ Splines |
| [Polynomials](polynomials.md) | Polynomial $\phi(z)$ and free $\delta_o$/$\delta_e$ channels |
| [Callable profiles](callables.md) | User `CallableField` / `CallableDepthProfile` escape hatches |

Shared preamble:

--8<-- "guides/building_structures/_preamble.md"

Typical Mix composition pattern:

```python
phi = Spline(
    knots=[0.0, 30.0, 70.0, 110.0, 150.0, 200.0],
    values=[0.20, 0.95, 0.25, 0.80, 0.15, 0.55],
)
layer = DepthProfile(
    material=Mix([mat_a, mat_b], rule="linear"),
    thickness=200.0,
    phi=phi,
)
stack = vac | layer | si
```
