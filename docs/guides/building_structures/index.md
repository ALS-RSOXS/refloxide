# Building structures

Guide to assembling stratified stacks in refloxide: homogeneous slabs, mixed
materials, callable depth laws, named functional forms, phase transitions,
diffusion profiles, and multi-layer stacking.

This section replaces the former single-page “Built-in slabs” guide.

Figure builders live in
[`docs/guides/builtin_slabs/build_figures.py`](../builtin_slabs/build_figures.py).

Regenerate figures:

```bash
uv run python docs/guides/builtin_slabs/build_figures.py
```

Depth panels: **left = sharp**, **right = erf-broadened interfacial roughness**
(display convention; kernel Nevot–Croce is unchanged). Reflectivity panels use
`refloxide.tmm.uniaxial_reflectivity` at 284.4 eV over
$q \in [0.008,\,0.275]\,\mathrm{\AA}^{-1}$ (near the full accessible window at
this energy).

Shared preamble used on every child page:

--8<-- "guides/building_structures/_preamble.md"

## Contents

| Page | What it covers |
|------|----------------|
| [Homogeneous slabs](homogeneous.md) | Free vs material tensors; uni (4) vs bi (6) diagonals |
| [Mixed slabs](mixed.md) | Binary MixRule comparison and linear $N=1,2,3$ mixes |
| [Functional forms](functional_forms.md) | Named/callable fields on $\phi$, orientation, or free $\delta$/$\beta$ |
| &nbsp;&nbsp;[Splines](splines.md) | Knotted $\phi(z)$ and free-optical Splines |
| &nbsp;&nbsp;[Polynomials](polynomials.md) | Polynomial $\phi(z)$ and free $\delta_o$/$\delta_e$ |
| &nbsp;&nbsp;[Callable profiles](callables.md) | User callables on any channel or full tensors |
| [Phase transitions](phase_transitions.md) | `SecondOrderTransition` on $\gamma$, $\phi$, $\langle\cos^2\gamma\rangle$ |
| [Diffusion](diffusion.md) | Couple vs exponential diffusion kinds |
| [Stacking examples](stacking.md) | Multi-layer stacks, incl. free-optical Poly/Spline/$\cos$ stack |
