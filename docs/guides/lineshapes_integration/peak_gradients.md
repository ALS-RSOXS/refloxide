# Peak gradients

Depth-varying channels on a peak-sum optical source. A film is built from
peak-sum materials (`PeakSumSLD`) and mixed in a `DepthProfile`. Every
gradient is a `LevelField` on one shared shape, so every effect follows the
same depth profile.

For the pieces and their conventions in brief, see the
[Lineshapes integration overview](index.md).

## Three-material film

**Example:** `examples/lineshape_three_material.py`

The example film has three materials, all built from the frozen lineshapes
ZnPc and Alq3 fits. The fits are frozen in
`examples/data/lineshapes_materials.json`, which records the lineshapes commit
and the fit rms. The example reads that file relative to its own path. The
`examples/data/freeze_lineshapes_fits.py` script regenerates it from a
lineshapes checkout (set `LINESHAPES_REPO` if the checkout is not at the
default path).

- **`alq3`** is an isotropic material: 14 Gaussian peaks at their fitted
  unit-density heights, plus a continuum. The first five peaks by energy carry
  the depth gradients.
- **`znpc`** is the full uniaxial ZnPc tensor, with 19 peaks. Each peak's
  height and TDM tilt come from the ZnPc derivation below.
- **`hc_contaminant`** is a synthetic stand-in for a hydrocarbon contaminant,
  built from the ZnPc peak set. See below.

The density values are illustrative, not measured: Alq3 is 1.31 g/cm$^3$ and
ZnPc is 1.60 g/cm$^3$. Each `PeakSumSLD` takes its density as an argument,
since the frozen tables are per unit density.

The film is 200 Å thick with 40 microslabs and linear mixing. One
`SplineShape` with four knots spans it, at 0, 60, 120 and 200 Å (fractions
0, 0.3, 0.6 and 1 of the thickness $T$). The surface is at $z=0$, and $z=T$
is the film/oxide interface. Every gradient is a `LevelField` with four levels
on that shape. The surface and $z=T$ levels are fixed by the film. The two
interior levels are chosen so each profile curves, rather than following
the straight line between the endpoints. Three materials need the linear
mixing rule, since Maxwell-Garnett and Bruggeman support two components.

### Alq3: an isotropic model with depth-dependent dipoles

The Alq3 model has no tensor, so the outline is applied to its first five
peaks through heights and TDM tilts. The surface-rich outline is:

- Peaks p1, p2 and p3 have surface heights 1.5, 2.0 and 0.8 times their
  fitted (substrate) values. The surface level is the ratio times the fitted
  height. The interior levels follow the curved four-level pattern
  $h\,(1+(r-1)w)$ with weights $w=(1, 1/3, 1/12, 0)$, so the height relaxes
  to the fitted value at $z=T$.
- The TDM tilt of p1 and p2 goes from $0$ (along the molecular normal) at the
  surface to $20^\circ$ at $z=T$, through $12^\circ$ and $17^\circ$ at 60 and
  120 Å for p1 (and $14^\circ$ and $18^\circ$ for p2).
- The TDM tilt of p4 and p5 goes from $90^\circ$ (in the molecular plane) at
  the surface to $80^\circ$ at $z=T$.
- Every other Alq3 peak is a constant at its fitted height, with
  `tdm_tilt=MAGIC_ANGLE`, which is isotropic.

| Peak | Energy (eV) | Fitted height (z = T) | Surface height | TDM tilt, surface | TDM tilt, z = T |
|---|---|---|---|---|---|
| p1 | 284.59 | $3.068\times10^{-4}$ | $4.603\times10^{-4}$ (1.5x) | 0° | 20° |
| p2 | 284.80 | $8.434\times10^{-4}$ | $1.687\times10^{-3}$ (2.0x) | 0° | 20° |
| p3 | 285.54 | $1.407\times10^{-3}$ | $1.126\times10^{-3}$ (0.8x) | magic angle | magic angle |
| p4 | 286.20 | $5.733\times10^{-4}$ (constant) | same | 90° | 80° |
| p5 | 287.14 | $5.550\times10^{-4}$ (constant) | same | 90° | 80° |

The fitted heights are unit-density beta. Heights are absolute values, not
multipliers.

### ZnPc: the full uniaxial tensor

The ZnPc peaks come from the five-angle series, which gives each peak
film-frame in-plane and out-of-plane strengths, $F_{xx}$ and $F_{zz}$. The
peak height is the isotropic strength $F_{\rm iso}=(2F_{xx}+F_{zz})/3$. Each
peak's molecular TDM tilt is solved from the measured film anisotropy at the
measured molecular tilt $\gamma_{\rm ref}$, which is about 73.5° from the
strong $\pi^*$ peaks. ZnPc runs from amorphous (`MAGIC_ANGLE`) at the surface
to that ordered state at the substrate. Some peaks are clipped to the nearest
reachable tilt, at 0 or $\pi/2$.

### Hydrocarbon contaminant stand-in

`hc_contaminant` is a synthetic stand-in for a hydrocarbon contaminant layer.
It is not a measured hydrocarbon spectrum: it is built from the ZnPc peak set
(same tilts and baseline) with three seeded changes: heights swapped between
three pairs of strong in-window peaks, a random height factor, and small
random energy shifts. The changes come from one generator,
`np.random.default_rng(2025)`. The swaps are among the 8 strongest peaks inside
283.5 to 289.5 eV. Each height is multiplied by a lognormal factor (sigma 0.15,
clipped to 0.7 to 1.4), and each energy is shifted by a Gaussian (sigma 0.05 eV,
clipped to $\pm$0.10 eV). The swapped pairs, by original energy, are:

| Swap | Peak energies (eV) |
|---|---|
| 1 | 285.06 and 284.40 |
| 2 | 285.39 and 289.02 |
| 3 | 284.03 and 287.23 |

Its density, 1.60 g/cm$^3$ like ZnPc, is illustrative, as are the other
densities. In this demo the contaminant fraction rises toward the substrate
(5 to 25 %), which is a demo choice for showing a third depth profile, not a
claim about where contaminants sit.

### Construction code

The construction code below needs these imports:

```python
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import brentq

from refloxide.data import OpticalConstants
from refloxide.fields import LevelField, Shape, SplineShape
from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD, tdm_fractions
from refloxide.model import MaterialSLD, Mix, ReflectModel, Structure
from refloxide.profiles import DepthProfile
```

The included block reads the frozen data file from the example's own path
(`DATA_FILE`, from `__file__`). Run it as the example module, not pasted
into a bare session. The shared depth shape is one `SplineShape`. Any shape
family from the [overview](index.md) can be swapped in for it.

```python
--8<-- "examples/lineshape_three_material.py:construction"
```

The stack is vacuum | film | native SiO2 | Si, built as
`vacuum | build_profile() | oxide | substrate`. The oxide is a 15 Å SiO2 slab
on the silicon substrate. The oxide thickness and both roughness values are
illustrative.

Roughness follows the library convention: a slab's `rough` is the roughness
of the interface directly above it. So the film's own `roughness` (left at 0,
sharp) is the vacuum/film interface, the oxide's `rough` (3 Å) is the
film/oxide interface, and the substrate's `rough` (3 Å) is the oxide/Si
interface.

Each channel follows this profile:

- **Volume fractions.** Alq3 starts at 0.85 and falls fast over the top 60 Å
  to 0.55, then levels off near 0.40 at $z=T$. The two ZnPc layers enrich
  toward the substrate. ZnPc rises from 0.10 to about 0.36 near 90 Å, then
  settles at 0.35. Hydrocarbon contaminant stand-in rises from 0.05 to 0.25. The three fractions
  sum to 1 at every knot.
- **Total density.** `density_scale` multiplies the mixed tensor after mixing.
  It runs from 0.85 at the surface to 1.15 at $z=T$. Most of the rise is in
  the lower half.
- **Orientation.** Alq3 stays near face-on at the surface and reaches the
  magic angle at $z=T$. Both ZnPc layers start at the magic angle and order
  toward $\gamma_{\rm ref}$, 73.5°, with hydrocarbon contaminant ordering a little later
  in depth.
- **Alq3 peaks.** The heights of p1 to p3 and the TDM tilts of p1, p2, p4 and
  p5 follow the outline table above.

### Channels versus depth

Every channel versus depth, sampled with `DepthProfile.named_values_at`.
Angles are shown in degrees.

![Channels versus depth](../../assets/lineshape-gradient-channels.png)

### Building blocks over 283.5 to 289 eV

Unit-density $\beta(E)$ of alq3 (isotropic), of znpc (the $[0,0]$ and $[2,2]$
diagonal at $\gamma_{\rm ref}$, and their average), and of znpc against
hc_contaminant (isotropic average). The strip below shows the per-peak height change of hc_contaminant relative to znpc, for the peaks inside 283.5 to 289.5 eV only.

![Building blocks](../../assets/lineshape-gradient-materials.png)

### Depth-resolved NEXAFS

Local $\beta_o(E)$ (left) and $\beta_e(E)$ (right) at six depths, with the
colour set by depth.

The local tensors are taken at the microslab midpoint nearest each depth, on a
0.02 eV grid of 276 points. The requested depths 5, 40, 80, 120, 160 and 195 Å
fall exactly between two midpoints, so the lower slab is used. The depths
plotted are 2.5, 37.5, 77.5, 117.5, 157.5 and 192.5 Å.

![Depth-resolved NEXAFS](../../assets/lineshape-gradient-nexafs.png)

### Reflectivity ridge plot

Joyplot of $\log_{10}(R\,q^4)$ for $R_{ss}$ (left) and $R_{pp}$ (right), with
one ridge per energy at 56 energies from 283.5 to 289.0 eV in 0.1 eV steps.

Multiplying by $q^4$ removes the Fresnel decay. Each panel shares one window:
its top is the 99.9th percentile of its values, and its span is 2.8 decades.

![Reflectivity ridge plot](../../assets/lineshape-gradient-ridge.png)

## Single-material film

With one material there is no `Mix`, and the same channels still apply.
This snippet builds the Alq3 material on the same four-knot shape. Its
volume fraction is 1, so `density_scale` and `gamma` act on that one material
alone:

```python
shape = make_shape()
materials = load_materials()
alq3 = PeakSumSLD(
    alq3_peaks(shape, materials["alq3"]),
    baseline=baseline_table(materials["alq3"]),
    density=DENSITY_ALQ3,
    name="alq3",
)
layer = DepthProfile(
    alq3,
    thickness=THICKNESS,
    n_slabs=N_SLABS,
    gamma=LevelField(shape, [0.0, 0.10, 0.50, MAGIC_ANGLE], name="gamma"),
    density_scale=LevelField(shape, [0.85, 0.87, 0.97, 1.15], name="density_scale"),
)
vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
oxide = MaterialSLD("SiO2", density=2.2, name="native_oxide")(15.0, 3.0)
substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3.0)
structure = vacuum | layer | oxide | substrate
```

## Other field families

Any depth field can bind to a peak parameter. A `TransitionShape` gives a
relaxation from the surface and bulk to the substrate. A `LocalizedShape`
adds a Gaussian surface layer. `Derived` adds the two pointwise. Here the
orientation and density stay constant. These snippets are standalone: they
use their own peak and do not depend on the example film.

```python
import numpy as np

from refloxide.fields import Derived, LevelField, LocalizedShape, TransitionShape
from refloxide.lineshapes import Peak, PeakSumSLD
from refloxide.profiles import DepthProfile

transition = TransitionShape(tau_top=25.0, tau_bottom=40.0, name="transition")
surface = LocalizedShape(center=0.0, width=20.0, name="surface")
bulk_height = LevelField(transition, [0.014, 0.012, 0.010], name="bulk_height")
surface_bump = LevelField(surface, [0.0, 0.006], name="surface_bump")
height = Derived(
    lambda bulk, bump: bulk + bump,
    bulk=bulk_height,
    bump=surface_bump,
    name="p1_height",
)
film = PeakSumSLD(
    [Peak(label="p1", energy=285.9, width=0.5, height=height)], name="film"
)
layer = DepthProfile(
    film,
    thickness=200.0,
    n_slabs=40,
    gamma=np.deg2rad(30.0),
    density=1.0,
)
```

A plain Python function can be a channel too. `CallableField` calls
`fn(z, thickness=..., **params)` at each microslab. Here it is a ripple on the
total density of the film from the previous snippet (the one with the
`Derived` peak height).

```python
from refloxide.profiles import CallableField


def ripple(z, *, thickness, amp, period):
    """Density ripple of relative amplitude `amp` and period `period` (Angstrom)."""
    return 1.0 + amp * np.sin(2.0 * np.pi * z / period)


density_scale = CallableField(
    ripple, params={"amp": 0.05, "period": 80.0}, name="ripple"
)
layer = DepthProfile(film, thickness=200.0, n_slabs=40, density_scale=density_scale)
```

A `SinusoidShape` gives a periodic modulation. Its levels are the offset and
the amplitude, and its `wavelength` is in Å. Here the peak height ripples
about a mean of 0.012 with amplitude 0.002 and a period of 80 Å.

```python
from refloxide.fields import LevelField, SinusoidShape
from refloxide.lineshapes import Peak, PeakSumSLD
from refloxide.profiles import DepthProfile

modulation = SinusoidShape(wavelength=80.0, name="ripple")
ripple_height = LevelField(modulation, [0.012, 0.002], name="ripple_height")
film = PeakSumSLD(
    [Peak(label="p1", energy=285.9, width=0.5, height=ripple_height)], name="film"
)
layer = DepthProfile(film, thickness=200.0, n_slabs=40, gamma=0.0, density=1.0)
```

## Fitting knot positions

The knot positions of a `SplineShape` are fit parameters too. Each knot is a
`Parameter` named `depth_knot0`, `depth_knot1`, and so on for the shape named
`depth`. Knots are fixed at `vary=False` until you open them. The number of
knots is fixed when the shape is built, so every level field on the shape
keeps its level count.

The snippet builds its own five-knot shape at 0, 50, 100, 150 and 200 Å. It
is separate from the four-knot shape of the example film. It opens the two
interior knots, which start at 50 Å and 100 Å. Knot 1 is bounded to 20 to
75 Å and knot 2 to 80 to 140 Å. Those ranges do not overlap, so the knots
stay in order during a fit: knot 1 stays below knot 2, and knot 2 stays below
the fixed knot 3 at 150 Å. Knot 1 stays above the fixed knot 0 at 0 Å. Both
starts lie inside their bounds.

```python
import numpy as np

from refloxide.fields import LevelField, SplineShape
from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD
from refloxide.profiles import DepthProfile

shape = SplineShape(np.linspace(0.0, 200.0, 5), name="depth")
shape.knot_parameters[1].setp(vary=True, bounds=(20.0, 75.0))
shape.knot_parameters[2].setp(vary=True, bounds=(80.0, 140.0))

film = PeakSumSLD(
    [
        Peak(
            label="p1",
            energy=285.9,
            width=0.5,
            height=LevelField(
                shape, [0.018, 0.016, 0.014, 0.012, 0.010], name="p1_height"
            ),
        )
    ],
    name="film",
)
layer = DepthProfile(
    film,
    thickness=200.0,
    n_slabs=40,
    gamma=LevelField(shape, [0.0, 0.3, 0.6, 0.8, MAGIC_ANGLE], name="gamma"),
    density_scale=LevelField(shape, [0.9, 0.95, 1.0, 1.05, 1.1], name="density_scale"),
)

knots = [
    p.name
    for p in layer.parameters.flattened(unique=True)
    if "depth_knot" in p.name
]
print(knots)
print([p.name for p in layer.parameters.varying_parameters() if "depth_knot" in p.name])
```

The shape is shared by `gamma`, `density_scale`, and the peak height, so in
the plain `layer.parameters.flattened()` listing the knots of a shared shape
repeat. Pass `unique=True`, as above, to list each knot once. The
`varying_parameters()` list is already unique and shows the two opened knots,
`depth_knot1` and `depth_knot2`. Running the snippet prints:

```text
['depth_knot0', 'depth_knot1', 'depth_knot2', 'depth_knot3', 'depth_knot4']
['depth_knot1', 'depth_knot2']
```

To tie one knot to another instead of bounding it, use a refnx constraint.
This keeps knot 2 exactly 40 Å above knot 1, and knot 2 is then not a free
parameter:

```python
knots = shape.knot_parameters
knots[2].setp(vary=False, constraint=knots[1] + 40.0)
```

## Conventions

- **Radians in code.** Every angle in the API is in radians. The example
  converts from degrees with `np.deg2rad`.
- **Orientation `gamma`.** The tilt of the molecular normal from the surface
  normal. $\gamma=0$ is face-on, $\gamma=\pi/2$ is edge-on, and the magic
  angle $\arccos(1/\sqrt{3})\approx54.7^\circ$ is amorphous (`MAGIC_ANGLE`).
- **TDM tilt.** The polar angle $\theta$ of a peak's transition dipole, measured
  from the molecular normal. $\theta=0$ lies along the normal and
  $\theta=\pi/2$ lies in the molecular plane. The default is `MAGIC_ANGLE`,
  which gives equal in-plane and out-of-plane strength.
- **Heights.** A peak's `height` is the height itself. There is no multiplier
  mechanism, so the example writes surface values as absolute heights.
- **Scalars are constant.** A peak parameter that is a plain number is
  evaluated once per energy. Only parameters bound to a depth field are
  evaluated per microslab.
