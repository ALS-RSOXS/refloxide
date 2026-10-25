# Lineshapes integration

A peak-sum optical source. `PeakSumSLD` is a material whose molecular indices
are a sum of labeled lineshape peaks (`Peak`), optionally added to a tabulated
baseline. Each peak parameter can be a plain number or a depth field, so the
same source can be used in a uniform slab or in a depth-resolved film. The
peak sums are evaluated by the Rust `peak_sum_molecular` kernel, which uses the
`lineshapes` crate.

## The pieces

| Piece | Module | Role |
|---|---|---|
| `Peak` | `refloxide.lineshapes` | One labeled peak: `kind` (`gaussian`, `lorentzian`, `voigt`, `pseudo_voigt`), energy, width, `height`, `tdm_tilt` |
| `PeakSumSLD` | `refloxide.lineshapes` | Material from a sum of peaks, plus an optional `baseline` table; `density` and `rotation` (orientation) |
| `MAGIC_ANGLE` | `refloxide.lineshapes` | $\arccos(1/\sqrt{3})$, about 54.7°, the amorphous orientation |
| `tdm_fractions` | `refloxide.lineshapes` | Splits a TDM tilt into in-plane and out-of-plane strengths |
| `TransitionShape`, `SplineShape`, `PolynomialShape`, `SinusoidShape`, `LocalizedShape` | `refloxide.fields` | Depth shapes; a shape gives the form, its levels give the values |
| `LevelField` | `refloxide.fields` | A shape with its own level values. One shape can back several fields |
| `Derived`, `cos2_of` | `refloxide.fields` | Pointwise combinations of other fields; `cos2_of` gives $\langle\cos^2\gamma\rangle$ |
| `DepthProfile` | `refloxide.profiles` | Microslab film built from a material. Channels include `phi`, `gamma`, `density` and `density_scale` |
| `Mix` | `refloxide.model` | Linear mixing of components by volume fraction |

## How they compose

- **One shape, many channels.** A shared shape drives several channels. Each
  channel is a `LevelField` on that shape with its own levels, so the volume
  fractions, orientation, density and peak heights follow one depth profile.
  Shape parameters are fit parameters once, however many channels use them.
- **Knots are fittable.** The knot positions of a `SplineShape` are `Parameter`s,
  fixed by default. Open them with `vary=True` and bounds to fit them.
- **Scalars are constant.** A peak parameter that is a plain number, frozen or
  fit, is evaluated once per energy. Only parameters bound to a depth field are
  evaluated per microslab.
- **Outside a profile.** A depth-bound parameter evaluated without a
  `DepthProfile` uses the field's bulk value.

## A minimal structure

A uniform 100 Å film of two peaks on a native oxide and a silicon substrate,
evaluated with `ReflectModel`:

```python
import numpy as np

from refloxide.lineshapes import MAGIC_ANGLE, Peak, PeakSumSLD
from refloxide.model import MaterialSLD, ReflectModel

peaks = [
    Peak(label="p1", energy=284.8, width=0.5, height=0.012, tdm_tilt=MAGIC_ANGLE),
    Peak(label="p2", energy=286.0, width=0.5, height=0.008, tdm_tilt=np.deg2rad(30.0)),
]
film = PeakSumSLD(peaks, name="film")
vacuum = MaterialSLD("", 0, name="vacuum")(0, 0)
oxide = MaterialSLD("SiO2", density=2.2, name="native_oxide")(15.0, 3.0)
substrate = MaterialSLD("Si", density=2.33, name="substrate")(0, 3.0)
structure = vacuum | film(100.0, 2.0) | oxide | substrate

q = np.linspace(0.01, 0.27, 200)  # 1/Angstrom
energies = np.array([284.8, 286.0])  # eV
refl = ReflectModel(structure)(q, energies)
```

The oxide thickness and roughness values are illustrative. For a depth-resolved
film, see the worked example.

## Conventions

- Angles are in radians everywhere in code. Use `np.deg2rad` for degrees.
- Orientation $\gamma$ is the tilt of the molecular normal from the surface
  normal: $0$ is face-on, $\pi/2$ is edge-on, and `MAGIC_ANGLE` is amorphous.
- A peak's `tdm_tilt` is measured from the molecular normal.
- A peak's `height` is the height itself, with no multiplier.

The [worked example](peak_gradients.md#conventions) lists the same conventions
with more detail.

## Limits

- Maxwell-Garnett and Bruggeman mixing support only two components. For three
  or more, use `rule="linear"`.
- A peak-only material has no continuum. Add a tabulated `baseline` for a
  background.
- There is no Rust `OpticalSource` implementation yet. The Python protocol in
  `refloxide.sources` is the boundary that external sum-of-states code is meant
  to plug into.

## Next

- [Peak gradients](peak_gradients.md): a three-material film (frozen Alq3 and
  ZnPc fits, a full ZnPc tensor and a shifted ZnPc) with depth-varying
  composition, orientation and peaks, a dense energy set with NEXAFS and ridge
  figures, and the knot fit.
- [Building structures](../building_structures/index.md): the other depth-law
  families, such as splines, polynomials, diffusion and phase transitions.
