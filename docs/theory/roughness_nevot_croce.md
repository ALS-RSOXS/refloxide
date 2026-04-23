# Roughness model: Nevot-Croce

## Scope

This page will document the Nevot-Croce correction as implemented in
`refloxide` for interfaces where the short-correlation approximation is
appropriate.

## Use regime

- Small to moderate roughness compared with wavelength-scale phase
  variation
- Interface-dominated correction where diffuse off-specular terms are out
  of scope

## Implementation notes

- Apply correction at interface terms in a way that preserves the chosen
  polarization/mode basis conventions
- Keep numerical behavior stable near grazing and critical-angle regions

## Validation

- Recover sharp-interface limit as $\sigma \to 0$
- Compare against scalar isotropic references in the isotropic limit
