# Roughness model: Debye-Waller

## Scope

This page will document the Debye-Waller-style roughness correction and
its practical overlap and differences relative to Nevot-Croce within the
`refloxide` pipeline.

## Use regime

- Long-correlation roughness approximations in reflectivity workflows
- Cases where multiplicative attenuation factors are appropriate for the
  modeled observable

## Implementation notes

- Preserve consistent amplitude and phase conventions with the transfer
  matrix formulation
- Document when the model is numerically or physically unreliable for
  large roughness and strong anisotropy

## Validation

- Recover sharp-interface behavior as $\sigma \to 0$
- Match expected trends versus incidence coordinate sweeps
