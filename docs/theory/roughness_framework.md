# Roughness framework

## Scope

This page defines where roughness enters the `refloxide` computation
graph and how each roughness model composes with the core 4x4 pipeline.

## Integration points

- Multiplicative interface corrections applied to reflection-style
  interface terms
- Structure pre-processing via graded-interface discretization before the
  eigensolve and transfer-product stages

## Shared interface

Each roughness model should provide:

- Model parameters and validity assumptions
- A deterministic transformation from ideal interface quantities to
  roughness-adjusted quantities
- Clear numerical limits and fallback behavior near model breakdown

## Model pages

- `roughness_nevot_croce.md`
- `roughness_debye_waller.md`
- `roughness_graded_interface.md`
- `roughness_selection_guide.md`
