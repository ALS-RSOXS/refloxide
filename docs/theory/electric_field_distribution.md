# Electric field distribution

## Scope

This page will document stage 6 of the 4x4 workflow: reconstruction of
the complex electric field inside each layer after the reflection and
transmission coefficients have been determined.

## Inputs from the core pipeline

- Layer eigenvectors and modal basis matrices from `eigenmode_analysis.md`
- Interface-normalized matrices from `interface_matrices.md`
- Propagation factors and stack transfer products from
  `propagation_and_assembly.md`
- Amplitude coefficients from `reflection_transmission.md`

## Output contract

The field module should return deterministic arrays for
$\vec{E}(x, y, z)$ or reduced slices (for example $E_x(z)$ at fixed
in-plane coordinates), along with a documented mode-normalization
convention.

## Validation targets

- Continuity of tangential field components at interfaces
- Agreement with published erratum-corrected expressions
- Convergence against finer $z$ sampling in graded-interface expansions
