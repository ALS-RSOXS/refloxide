# Roughness model selection guide

## Scope

This page will provide a practical decision framework for choosing among
roughness treatments in `refloxide`.

## Decision factors

- Roughness magnitude relative to wavelength and layer scales
- Expected correlation-length regime
- Whether phase-sensitive internal field quantities are required
- Accuracy versus runtime constraints

## Recommended flow

1. Start with a fast multiplicative model for baseline sweeps.
2. Escalate to graded-interface discretization when roughness is large or
   residuals indicate model mismatch.
3. Re-check model assumptions whenever fitting moves into a new parameter
   regime.

## Output

The guide should map each regime to a default model and include explicit
conditions that trigger model escalation.
