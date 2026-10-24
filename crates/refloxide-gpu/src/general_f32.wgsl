// Single-precision prelude for general.wgsl: hardware transcendentals.
// Diagnostic build; the general 4x4 is not accurate in f32 for tilted stacks.

alias real = f32;

const EPS: real = 1.1920929e-7;
const SQRT_EPS: real = 3.4526698e-4;

fn rexp(x: real) -> real {
    return exp(x);
}

// (cos x, sin x).
fn rcossin(x: real) -> vec2<real> {
    return vec2<real>(cos(x), sin(x));
}
