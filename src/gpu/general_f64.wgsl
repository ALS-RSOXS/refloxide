// Double-precision prelude for general.wgsl (requires wgpu SHADER_F64).
//
// Vulkan and DX12 provide f64 arithmetic, sqrt and fma but no f64
// exp/sin/cos, so they are evaluated here with Cody-Waite range reduction
// (constants split so n * part is exact) and Taylor polynomials on the
// reduced interval; relative error is a few ulp.

alias real = f64;

const EPS: real = 2.220446049250313e-16lf;
const SQRT_EPS: real = 1.4901161193847656e-8lf;

const LOG2E: real = 1.4426950408889634lf;
const LN2_HI: real = 0.6931471805601177lf;
const LN2_LO: real = -1.7239444525614835e-13lf;
const TWO_OVER_PI: real = 0.6366197723675814lf;
const PIO2_1: real = 1.5707963267341256lf;
const PIO2_2: real = 6.077100506303966e-11lf;
const PIO2_3: real = 2.0222662487959506e-21lf;

// 1/n!, n = 0..14.
const EXP_C = array<real, 15>(
    1.0lf,
    1.0lf,
    0.5lf,
    0.16666666666666666lf,
    0.041666666666666664lf,
    0.008333333333333333lf,
    0.001388888888888889lf,
    0.0001984126984126984lf,
    2.48015873015873e-05lf,
    2.7557319223985893e-06lf,
    2.755731922398589e-07lf,
    2.505210838544172e-08lf,
    2.08767569878681e-09lf,
    1.6059043836821613e-10lf,
    1.1470745597729725e-11lf
);

// (-1)^k / (2k+1)!, k = 0..10.
const SIN_C = array<real, 11>(
    1.0lf,
    -0.16666666666666666lf,
    0.008333333333333333lf,
    -0.0001984126984126984lf,
    2.7557319223985893e-06lf,
    -2.505210838544172e-08lf,
    1.6059043836821613e-10lf,
    -7.647163731819816e-13lf,
    2.8114572543455206e-15lf,
    -8.22063524662433e-18lf,
    1.9572941063391263e-20lf
);

// (-1)^k / (2k)!, k = 0..10.
const COS_C = array<real, 11>(
    1.0lf,
    -0.5lf,
    0.041666666666666664lf,
    -0.001388888888888889lf,
    2.48015873015873e-05lf,
    -2.755731922398589e-07lf,
    2.08767569878681e-09lf,
    -1.1470745597729725e-11lf,
    4.779477332387385e-14lf,
    -1.5619206968586225e-16lf,
    4.110317623312165e-19lf
);

// Exact 2^k for |k| < 1024 by binary powering of exact powers of two.
fn pow2i(k: i32) -> real {
    var r: real = 1.0lf;
    var b: real = select(2.0lf, 0.5lf, k < 0);
    var e = u32(abs(k));
    for (var i = 0; i < 10; i++) {
        if ((e & 1u) == 1u) {
            r = r * b;
        }
        b = b * b;
        e = e >> 1u;
    }
    return r;
}

fn rexp(x: real) -> real {
    if (x < -700.0lf) {
        return 0.0lf;
    }
    let xc = min(x, 700.0lf);
    let k = floor(fma(xc, LOG2E, 0.5lf));
    var r = fma(-k, LN2_HI, xc);
    r = fma(-k, LN2_LO, r);
    var p = EXP_C[14];
    for (var i = 13; i >= 0; i--) {
        p = fma(p, r, EXP_C[i]);
    }
    let ki = i32(k);
    let k1 = ki / 2;
    return p * pow2i(k1) * pow2i(ki - k1);
}

// (cos x, sin x).
fn rcossin(x: real) -> vec2<real> {
    let xc = clamp(x, -1.0e9lf, 1.0e9lf);
    let n = floor(fma(xc, TWO_OVER_PI, 0.5lf));
    var r = fma(-n, PIO2_1, xc);
    r = fma(-n, PIO2_2, r);
    r = fma(-n, PIO2_3, r);
    let r2 = r * r;
    var s = SIN_C[10];
    var c = COS_C[10];
    for (var i = 9; i >= 0; i--) {
        s = fma(s, r2, SIN_C[i]);
        c = fma(c, r2, COS_C[i]);
    }
    s = s * r;
    let quadrant = u32(i32(n) & 3);
    if (quadrant == 0u) {
        return vec2<real>(c, s);
    }
    if (quadrant == 1u) {
        return vec2<real>(-s, c);
    }
    if (quadrant == 2u) {
        return vec2<real>(-c, -s);
    }
    return vec2<real>(s, -c);
}
