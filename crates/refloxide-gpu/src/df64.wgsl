// Double-single ("df64") arithmetic: a value is the unevaluated sum
// hi + lo of two f32 with |lo| <= ulp(hi) / 2, giving ~48 mantissa bits and
// f32 exponent range. Built on error-free transformations (Knuth two-sum,
// fma-based two-product); every operation below must be compiled without
// floating-point reassociation, which the self-test verifies on each device.
//
// Real df64 is vec2<f32>(hi, lo); complex df64 is vec4<f32>(re.hi, re.lo,
// im.hi, im.lo).

// Runtime zero bit mask loaded from a uniform by every df64 kernel before
// any arithmetic. XOR-ing an intermediate's bits with it changes no value
// but routes it through integer operations, which hides the algebraic
// identities ((a + b) - a == b, fma(a, b, -a*b) == 0) that a fast-math
// compiler (Metal compiles WGSL with fast math enabled) would otherwise
// fold, collapsing df64 back to f32. A float multiply by a runtime 1.0 is
// not enough on Metal.
var<private> df_mask: u32 = 0u;

fn df_opaque(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ df_mask);
}

// Every addend whose grouping matters is an opaque atom: fast math may
// otherwise regroup e.g. (a - (s - v)) + (b - v) as (a + b) - (s - v) - v,
// where a + b rounds back to s and the error term vanishes.
fn df_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = df_opaque(a + b);
    let v = df_opaque(s - a);
    let e = df_opaque(a - df_opaque(s - v)) + df_opaque(b - v);
    return vec2<f32>(s, e);
}

fn df_quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = df_opaque(a + b);
    let e = df_opaque(b - df_opaque(s - a));
    return vec2<f32>(s, e);
}

fn df_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = df_opaque(a * b);
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

fn df(x: f32) -> vec2<f32> {
    return vec2<f32>(x, 0.0);
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df_two_sum(a.x, b.x);
    let t = df_two_sum(a.y, b.y);
    var r = df_quick_two_sum(s.x, df_opaque(s.y + t.x));
    r = df_quick_two_sum(r.x, df_opaque(r.y + t.y));
    return r;
}

fn df_neg(a: vec2<f32>) -> vec2<f32> {
    return -a;
}

fn df_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return df_add(a, -b);
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = df_two_prod(a.x, b.x);
    let cross = df_opaque(df_opaque(a.x * b.y) + df_opaque(a.y * b.x));
    return df_quick_two_sum(p.x, df_opaque(p.y + cross));
}

fn df_mul_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    let p = df_two_prod(a.x, b);
    return df_quick_two_sum(p.x, df_opaque(p.y + df_opaque(a.y * b)));
}

fn df_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = a.x / b.x;
    var r = df_sub(a, df_mul_f(b, q1));
    let q2 = r.x / b.x;
    r = df_sub(r, df_mul_f(b, q2));
    let q3 = r.x / b.x;
    return df_add(df_quick_two_sum(q1, q2), df(q3));
}

fn df_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (a.x <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let q = sqrt(a.x);
    let r = df_sub(a, df_two_prod(q, q));
    return df_quick_two_sum(q, r.x / (2.0 * q));
}

fn df_abs(a: vec2<f32>) -> vec2<f32> {
    return select(a, -a, a.x < 0.0);
}

// ln(2), pi/2, and 1/n! as exactly rounded (hi, lo) f32 pairs (mpmath).
const DF_LN2 = vec2<f32>(0.6931471824645996, -1.9046542121259336e-09);
const DF_HALF_PI = vec2<f32>(1.5707963705062866, -4.371138828673793e-08);
const DF_INV_FACT = array<vec2<f32>, 18>(
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.5, 0.0),
    vec2<f32>(0.1666666716337204, -4.967053879312289e-09),
    vec2<f32>(0.0416666679084301, -1.2417634698280722e-09),
    vec2<f32>(0.008333333767950535, -4.34617203337595e-10),
    vec2<f32>(0.0013888889225199819, -3.3631094437103215e-11),
    vec2<f32>(0.00019841270113829523, -2.725596874933456e-12),
    vec2<f32>(2.4801587642286904e-05, -3.40699609366682e-13),
    vec2<f32>(2.7557318844628753e-06, 3.793571224297229e-14),
    vec2<f32>(2.755731998149713e-07, -7.575112209051195e-15),
    vec2<f32>(2.5052107943679403e-08, 4.4176230446483665e-16),
    vec2<f32>(2.0876755879584152e-09, 1.1082839147459852e-16),
    vec2<f32>(1.6059044372074283e-10, -5.352526511562726e-18),
    vec2<f32>(1.147074536050896e-11, 2.372207689231238e-19),
    vec2<f32>(7.647163609812713e-13, 1.2200710471178288e-20),
    vec2<f32>(4.7794772561329454e-14, 7.62544404448643e-22),
    vec2<f32>(2.8114573589663704e-15, -1.0462084739763658e-22),
);

// exp(a): a = k ln2 + r with |r| <= ln2 / 2, exp(r / 256) by Horner on
// 1/n! to n = 9, then 8 squarings and an exact 2^k scale. Results below
// ~1e-26 progressively lose the low word to f32 denormal flushing (df64 has
// the f32 exponent range), degrading toward f32 relative precision.
// Scaling uses ldexp, an exact exponent shift; exp2 is approximate under
// fast math.
fn df_exp(a: vec2<f32>) -> vec2<f32> {
    if (a.x < -87.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let k = round(a.x / DF_LN2.x);
    let r = df_mul_f(df_sub(a, df_mul_f(DF_LN2, k)), 1.0 / 256.0);
    let c = DF_INV_FACT;
    var sum = c[9];
    for (var n = 8; n >= 0; n = n - 1) {
        sum = df_add(df_mul(sum, r), c[n]);
    }
    for (var s = 0; s < 8; s = s + 1) {
        sum = df_mul(sum, sum);
    }
    return vec2<f32>(ldexp(sum.x, i32(k)), ldexp(sum.y, i32(k)));
}

// ln(a) for a > 0: f32 seed refined by one Newton step on exp.
fn df_log(a: vec2<f32>) -> vec2<f32> {
    let y = df(log(a.x));
    return df_add(y, df_sub(df_mul(a, df_exp(df_neg(y))), df(1.0)));
}

// sin and cos of a reduced |r| <= pi/4 by Horner in r^2 on 1/n! (to r^17).
fn df_sincos_reduced(r: vec2<f32>) -> vec4<f32> {
    let r2 = df_mul(r, r);
    let c = DF_INV_FACT;
    var s = c[17];
    var co = c[16];
    let neg_r2 = df_neg(r2);
    for (var n = 7; n >= 0; n = n - 1) {
        s = df_add(df_mul(s, neg_r2), c[2 * n + 1]);
        co = df_add(df_mul(co, neg_r2), c[2 * n]);
    }
    return vec4<f32>(df_mul(s, r), co);
}

// (sin a, cos a) with reduction by multiples of pi/2 in df64.
fn df_sincos(a: vec2<f32>) -> vec4<f32> {
    let k = round(a.x / DF_HALF_PI.x);
    let r = df_sub(df_sub(a, df_mul_f(df(DF_HALF_PI.x), k)), df_mul_f(df(DF_HALF_PI.y), k));
    let sc = df_sincos_reduced(r);
    let m = i32(k) & 3;
    let s = sc.xy;
    let c = sc.zw;
    if (m == 0) { return vec4<f32>(s, c); }
    if (m == 1) { return vec4<f32>(c, -s); }
    if (m == 2) { return vec4<f32>(-s, -c); }
    return vec4<f32>(-c, s);
}

// ---- complex df64: vec4(re.hi, re.lo, im.hi, im.lo) ----

fn cd(re: vec2<f32>, im: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(re, im);
}

fn cd_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return cd(df_add(a.xy, b.xy), df_add(a.zw, b.zw));
}

fn cd_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return cd(df_sub(a.xy, b.xy), df_sub(a.zw, b.zw));
}

fn cd_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return cd(
        df_sub(df_mul(a.xy, b.xy), df_mul(a.zw, b.zw)),
        df_add(df_mul(a.xy, b.zw), df_mul(a.zw, b.xy)),
    );
}

fn cd_scale(a: vec4<f32>, k: vec2<f32>) -> vec4<f32> {
    return cd(df_mul(a.xy, k), df_mul(a.zw, k));
}

fn cd_norm_sqr(a: vec4<f32>) -> vec2<f32> {
    return df_add(df_mul(a.xy, a.xy), df_mul(a.zw, a.zw));
}

fn cd_div(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let n = cd_norm_sqr(b);
    let re = df_add(df_mul(a.xy, b.xy), df_mul(a.zw, b.zw));
    let im = df_sub(df_mul(a.zw, b.xy), df_mul(a.xy, b.zw));
    return cd(df_div(re, n), df_div(im, n));
}

fn cd_exp(a: vec4<f32>) -> vec4<f32> {
    let e = df_exp(a.xy);
    let sc = df_sincos(a.zw);
    return cd(df_mul(e, sc.zw), df_mul(e, sc.xy));
}

// Principal square root without cancellation on either half-plane.
fn cd_sqrt(z: vec4<f32>) -> vec4<f32> {
    let r = df_sqrt(cd_norm_sqr(z));
    if (r.x == 0.0) {
        return vec4<f32>(0.0);
    }
    let t = df_sqrt(df_mul_f(df_add(r, df_abs(z.xy)), 0.5));
    let two_t = df_mul_f(t, 2.0);
    if (z.x >= 0.0) {
        return cd(t, df_div(z.zw, two_t));
    }
    let im = select(df_neg(t), t, z.z >= 0.0);
    return cd(df_div(df_abs(z.zw), two_t), im);
}
