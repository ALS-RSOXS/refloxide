// Decoupled uniaxial-z recursion in df64 (double-single) arithmetic.
// Mirrors recursive.wgsl operation for operation; requires df64.wgsl.
// Real df64 values are vec2<f32>(hi, lo); complex are vec4<f32>.

struct Layer {
    chi_o: vec4<f32>,
    chi_e: vec4<f32>,
    thickness: vec2<f32>,
    sigma: vec2<f32>,
}

struct Params {
    n_points: u32,
    n_layers: u32,
    n_stacks: u32,
    row_pitch: u32,
    mask: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Modes {
    k_o: vec4<f32>,
    k_e: vec4<f32>,
    z: vec4<f32>,
    x_e: vec4<f32>,
    pi: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> k0s: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> layers: array<Layer>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> stack_of: array<u32>;

fn cd_real(x: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(x, 0.0, 0.0);
}

fn modes(l: Layer, neg_kzv2: vec4<f32>, k0sq: vec2<f32>) -> Modes {
    let one = cd_real(df(1.0));
    let e_o = cd_add(one, l.chi_o);
    let one_nu = cd_add(one, cd_div(cd_sub(l.chi_e, l.chi_o), e_o));
    let x_e = cd_sub(cd_scale(l.chi_e, k0sq), neg_kzv2);
    let k_o = cd_sqrt(cd_sub(cd_scale(l.chi_o, k0sq), neg_kzv2));
    let k_e = cd_div(cd_sqrt(cd_mul(one_nu, x_e)), one_nu);
    let pi = cd_add(cd_add(l.chi_o, l.chi_e), cd_mul(l.chi_o, l.chi_e));
    return Modes(k_o, k_e, cd_div(k_e, e_o), x_e, pi);
}

struct Fresnel {
    r_s: vec4<f32>,
    r_p: vec4<f32>,
}

// Nevot-Croce factor exp(s2 k_a k_b); exactly 1 for a sharp interface, so
// the (costly) df64 exponential is skipped when sigma == 0.
fn roughness(s2: vec4<f32>, ka: vec4<f32>, kb: vec4<f32>) -> vec4<f32> {
    if (s2.x == 0.0) {
        return cd_real(df(1.0));
    }
    return cd_exp(cd_mul(cd_mul(s2, ka), kb));
}

fn fresnel(a: Layer, ma: Modes, b: Layer, mb: Modes, sigma: vec2<f32>, k0sq: vec2<f32>) -> Fresnel {
    let one = cd_real(df(1.0));
    let s2 = cd_real(df_mul_f(df_mul(sigma, sigma), -2.0));
    let sum_k = cd_add(ma.k_o, mb.k_o);
    let dk = cd_div(cd_scale(cd_sub(a.chi_o, b.chi_o), k0sq), sum_k);
    let r_s = cd_mul(cd_div(dk, sum_k), roughness(s2, ma.k_o, mb.k_o));
    let num = cd_sub(
        cd_add(cd_scale(cd_sub(a.chi_e, b.chi_e), k0sq), cd_mul(ma.x_e, mb.pi)),
        cd_mul(mb.x_e, ma.pi),
    );
    let sum_z = cd_add(ma.z, mb.z);
    let dz = cd_div(num, cd_mul(cd_mul(cd_add(one, ma.pi), cd_add(one, mb.pi)), sum_z));
    let r_p = cd_mul(cd_div(dz, sum_z), roughness(s2, ma.k_e, mb.k_e));
    return Fresnel(r_s, r_p);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.row_pitch;
    if (idx >= params.n_points) {
        return;
    }
    df_mask = params.mask;
    let si = stack_of[idx];
    let n = params.n_layers;
    let base = si * n;
    let k0 = k0s[si];
    let k0sq = df_mul(k0, k0);
    var kz_vac = df_mul_f(q[idx], 0.5);
    if (df_abs(kz_vac).x > k0.x) {
        kz_vac = select(df_neg(k0), k0, kz_vac.x > 0.0);
    }
    let neg_kzv2 = cd_real(df_neg(df_mul(kz_vac, kz_vac)));
    let one = cd_real(df(1.0));
    let two_i = vec4<f32>(0.0, 0.0, 2.0, 0.0);

    var lb = layers[base + n - 1u];
    var la = layers[base + n - 2u];
    var mb = modes(lb, neg_kzv2, k0sq);
    var ma = modes(la, neg_kzv2, k0sq);
    let r0 = fresnel(la, ma, lb, mb, lb.sigma, k0sq);
    var x_s = r0.r_s;
    var x_p = r0.r_p;
    lb = la;
    mb = ma;
    for (var j = n - 2u; j >= 1u; j = j - 1u) {
        la = layers[base + j - 1u];
        ma = modes(la, neg_kzv2, k0sq);
        let r = fresnel(la, ma, lb, mb, lb.sigma, k0sq);
        let d = cd_real(lb.thickness);
        let ph_s = cd_exp(cd_mul(cd_mul(two_i, mb.k_o), d));
        let ph_p = cd_exp(cd_mul(cd_mul(two_i, mb.k_e), d));
        let xs_ph = cd_mul(x_s, ph_s);
        let xp_ph = cd_mul(x_p, ph_p);
        x_s = cd_div(cd_add(r.r_s, xs_ph), cd_add(one, cd_mul(r.r_s, xs_ph)));
        x_p = cd_div(cd_add(r.r_p, xp_ph), cd_add(one, cd_mul(r.r_p, xp_ph)));
        lb = la;
        mb = ma;
    }
    out[idx] = vec4<f32>(cd_norm_sqr(x_p), cd_norm_sqr(x_s));
}
