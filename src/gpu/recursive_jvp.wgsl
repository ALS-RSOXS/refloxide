// Forward-mode derivative of the decoupled uniaxial-z recursion.
// Mirrors refloxide::kernel::solve_point_recursive_jvp: one invocation per
// (point, direction); every complex intermediate is a dual number
// (value, tangent) with exact derivative rules.

struct Layer {
    chi_o: vec2<f32>,
    chi_e: vec2<f32>,
    thickness: f32,
    sigma: f32,
}

struct Params {
    n_points: u32,
    n_layers: u32,
    n_stacks: u32,
    row_pitch: u32,
    n_dirs: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct D {
    v: vec2<f32>,
    t: vec2<f32>,
}

struct Modes {
    chi_o: D,
    chi_e: D,
    k_o: D,
    k_e: D,
    z: D,
    x_e: D,
    pi: D,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k0s: array<f32>;
@group(0) @binding(3) var<storage, read> layers: array<Layer>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> stack_of: array<u32>;
@group(0) @binding(6) var<storage, read> dlayers: array<Layer>;
@group(0) @binding(7) var<storage, read> dq: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let n = b.x * b.x + b.y * b.y;
    return vec2<f32>((a.x * b.x + a.y * b.y) / n, (a.y * b.x - a.x * b.y) / n);
}

fn cexp(z: vec2<f32>) -> vec2<f32> {
    let e = exp(z.x);
    return vec2<f32>(e * cos(z.y), e * sin(z.y));
}

fn csqrt(z: vec2<f32>) -> vec2<f32> {
    let r = length(z);
    if (r == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let t = sqrt(0.5 * (r + abs(z.x)));
    if (z.x >= 0.0) {
        return vec2<f32>(t, z.y / (2.0 * t));
    }
    return vec2<f32>(abs(z.y) / (2.0 * t), select(-t, t, z.y >= 0.0));
}

fn dreal(x: f32) -> D {
    return D(vec2<f32>(x, 0.0), vec2<f32>(0.0, 0.0));
}

fn dadd(a: D, b: D) -> D {
    return D(a.v + b.v, a.t + b.t);
}

fn dsub(a: D, b: D) -> D {
    return D(a.v - b.v, a.t - b.t);
}

fn dmul(a: D, b: D) -> D {
    return D(cmul(a.v, b.v), cmul(a.t, b.v) + cmul(a.v, b.t));
}

fn ddiv(a: D, b: D) -> D {
    let qv = cdiv(a.v, b.v);
    return D(qv, cdiv(a.t - cmul(qv, b.t), b.v));
}

fn dscale(a: D, k: f32) -> D {
    return D(a.v * k, a.t * k);
}

fn dsqrt(a: D) -> D {
    let s = csqrt(a.v);
    return D(s, cdiv(a.t, s * 2.0));
}

fn dexp(a: D) -> D {
    let e = cexp(a.v);
    return D(e, cmul(e, a.t));
}

fn modes(l: Layer, dl: Layer, k0sq: f32, neg_kzv2: D) -> Modes {
    let one = dreal(1.0);
    let chi_o = D(l.chi_o, dl.chi_o);
    let chi_e = D(l.chi_e, dl.chi_e);
    let e_o = dadd(one, chi_o);
    let one_nu = dadd(one, ddiv(dsub(chi_e, chi_o), e_o));
    let x_e = dsub(dscale(chi_e, k0sq), neg_kzv2);
    let k_o = dsqrt(dsub(dscale(chi_o, k0sq), neg_kzv2));
    let k_e = ddiv(dsqrt(dmul(one_nu, x_e)), one_nu);
    let pi = dadd(dadd(chi_o, chi_e), dmul(chi_o, chi_e));
    return Modes(chi_o, chi_e, k_o, k_e, ddiv(k_e, e_o), x_e, pi);
}

struct Fresnel {
    r_s: D,
    r_p: D,
}

fn fresnel(a: Modes, b: Modes, sigma: D, k0sq: f32) -> Fresnel {
    let one = dreal(1.0);
    let s2 = dscale(dmul(sigma, sigma), -2.0);
    let sum_k = dadd(a.k_o, b.k_o);
    let dk = ddiv(dscale(dsub(a.chi_o, b.chi_o), k0sq), sum_k);
    let r_s = dmul(ddiv(dk, sum_k), dexp(dmul(dmul(s2, a.k_o), b.k_o)));
    let num = dsub(
        dadd(dscale(dsub(a.chi_e, b.chi_e), k0sq), dmul(a.x_e, b.pi)),
        dmul(b.x_e, a.pi),
    );
    let sum_z = dadd(a.z, b.z);
    let dz = ddiv(num, dmul(dmul(dadd(one, a.pi), dadd(one, b.pi)), sum_z));
    let r_p = dmul(ddiv(dz, sum_z), dexp(dmul(dmul(s2, a.k_e), b.k_e)));
    return Fresnel(r_s, r_p);
}

fn rt(x: D) -> vec2<f32> {
    let r = dot(x.v, x.v);
    if (r > 1.0) {
        return vec2<f32>(1.0, 0.0);
    }
    return vec2<f32>(r, 2.0 * dot(x.v, x.t));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.row_pitch;
    if (idx >= params.n_points * params.n_dirs) {
        return;
    }
    let pt = idx % params.n_points;
    let dir = idx / params.n_points;
    let si = stack_of[pt];
    let n = params.n_layers;
    let base = si * n;
    let dbase = (dir * params.n_stacks + si) * n;
    let k0 = k0s[si];
    let k0sq = k0 * k0;
    let half_q = q[pt] * 0.5;
    let kz_vac = clamp(half_q, -k0, k0);
    let dkz_vac = select(dq[dir * params.n_points + pt] * 0.5, 0.0, abs(half_q) > k0);
    let neg_kzv2 = D(vec2<f32>(-(kz_vac * kz_vac), 0.0), vec2<f32>(-2.0 * kz_vac * dkz_vac, 0.0));
    let one = dreal(1.0);
    let two_i = D(vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0));

    var lb = layers[base + n - 1u];
    var dlb = dlayers[dbase + n - 1u];
    var la = layers[base + n - 2u];
    var dla = dlayers[dbase + n - 2u];
    var mb = modes(lb, dlb, k0sq, neg_kzv2);
    var ma = modes(la, dla, k0sq, neg_kzv2);
    let f0 = fresnel(ma, mb, D(vec2<f32>(lb.sigma, 0.0), vec2<f32>(dlb.sigma, 0.0)), k0sq);
    var x_s = f0.r_s;
    var x_p = f0.r_p;
    lb = la;
    dlb = dla;
    mb = ma;
    for (var j = n - 2u; j >= 1u; j = j - 1u) {
        la = layers[base + j - 1u];
        dla = dlayers[dbase + j - 1u];
        ma = modes(la, dla, k0sq, neg_kzv2);
        let f = fresnel(ma, mb, D(vec2<f32>(lb.sigma, 0.0), vec2<f32>(dlb.sigma, 0.0)), k0sq);
        let d = D(vec2<f32>(lb.thickness, 0.0), vec2<f32>(dlb.thickness, 0.0));
        let ys = dmul(x_s, dexp(dmul(dmul(two_i, mb.k_o), d)));
        let yp = dmul(x_p, dexp(dmul(dmul(two_i, mb.k_e), d)));
        x_s = ddiv(dadd(f.r_s, ys), dadd(one, dmul(f.r_s, ys)));
        x_p = ddiv(dadd(f.r_p, yp), dadd(one, dmul(f.r_p, yp)));
        lb = la;
        dlb = dla;
        mb = ma;
    }
    let pp = rt(x_p);
    let ss = rt(x_s);
    out[idx] = vec4<f32>(pp.x, ss.x, pp.y, ss.y);
}
