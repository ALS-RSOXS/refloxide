// Decoupled uniaxial-z Parratt recursion, one invocation per q-point; each
// point names the stack it is evaluated against.
// Mirrors refloxide::kernel::solve_point_recursive operation for operation;
// complex numbers are vec2<f32> as (re, im).

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
}

struct Modes {
    k_o: vec2<f32>,
    k_e: vec2<f32>,
    z: vec2<f32>,
    x_e: vec2<f32>,
    pi: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k0s: array<f32>;
@group(0) @binding(3) var<storage, read> layers: array<Layer>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> stack_of: array<u32>;

const ONE = vec2<f32>(1.0, 0.0);

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

// Principal square root without cancellation on either half-plane.
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

fn modes(l: Layer, neg_kzv2: vec2<f32>, k0sq: f32) -> Modes {
    let e_o = ONE + l.chi_o;
    let one_nu = ONE + cdiv(l.chi_e - l.chi_o, e_o);
    let x_e = l.chi_e * k0sq - neg_kzv2;
    let k_o = csqrt(l.chi_o * k0sq - neg_kzv2);
    let k_e = cdiv(csqrt(cmul(one_nu, x_e)), one_nu);
    let pi = l.chi_o + l.chi_e + cmul(l.chi_o, l.chi_e);
    return Modes(k_o, k_e, cdiv(k_e, e_o), x_e, pi);
}

// Returns (r_s.re, r_s.im, r_p.re, r_p.im) for the interface a -> b.
fn fresnel(a: Layer, ma: Modes, b: Layer, mb: Modes, sigma: f32, k0sq: f32) -> vec4<f32> {
    let s2 = vec2<f32>(-2.0 * sigma * sigma, 0.0);
    let sum_k = ma.k_o + mb.k_o;
    let dk = cdiv((a.chi_o - b.chi_o) * k0sq, sum_k);
    let r_s = cmul(cdiv(dk, sum_k), cexp(cmul(cmul(s2, ma.k_o), mb.k_o)));
    let num = (a.chi_e - b.chi_e) * k0sq + cmul(ma.x_e, mb.pi) - cmul(mb.x_e, ma.pi);
    let sum_z = ma.z + mb.z;
    let dz = cdiv(num, cmul(cmul(ONE + ma.pi, ONE + mb.pi), sum_z));
    let r_p = cmul(cdiv(dz, sum_z), cexp(cmul(cmul(s2, ma.k_e), mb.k_e)));
    return vec4<f32>(r_s, r_p);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.row_pitch;
    if (idx >= params.n_points) {
        return;
    }
    let si = stack_of[idx];
    let n = params.n_layers;
    let base = si * n;
    let k0 = k0s[si];
    let k0sq = k0 * k0;
    let kz_vac = clamp(q[idx] * 0.5, -k0, k0);
    let neg_kzv2 = vec2<f32>(-(kz_vac * kz_vac), 0.0);
    let two_i = vec2<f32>(0.0, 2.0);

    var lb = layers[base + n - 1u];
    var la = layers[base + n - 2u];
    var mb = modes(lb, neg_kzv2, k0sq);
    var ma = modes(la, neg_kzv2, k0sq);
    let r0 = fresnel(la, ma, lb, mb, lb.sigma, k0sq);
    var x_s = r0.xy;
    var x_p = r0.zw;
    lb = la;
    mb = ma;
    for (var j = n - 2u; j >= 1u; j = j - 1u) {
        la = layers[base + j - 1u];
        ma = modes(la, neg_kzv2, k0sq);
        let r = fresnel(la, ma, lb, mb, lb.sigma, k0sq);
        let d = vec2<f32>(lb.thickness, 0.0);
        let ph_s = cexp(cmul(cmul(two_i, mb.k_o), d));
        let ph_p = cexp(cmul(cmul(two_i, mb.k_e), d));
        let xs_ph = cmul(x_s, ph_s);
        let xp_ph = cmul(x_p, ph_p);
        x_s = cdiv(r.xy + xs_ph, ONE + cmul(r.xy, xs_ph));
        x_p = cdiv(r.zw + xp_ph, ONE + cmul(r.zw, xp_ph));
        lb = la;
        mb = ma;
    }
    out[idx] = vec2<f32>(min(dot(x_p, x_p), 1.0), min(dot(x_s, x_s), 1.0));
}
