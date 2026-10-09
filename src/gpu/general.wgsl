// General-tensor 4x4 reflection-matrix recursion, one invocation per q-point.
// Mirrors refloxide::kernel::solve_point_general (kernel/general.rs) and
// kernel::rmatrix::reflect_chain function for function. Requires a prelude
// defining `real`, `EPS`, `SQRT_EPS`, `rexp` and `rcossin`
// (general_f32.wgsl or general_f64.wgsl).
//
// Complex numbers are vec2<real> as (re, im). Layer records are flat
// arrays of 20 reals: chi row-major as (re, im) pairs (18 values),
// thickness, sigma. Each point writes four reals: the row-major flattening
// of the CPU packing [[R_pp, R_sp], [R_ps, R_ss]]. A singular point writes
// -(1 + layer) in the first slot and zeros elsewhere.

// Array types are spelled out: naga's MSL backend emits distinct structs for
// a named array alias and the same anonymous array, which Metal rejects.
alias C = vec2<real>;

struct Params {
    n_points: u32,
    n_layers: u32,
    n_stacks: u32,
    row_pitch: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_in: array<real>;
@group(0) @binding(2) var<storage, read> k0s: array<real>;
@group(0) @binding(3) var<storage, read> layers: array<real>;
@group(0) @binding(4) var<storage, read_write> out: array<real>;
@group(0) @binding(5) var<storage, read> stack_of: array<u32>;

const STRIDE = 20u;
const C_ONE = vec2<real>(1.0, 0.0);
const C_ZERO = vec2<real>(0.0, 0.0);

// ---------------------------------------------------------------- complex

fn cr(x: real) -> C {
    return C(x, 0.0);
}

fn cm(a: C, b: C) -> C {
    return C(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cd(a: C, b: C) -> C {
    let n = b.x * b.x + b.y * b.y;
    return C((a.x * b.x + a.y * b.y) / n, (a.y * b.x - a.x * b.y) / n);
}

fn cnsq(a: C) -> real {
    return a.x * a.x + a.y * a.y;
}

fn cnorm(a: C) -> real {
    return sqrt(cnsq(a));
}

fn cconj(a: C) -> C {
    return C(a.x, -a.y);
}

fn cexp(z: C) -> C {
    let e = rexp(z.x);
    let cs = rcossin(z.y);
    return C(e * cs.x, e * cs.y);
}

// Principal square root without cancellation on either half-plane.
fn csqrt(z: C) -> C {
    let r = cnorm(z);
    if (r == 0.0) {
        return C_ZERO;
    }
    let t = sqrt(0.5 * (r + abs(z.x)));
    if (z.x >= 0.0) {
        return C(t, z.y / (2.0 * t));
    }
    return C(abs(z.y) / (2.0 * t), select(-t, t, z.y >= 0.0));
}

// Principal complex cube root: single-precision polar seed, two Newton steps.
fn ccbrt(z: C) -> C {
    if (cnorm(z) == 0.0) {
        return z;
    }
    var boost: f32 = 1.0;
    var zf = vec2<f32>(f32(z.x), f32(z.y));
    if (length(zf) < 1.0e-30) {
        boost = 1.0e30;
        zf = vec2<f32>(f32(z.x * 1.0e30), f32(z.y * 1.0e30));
    }
    let rf = pow(length(zf), 1.0 / 3.0);
    let th = atan2(zf.y, zf.x) / 3.0;
    var w = C(real(rf * cos(th)), real(rf * sin(th))) / real(pow(boost, 1.0 / 3.0));
    for (var i = 0; i < 2; i++) {
        w = (w * 2.0 + cd(z, cm(w, w))) / 3.0;
    }
    return w;
}

fn cbrt_norm(x: real) -> real {
    return real(pow(f32(x), 1.0 / 3.0));
}

// ------------------------------------------------------------ 4x4 helpers

fn mm(a: array<array<C, 4>, 4>, b: array<array<C, 4>, 4>) -> array<array<C, 4>, 4> {
    var o: array<array<C, 4>, 4>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            var s = C_ZERO;
            for (var k = 0u; k < 4u; k++) {
                s += cm(a[i][k], b[k][j]);
            }
            o[i][j] = s;
        }
    }
    return o;
}

fn shifted(m: array<array<C, 4>, 4>, q: C) -> array<array<C, 4>, 4> {
    var a = m;
    for (var i = 0u; i < 4u; i++) {
        a[i][i] = a[i][i] - q;
    }
    return a;
}

fn norm4(v: array<C, 4>) -> real {
    var s: real = 0.0;
    for (var i = 0u; i < 4u; i++) {
        s += cnsq(v[i]);
    }
    return sqrt(s);
}

fn det3(m: array<array<C, 3>, 3>) -> C {
    let t0 = cm(m[0][0], cm(m[1][1], m[2][2]) - cm(m[1][2], m[2][1]));
    let t1 = cm(m[0][1], cm(m[1][0], m[2][2]) - cm(m[1][2], m[2][0]));
    let t2 = cm(m[0][2], cm(m[1][0], m[2][1]) - cm(m[1][1], m[2][0]));
    return t0 - t1 + t2;
}

// Cofactor matrix C[i][j] = (-1)^(i+j) det(minor_ij).
fn cofactors(a: array<array<C, 4>, 4>) -> array<array<C, 4>, 4> {
    var o: array<array<C, 4>, 4>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            var minor: array<array<C, 3>, 3>;
            var mr = 0u;
            for (var r = 0u; r < 4u; r++) {
                if (r == i) {
                    continue;
                }
                var mc = 0u;
                for (var c = 0u; c < 4u; c++) {
                    if (c == j) {
                        continue;
                    }
                    minor[mr][mc] = a[r][c];
                    mc++;
                }
                mr++;
            }
            let sgn = select(real(-1.0), real(1.0), (i + j) % 2u == 0u);
            o[i][j] = det3(minor) * sgn;
        }
    }
    return o;
}

struct Inverse {
    ok: bool,
    m: array<array<C, 4>, 4>,
}

// Adjugate inverse; ok = false when |det| < EPS^2 (as exact_inv_4x4_generic).
fn inv4(m: array<array<C, 4>, 4>) -> Inverse {
    let cof = cofactors(m);
    var det = C_ZERO;
    for (var j = 0u; j < 4u; j++) {
        det += cm(m[0][j], cof[0][j]);
    }
    var res: Inverse;
    if (cnorm(det) < EPS * EPS) {
        res.ok = false;
        return res;
    }
    res.ok = true;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            res.m[i][j] = cd(cof[j][i], det);
        }
    }
    return res;
}

// -------------------------------------------------------- eigenvalue solve

struct Berreman {
    delta: array<array<C, 4>, 4>,
    a: array<C, 3>,
}

fn eps_of(chi: array<array<C, 3>, 3>, i: u32, j: u32) -> C {
    if (i == j) {
        return C_ONE + chi[i][j];
    }
    return chi[i][j];
}

fn berreman(chi: array<array<C, 3>, 3>, zeta: real) -> Berreman {
    let z = cr(zeta);
    let ezz = eps_of(chi, 2u, 2u);
    let a_x = -cd(eps_of(chi, 2u, 0u), ezz);
    let a_y = -cd(eps_of(chi, 2u, 1u), ezz);
    let a_h = -cd(z, ezz);
    let e00 = eps_of(chi, 0u, 0u);
    let e01 = eps_of(chi, 0u, 1u);
    let e02 = eps_of(chi, 0u, 2u);
    let e10 = eps_of(chi, 1u, 0u);
    let e11 = eps_of(chi, 1u, 1u);
    let e12 = eps_of(chi, 1u, 2u);
    var r: Berreman;
    r.a = array<C, 3>(a_x, a_y, a_h);
    r.delta[0] = array<C, 4>(cm(z, a_x), C_ONE + cm(z, a_h), cm(z, a_y), C_ZERO);
    r.delta[1] = array<C, 4>(e00 + cm(e02, a_x), cm(e02, a_h), e01 + cm(e02, a_y), C_ZERO);
    r.delta[2] = array<C, 4>(C_ZERO, C_ZERO, C_ZERO, -C_ONE);
    r.delta[3] = array<C, 4>(
        -(e10 + cm(e12, a_x)),
        -cm(e12, a_h),
        cm(z, z) - e11 - cm(e12, a_y),
        C_ZERO
    );
    return r;
}

// Monic characteristic polynomial q^4 + c0 q^3 + c1 q^2 + c2 q + c3.
fn char_poly(delta: array<array<C, 4>, 4>) -> array<C, 4> {
    var coeffs: array<C, 4>;
    var m = delta;
    for (var k = 1u; k <= 4u; k++) {
        if (k > 1u) {
            m = mm(delta, shifted(m, -coeffs[k - 2u]));
        }
        let tr = m[0][0] + m[1][1] + m[2][2] + m[3][3];
        coeffs[k - 1u] = cd(-tr, cr(real(k)));
    }
    return coeffs;
}

fn eval_poly(c: array<C, 4>, x: C) -> array<C, 2> {
    let f = cm(cm(cm(x + c[0], x) + c[1], x) + c[2], x) + c[3];
    let df = cm(cm(x * 4.0 + c[0] * 3.0, x) + c[1] * 2.0, x) + c[2];
    return array<C, 2>(f, df);
}

// A root of m^3 + a m^2 + b m + d with the largest magnitude (Cardano).
fn cubic_root_nonzero(a: C, b: C, d: C) -> C {
    let p = b - cm(a, a) / 3.0;
    let q = cm(cm(a, a), a) * (2.0 / 27.0) - cm(a, b) / 3.0 + d;
    let disc = csqrt(cm(q, q) * 0.25 + cm(cm(p, p), p) / 27.0);
    let u_plus = -q * 0.5 + disc;
    let u_minus = -q * 0.5 - disc;
    var u: C;
    if (cnorm(u_plus) >= cnorm(u_minus)) {
        u = ccbrt(u_plus);
    } else {
        u = ccbrt(u_minus);
    }
    let omega = C(-0.5, 0.8660254037844386);
    let shift = -a / 3.0;
    var best = shift;
    var best_norm: real = -1.0;
    var w = C_ONE;
    for (var i = 0; i < 3; i++) {
        let uk = cm(u, w);
        var root = shift;
        if (cnorm(uk) > 0.0) {
            root = uk - cd(p, uk * 3.0) + shift;
        }
        if (cnorm(root) > best_norm) {
            best_norm = cnorm(root);
            best = root;
        }
        w = cm(w, omega);
    }
    return best;
}

// Roots of the monic quartic by Ferrari's method, each Newton-polished.
fn quartic_roots(c: array<C, 4>) -> array<C, 4> {
    let a = c[0];
    let b = c[1];
    let cc = c[2];
    let d = c[3];
    let a2 = cm(a, a);
    let p = b - a2 * (3.0 / 8.0);
    let r1 = cc - cm(a, b) * 0.5 + cm(a2, a) * (1.0 / 8.0);
    let r0 = d - cm(a, cc) * 0.25 + cm(a2, b) * (1.0 / 16.0) - cm(a2, a2) * (3.0 / 256.0);
    let shift = -a * 0.25;
    let scale = max(cnorm(p) + sqrt(cnorm(r0)) + cbrt_norm(cnorm(r1)), EPS);

    var roots: array<C, 4>;
    if (cnorm(r1) <= EPS * scale * scale * scale) {
        let disc = csqrt(cm(p, p) - r0 * 4.0);
        let y1 = csqrt((-p + disc) * 0.5);
        let y2 = csqrt((-p - disc) * 0.5);
        roots = array<C, 4>(y1, -y1, y2, -y2);
    } else {
        // Resolvent 8 m^3 + 8 p m^2 + (2 p^2 - 8 r0) m - r1^2 = 0, monic form.
        let ca = p;
        let cb = cm(p, p) * 0.25 - r0;
        let cdd = -cm(r1, r1) * (1.0 / 8.0);
        let m = cubic_root_nonzero(ca, cb, cdd);
        let s = csqrt(m * 2.0);
        let t = cd(r1, s * 2.0);
        let half_p_m = p * 0.5 + m;
        let q1 = csqrt(cm(s, s) - (half_p_m + t) * 4.0);
        let q2 = csqrt(cm(s, s) - (half_p_m - t) * 4.0);
        roots = array<C, 4>(
            (s + q1) * 0.5,
            (s - q1) * 0.5,
            (-s + q2) * 0.5,
            (-s - q2) * 0.5
        );
    }
    for (var i = 0u; i < 4u; i++) {
        var r = roots[i] + shift;
        for (var it = 0; it < 8; it++) {
            let fd = eval_poly(c, r);
            let rn = cnorm(r) + 1.0;
            if (cnorm(fd[1]) <= SQRT_EPS * rn * rn * rn) {
                break;
            }
            r = r - cd(fd[0], fd[1]);
        }
        roots[i] = r;
    }
    return roots;
}

// ------------------------------------------------------------ eigenvectors

struct NullVectors {
    right: array<C, 4>,
    left: array<C, 4>,
}

// Right and left null vectors of a rank-3 matrix via adjugate rows/columns.
fn null_vectors(a: array<array<C, 4>, 4>) -> NullVectors {
    let cof = cofactors(a);
    var best_r = cof[0];
    var best_c = array<C, 4>(cof[0][0], cof[1][0], cof[2][0], cof[3][0]);
    for (var i = 0u; i < 4u; i++) {
        let row = cof[i];
        if (norm4(row) > norm4(best_r)) {
            best_r = row;
        }
        let col = array<C, 4>(cof[0][i], cof[1][i], cof[2][i], cof[3][i]);
        if (norm4(col) > norm4(best_c)) {
            best_c = col;
        }
    }
    return NullVectors(best_r, best_c);
}

struct MaybeVec {
    ok: bool,
    v: array<C, 4>,
}

// Mode vector at eigenvalue q with Hy (pin = 1) or Ey (pin = 2) set to 1.
fn mode_vector(delta: array<array<C, 4>, 4>, q: C, pin: u32) -> MaybeVec {
    var res: MaybeVec;
    res.ok = false;
    var b: array<array<C, 3>, 3>;
    b[0] = array<C, 3>(delta[0][0] - q, delta[0][1], delta[0][2]);
    b[1] = array<C, 3>(delta[1][0], delta[1][1] - q, delta[1][2]);
    b[2] = array<C, 3>(delta[3][0], delta[3][1], delta[3][2] + cm(q, q));
    var u0i = 0u;
    var u1i = 2u;
    if (pin != 1u) {
        u1i = 1u;
    }
    var have = false;
    var br1 = 0u;
    var br2 = 1u;
    var bdet = C_ZERO;
    for (var pr = 0u; pr < 3u; pr++) {
        var r1 = 0u;
        var r2 = 1u;
        if (pr == 1u) {
            r2 = 2u;
        }
        if (pr == 2u) {
            r1 = 1u;
            r2 = 2u;
        }
        let det = cm(b[r1][u0i], b[r2][u1i]) - cm(b[r1][u1i], b[r2][u0i]);
        if (!have || cnorm(det) > cnorm(bdet)) {
            have = true;
            br1 = r1;
            br2 = r2;
            bdet = det;
        }
    }
    var b_scale: real = 0.0;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            b_scale = max(b_scale, cnorm(b[i][j]));
        }
    }
    if (cnorm(bdet) <= 1.0e3 * EPS * b_scale * b_scale) {
        return res;
    }
    let rhs1 = -b[br1][pin];
    let rhs2 = -b[br2][pin];
    let u0 = cd(cm(rhs1, b[br2][u1i]) - cm(b[br1][u1i], rhs2), bdet);
    let u1 = cd(cm(b[br1][u0i], rhs2) - cm(rhs1, b[br2][u0i]), bdet);
    var reduced: array<C, 3>;
    reduced[pin] = C_ONE;
    reduced[u0i] = u0;
    reduced[u1i] = u1;
    res.ok = true;
    res.v = array<C, 4>(reduced[0], reduced[1], reduced[2], -cm(q, reduced[2]));
    return res;
}

struct MaybeC {
    ok: bool,
    v: C,
}

// Two-sided Rayleigh quotient w^T Delta v / w^T v.
fn rayleigh(delta: array<array<C, 4>, 4>, v: array<C, 4>, w: array<C, 4>) -> MaybeC {
    var num = C_ZERO;
    var den = C_ZERO;
    for (var i = 0u; i < 4u; i++) {
        var dv = C_ZERO;
        for (var j = 0u; j < 4u; j++) {
            dv += cm(delta[i][j], v[j]);
        }
        num += cm(w[i], dv);
        den += cm(w[i], v[i]);
    }
    var res: MaybeC;
    if (cnorm(den) <= SQRT_EPS * norm4(v) * norm4(w)) {
        res.ok = false;
    } else {
        res.ok = true;
        res.v = cd(num, den);
    }
    return res;
}

struct Pair {
    p: array<C, 4>,
    s: array<C, 4>,
}

// p (Ey = 0, Ex = 1) and s (Ex = 0, Ey = 1) vectors of a 2-D eigenspace.
fn degenerate_pair(a: array<array<C, 4>, 4>) -> Pair {
    var r1b = 0u;
    var r2b = 1u;
    var best_det: real = -1.0;
    for (var r1 = 0u; r1 < 4u; r1++) {
        for (var r2 = r1 + 1u; r2 < 4u; r2++) {
            let det = cnorm(cm(a[r1][1], a[r2][3]) - cm(a[r1][3], a[r2][1]));
            if (det > best_det) {
                best_det = det;
                r1b = r1;
                r2b = r2;
            }
        }
    }
    let det = cm(a[r1b][1], a[r2b][3]) - cm(a[r1b][3], a[r2b][1]);
    var res: Pair;
    for (var f = 0u; f < 2u; f++) {
        let fx = f * 2u;
        let b1 = -a[r1b][fx];
        let b2 = -a[r2b][fx];
        let hy = cd(cm(b1, a[r2b][3]) - cm(a[r1b][3], b2), det);
        let hx = cd(cm(a[r1b][1], b2) - cm(b1, a[r2b][1]), det);
        var v: array<C, 4>;
        v[fx] = C_ONE;
        v[1] = hy;
        v[3] = hx;
        if (f == 0u) {
            res.p = v;
        } else {
            res.s = v;
        }
    }
    return res;
}

// Relative eigen-residual |A v| / |v|.
fn residual(a: array<array<C, 4>, 4>, v: array<C, 4>) -> real {
    var r: array<C, 4>;
    for (var i = 0u; i < 4u; i++) {
        var s = C_ZERO;
        for (var j = 0u; j < 4u; j++) {
            s += cm(a[i][j], v[j]);
        }
        r[i] = s;
    }
    return norm4(r) / norm4(v);
}

// Scales a mode so component `primary` is 1, falling back to `secondary`.
fn gauge(v: array<C, 4>, primary: u32, secondary: u32) -> array<C, 4> {
    let norm = norm4(v);
    var pivot = v[secondary];
    if (cnorm(v[primary]) >= 1.0e-3 * norm) {
        pivot = v[primary];
    }
    return array<C, 4>(cd(v[0], pivot), cd(v[1], pivot), cd(v[2], pivot), cd(v[3], pivot));
}

fn p_likeness(v: array<C, 4>) -> real {
    let ex = cnsq(v[0]);
    let ey = cnsq(v[2]);
    return ex / (ex + ey + EPS * EPS);
}

fn vec_of(delta: array<array<C, 4>, 4>, q: C, null_floor: real) -> array<C, 4> {
    let a_q = shifted(delta, q);
    let v = null_vectors(a_q).right;
    if (norm4(v) <= null_floor) {
        return degenerate_pair(a_q).p;
    }
    return v;
}

fn pinned(delta: array<array<C, 4>, 4>, q: C, v: array<C, 4>, pin: u32) -> MaybeVec {
    if (cnorm(v[pin]) >= 1.0e-3 * norm4(v)) {
        return mode_vector(delta, q, pin);
    }
    return MaybeVec(true, v);
}

fn poynting(v: array<C, 4>) -> real {
    return cm(v[0], cconj(v[1])).x - cm(v[2], cconj(v[3])).x;
}

// ------------------------------------------------------------------ modes

struct ModeSet {
    q: array<C, 4>,
    d: array<array<C, 4>, 4>,
    ez: array<C, 4>,
}

fn modes_new(chi: array<array<C, 3>, 3>, zeta: real) -> ModeSet {
    let bm = berreman(chi, zeta);
    let delta = bm.delta;
    var roots = quartic_roots(char_poly(delta));
    var scale: real = 1.0;
    for (var i = 0u; i < 4u; i++) {
        scale = max(scale, cnorm(roots[i]));
    }
    let deg_tol = 1.0e3 * EPS * scale;
    let im_tol = SQRT_EPS * scale;
    var a_scale: real = 1.0;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            a_scale = max(a_scale, cnorm(delta[i][j]));
        }
    }
    let null_floor = SQRT_EPS * a_scale * a_scale * a_scale;

    for (var r = 0u; r < 4u; r++) {
        for (var it = 0; it < 2; it++) {
            let nv = null_vectors(shifted(delta, roots[r]));
            if (norm4(nv.right) <= null_floor) {
                break;
            }
            let ray = rayleigh(delta, nv.right, nv.left);
            if (!ray.ok) {
                break;
            }
            roots[r] = ray.v;
        }
    }

    var is_fwd: array<bool, 4>;
    for (var i = 0u; i < 4u; i++) {
        let qi = roots[i];
        if (abs(qi.y) > im_tol) {
            is_fwd[i] = qi.y > 0.0;
        } else {
            is_fwd[i] = poynting(vec_of(delta, qi, null_floor)) > 0.0;
        }
    }
    // Stable order: forward first, then descending Im q.
    var order = array<u32, 4>(0u, 1u, 2u, 3u);
    for (var k = 1u; k < 4u; k++) {
        var pos = k;
        while (pos > 0u) {
            let a = order[pos - 1u];
            let b = order[pos];
            let fa = u32(is_fwd[a]);
            let fb = u32(is_fwd[b]);
            let b_higher = fb > fa || (fb == fa && roots[b].y > roots[a].y);
            if (!b_higher) {
                break;
            }
            order[pos - 1u] = b;
            order[pos] = a;
            pos--;
        }
    }

    var res: ModeSet;
    for (var pr = 0u; pr < 2u; pr++) {
        let qa = roots[order[pr * 2u]];
        let qb = roots[order[pr * 2u + 1u]];
        let qm = (qa + qb) * 0.5;
        let a_m = shifted(delta, qm);
        let dps = degenerate_pair(a_m);
        let eigenspace = max(residual(a_m, dps.p), residual(a_m, dps.s)) <= deg_tol;
        var vp = dps.p;
        var vs = dps.s;
        var qp = qm;
        var qs = qm;
        if (!eigenspace) {
            let va = vec_of(delta, qa, null_floor);
            let vb = vec_of(delta, qb, null_floor);
            var vp0 = vb;
            var qp0 = qb;
            var vs0 = va;
            var qs0 = qa;
            if (p_likeness(va) >= p_likeness(vb)) {
                vp0 = va;
                qp0 = qa;
                vs0 = vb;
                qs0 = qb;
            }
            let mp = pinned(delta, qp0, vp0, 1u);
            let ms = pinned(delta, qs0, vs0, 2u);
            if (mp.ok && ms.ok) {
                vp = mp.v;
                qp = qp0;
                vs = ms.v;
                qs = qs0;
            }
        }
        vp = gauge(vp, 1u, 0u);
        vs = gauge(vs, 2u, 3u);
        res.q[pr] = qp;
        res.q[pr + 2u] = qs;
        for (var r = 0u; r < 4u; r++) {
            res.d[r][pr] = vp[r];
            res.d[r][pr + 2u] = vs[r];
        }
    }
    for (var j = 0u; j < 4u; j++) {
        res.ez[j] = cm(bm.a[0], res.d[0][j]) + cm(bm.a[1], res.d[2][j]) + cm(bm.a[2], res.d[1][j]);
    }
    return res;
}

fn mode_e_norms(m: ModeSet) -> array<real, 4> {
    var o: array<real, 4>;
    for (var j = 0u; j < 4u; j++) {
        o[j] = sqrt(cnsq(m.d[0][j]) + cnsq(m.d[2][j]) + cnsq(m.ez[j]));
    }
    return o;
}

// ------------------------------------------------- reflection-matrix chain

const DOWN = array<u32, 2>(0u, 2u);
const UP = array<u32, 2>(1u, 3u);

// Interface matrix K = (I + D_a^-1 dD) o W between a (above) and b (below).
fn interface_matrix(
    di_a: array<array<C, 4>, 4>, kz_a: array<C, 4>, kz_b: array<C, 4>, dd: array<array<C, 4>, 4>, dkz: array<C, 4>, sigma: real
) -> array<array<C, 4>, 4> {
    var k: array<array<C, 4>, 4>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            var s = C_ZERO;
            for (var m = 0u; m < 4u; m++) {
                s += cm(di_a[i][m], dd[m][j]);
            }
            if (i == j) {
                s += C_ONE;
            }
            k[i][j] = s;
        }
    }
    if (sigma != 0.0) {
        let r2_half = cr(sigma * sigma * 0.5);
        for (var s = 0u; s < 4u; s++) {
            let plus = kz_b[s] + kz_a[s];
            let eplus = cexp(-cm(cm(plus, plus), r2_half));
            let eminus = cexp(-cm(cm(dkz[s], dkz[s]), r2_half));
            for (var row = 0u; row < 4u; row++) {
                var f = eplus;
                if ((row + s) % 2u == 0u) {
                    f = eminus;
                }
                k[row][s] = cm(k[row][s], f);
            }
        }
    }
    return k;
}

fn block(k: array<array<C, 4>, 4>, rows: array<u32, 2>, cols: array<u32, 2>) -> array<array<C, 2>, 2> {
    var o: array<array<C, 2>, 2>;
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            o[i][j] = k[rows[i]][cols[j]];
        }
    }
    return o;
}

fn mul2(a: array<array<C, 2>, 2>, b: array<array<C, 2>, 2>) -> array<array<C, 2>, 2> {
    var o: array<array<C, 2>, 2>;
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            o[i][j] = cm(a[i][0], b[0][j]) + cm(a[i][1], b[1][j]);
        }
    }
    return o;
}

fn add2(a: array<array<C, 2>, 2>, b: array<array<C, 2>, 2>) -> array<array<C, 2>, 2> {
    var o: array<array<C, 2>, 2>;
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            o[i][j] = a[i][j] + b[i][j];
        }
    }
    return o;
}

fn layer_chi(rec: u32) -> array<array<C, 3>, 3> {
    var chi: array<array<C, 3>, 3>;
    let base = rec * STRIDE;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            let o = base + 2u * (3u * i + j);
            chi[i][j] = C(layers[o], layers[o + 1u]);
        }
    }
    return chi;
}

fn layer_thickness(rec: u32) -> real {
    return layers[rec * STRIDE + 18u];
}

fn layer_sigma(rec: u32) -> real {
    return layers[rec * STRIDE + 19u];
}

fn fail(idx: u32, layer: u32) {
    out[4u * idx] = -1.0 - real(layer);
    out[4u * idx + 1u] = 0.0;
    out[4u * idx + 2u] = 0.0;
    out[4u * idx + 3u] = 0.0;
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
    let kz_vac = clamp(q_in[idx] * 0.5, -k0, k0);
    let zeta = sqrt(k0 * k0 - kz_vac * kz_vac) / k0;

    var below = modes_new(layer_chi(base + n - 1u), zeta);
    var above = below;
    var g: array<array<C, 2>, 2>;
    for (var j = n - 1u; j >= 1u; j--) {
        above = modes_new(layer_chi(base + j - 1u), zeta);
        let thickness = layer_thickness(base + j);
        let sigma = layer_sigma(base + j);
        if (j < n - 1u) {
            let i_d = C(0.0, thickness);
            for (var a = 0u; a < 2u; a++) {
                for (var b = 0u; b < 2u; b++) {
                    let dk = (below.q[UP[a]] - below.q[DOWN[b]]) * k0;
                    g[a][b] = cm(g[a][b], cexp(-cm(dk, i_d)));
                }
            }
        }
        let inv = inv4(above.d);
        if (!inv.ok) {
            fail(idx, j - 1u);
            return;
        }
        var dd: array<array<C, 4>, 4>;
        var dkz: array<C, 4>;
        var kz_a: array<C, 4>;
        var kz_b: array<C, 4>;
        for (var r = 0u; r < 4u; r++) {
            for (var c = 0u; c < 4u; c++) {
                dd[r][c] = below.d[r][c] - above.d[r][c];
            }
            dkz[r] = (below.q[r] - above.q[r]) * k0;
            kz_a[r] = above.q[r] * k0;
            kz_b[r] = below.q[r] * k0;
        }
        let k = interface_matrix(inv.m, kz_a, kz_b, dd, dkz, sigma);
        let num = add2(block(k, UP, DOWN), mul2(block(k, UP, UP), g));
        let den = add2(block(k, DOWN, DOWN), mul2(block(k, DOWN, UP), g));
        let det = cm(den[0][0], den[1][1]) - cm(den[0][1], den[1][0]);
        let det_norm = cnorm(det);
        if (!(det_norm > 0.0)) {
            fail(idx, j - 1u);
            return;
        }
        var inv_den: array<array<C, 2>, 2>;
        inv_den[0][0] = cd(den[1][1], det);
        inv_den[0][1] = cd(-den[0][1], det);
        inv_den[1][0] = cd(-den[1][0], det);
        inv_den[1][1] = cd(den[0][0], det);
        g = mul2(num, inv_den);
        if (j == 1u) {
            break;
        }
        below = above;
    }
    let norms = mode_e_norms(above);
    // refl(a, b) with a the up index and b the down index; the CPU packing is
    // [[refl(0,0), refl(1,0)], [refl(0,1), refl(1,1)]], flattened row-major.
    for (var b = 0u; b < 2u; b++) {
        for (var a = 0u; a < 2u; a++) {
            let amp = g[a][b] * (norms[UP[a]] / norms[DOWN[b]]);
            out[4u * idx + b * 2u + a] = min(cnsq(amp), 1.0);
        }
    }
}
