// Per-operation df64 accuracy probe: out[i] = op(a[i], b[i]) as (hi, lo).
// op: 0 add, 1 mul, 2 div, 3 sqrt(a), 4 exp(a), 5 sin(a), 6 cos(a), 7 log(a),
// 8 two-sum error term of (a.hi, b.hi) (reassociation canary).

struct Params {
    n: u32,
    op: u32,
    row_pitch: u32,
    mask: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params.row_pitch;
    if (i >= params.n) {
        return;
    }
    df_mask = params.mask;
    let x = a[i];
    let y = b[i];
    var r = vec2<f32>(0.0);
    switch params.op {
        case 0u: { r = df_add(x, y); }
        case 1u: { r = df_mul(x, y); }
        case 2u: { r = df_div(x, y); }
        case 3u: { r = df_sqrt(x); }
        case 4u: { r = df_exp(x); }
        case 5u: { r = df_sincos(x).xy; }
        case 6u: { r = df_sincos(x).zw; }
        case 7u: { r = df_log(x); }
        default: { r = df_two_sum(x.x, y.x); }
    }
    out[i] = r;
}
