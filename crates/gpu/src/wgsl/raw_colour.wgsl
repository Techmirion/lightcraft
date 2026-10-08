// Camera colour on the GPU. Bindings: table (hue/saturation table), img (RGB, in place).
// Camera colour (`files::load_bytes_now`): white balance, camera matrix, baseline gain, and
// the fitted hue/saturation table (`camera_preview::HueSat` over `HsvTable::apply`).
// P[0] = end of this band (pixels), P[1..4] wb, P[4..13] matrix (row-major), P[13] gain, P[14] table on (0/1),
// P[15..24] to-ProPhoto, P[24..33] from-ProPhoto, P[33] hue divisions, P[34] sat divisions,
// P[35] val divisions, P[36] value axis sRGB-encoded (0/1), P[37] first pixel of this band. Table entries in `table`: 3 floats each,
// value-major, hue-middle, saturation-minor. The image is `img` (in place).

fn pm(base: u32, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        pf(base) * v.x + pf(base + 1u) * v.y + pf(base + 2u) * v.z,
        pf(base + 3u) * v.x + pf(base + 4u) * v.y + pf(base + 5u) * v.z,
        pf(base + 6u) * v.x + pf(base + 7u) * v.y + pf(base + 8u) * v.z,
    );
}

fn srgb_enc(x: f32) -> f32 {
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn srgb_dec(x: f32) -> f32 {
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn entry(v: u32, h: u32, s: u32) -> vec3<f32> {
    let k = 3u * ((v * pu(33u) + h) * pu(34u) + s);
    return vec3<f32>(table[k], table[k + 1u], table[k + 2u]);
}

fn axis(x: f32, n: u32) -> vec3<f32> {
    let f = clamp(x, 0.0, 1.0) * f32(n - 1u);
    let i0 = min(u32(f), max(n, 2u) - 2u);
    let i1 = min(i0 + 1u, n - 1u);
    return vec3<f32>(f32(i0), f32(i1), clamp(f - f32(i0), 0.0, 1.0));
}

fn table_plane(vi: u32, h0: u32, h1: u32, th: f32, s0: u32, s1: u32, ts: f32) -> vec3<f32> {
    let lo = mix(entry(vi, h0, s0), entry(vi, h0, s1), ts);
    let hi = mix(entry(vi, h1, s0), entry(vi, h1, s1), ts);
    return mix(lo, hi, th);
}

fn lookup(h: f32, s: f32, v: f32) -> vec3<f32> {
    let nh = pu(33u);
    let hf = clamp(rem_euclid(h, 360.0) / 360.0 * f32(nh), 0.0, f32(nh));
    let h0 = min(u32(hf), nh - 1u);
    let h1 = (h0 + 1u) % nh;
    let th = clamp(hf - f32(h0), 0.0, 1.0);
    let sa = axis(s, pu(34u));
    let s0 = u32(sa.x);
    let s1 = u32(sa.y);
    if (pu(35u) > 1u) {
        let va = axis(v, pu(35u));
        let v0 = u32(va.x);
        let v1 = u32(va.y);
        let a = table_plane(v0, h0, h1, th, s0, s1, sa.z);
        if (v0 == v1) {
            return a;
        }
        return mix(a, table_plane(v1, h0, h1, th, s0, s1, sa.z), va.z);
    }
    return table_plane(0u, h0, h1, th, s0, s1, sa.z);
}

fn hue_of(c: vec3<f32>, mx: f32, d: f32) -> f32 {
    if (d <= 0.0) {
        return 0.0;
    }
    var h: f32;
    if (mx == c.x) {
        h = rem_euclid((c.y - c.z) / d, 6.0);
    } else if (mx == c.y) {
        h = (c.z - c.x) / d + 2.0;
    } else {
        h = (c.x - c.y) / d + 4.0;
    }
    return h * 60.0;
}

fn limited(x: f32, k: f32) -> f32 {
    if (x <= 1.0) {
        return min(x * k, 1.0);
    }
    return x * min(k, 1.0);
}

fn hsv_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let hh = rem_euclid(h, 360.0) / 60.0;
    let k = min(u32(hh), 5u);
    let f = hh - f32(k);
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    switch k {
        case 0u: { return vec3<f32>(v, t, p); }
        case 1u: { return vec3<f32>(q, v, p); }
        case 2u: { return vec3<f32>(p, v, t); }
        case 3u: { return vec3<f32>(p, q, v); }
        case 4u: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

fn table_apply(rgb: vec3<f32>) -> vec3<f32> {
    let mx = max(max(rgb.x, rgb.y), rgb.z);
    if (!(mx > 0.0) || mx > 3.0e38) {
        return rgb;
    }
    let mn = min(min(rgb.x, rgb.y), rgb.z);
    let d = mx - mn;
    let hue = hue_of(rgb, mx, d);
    let sat = d / mx;
    let srgb_v = pu(36u) == 1u;
    var vi = mx;
    if (srgb_v) {
        vi = srgb_enc(min(mx, 1.0));
    }
    let e = lookup(hue, sat, vi);
    let s2 = limited(sat, e.y);
    var v2: f32;
    if (srgb_v && mx <= 1.0) {
        v2 = srgb_dec(min(srgb_enc(mx) * e.z, 1.0));
    } else if (srgb_v) {
        v2 = mx * min(srgb_dec(min(e.z, 1.0)), 1.0);
    } else {
        v2 = limited(mx, e.z);
    }
    return hsv_rgb(hue + e.x, s2, v2);
}

@compute @workgroup_size(256)
fn camera_colour(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = pu(37u) + lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let p = vec3<f32>(img[3u * i], img[3u * i + 1u], img[3u * i + 2u]);
    let c = p * vec3<f32>(pf(1u), pf(2u), pf(3u));
    var rgb = pm(4u, c) * pf(13u);
    if (pu(14u) == 1u) {
        let t = pm(24u, table_apply(pm(15u, rgb)));
        let before = lum2020(rgb);
        let after = lum2020(t);
        let finite = all(abs(t) < vec3<f32>(3.0e38)) && all(t == t);
        if (before > 0.0 && after > 0.0 && finite) {
            rgb = t * (before / after);
        }
    }
    let o = max(rgb, vec3<f32>(0.0));
    img[3u * i] = o.x;
    img[3u * i + 1u] = o.y;
    img[3u * i + 2u] = o.z;
}
