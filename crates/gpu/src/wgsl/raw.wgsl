// Raw development on the GPU: AHD demosaicing and the camera colour transform.
// Bindings: mos (normalised mosaic, w × h), gh / gv (directional greens), lab_h / lab_v (CIELab of
// the two candidates, 3 per pixel), hom (homogeneity counts: h | v << 8), out (RGB, 3 per pixel).
// P[0] = w, P[1] = h, P[2] = first row of this band, P[3] = rows in this band,
// P[4..8] = CFA colours of the 2 × 2 Bayer tile (row-major, anchored at (0, 0)).
// Each kernel mirrors `lightcraft_raw::demosaic::ahd` (reflected borders = the CPU's mirrored
// tiles) or `lightcraft_engine::files::load_bytes_now`'s colour step; keep them in step.

fn rw() -> i32 {
    return i32(pu(0u));
}

fn rh() -> i32 {
    return i32(pu(1u));
}

// `demosaic::reflect`: mirror without repeating the edge sample (keeps Bayer parity).
fn reflect(i: i32, n: i32) -> i32 {
    if (i >= 0 && i < n) {
        return i;
    }
    if (n == 1) {
        return 0;
    }
    let period = 2 * (n - 1);
    var r = i % period;
    if (r < 0) {
        r = r + period;
    }
    if (r >= n) {
        r = period - r;
    }
    return r;
}

fn idx(x: i32, y: i32) -> u32 {
    return u32(reflect(y, rh()) * rw() + reflect(x, rw()));
}

fn colour_at(x: i32, y: i32) -> u32 {
    let xr = reflect(x, rw());
    let yr = reflect(y, rh());
    return pu(4u + u32((yr & 1) * 2 + (xr & 1)));
}

fn m_at(x: i32, y: i32) -> f32 {
    return mos[idx(x, y)];
}

// Band pixel → (x, y), or false past the band.
fn band_xy(g: vec3<u32>, nw: vec3<u32>) -> vec3<i32> {
    let i = lin_index(g, nw);
    let w = pu(0u);
    if (i >= w * pu(3u)) {
        return vec3<i32>(0, 0, 0);
    }
    return vec3<i32>(i32(i % w), i32(pu(2u) + i / w), 1);
}

// 1. Directional greens (5-tap across the missing site, clamped to the adjacent greens).
@compute @workgroup_size(256)
fn ahd_green(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let p = band_xy(g, nw);
    if (p.z == 0) {
        return;
    }
    let x = p.x;
    let y = p.y;
    let i = idx(x, y);
    let c = mos[i];
    if (colour_at(x, y) == 1u) {
        gh[i] = c;
        gv[i] = c;
        return;
    }
    let l = m_at(x - 1, y);
    let r = m_at(x + 1, y);
    let eh = (l + r) * 0.5 + (2.0 * c - m_at(x - 2, y) - m_at(x + 2, y)) * 0.25;
    gh[i] = clamp(eh, min(l, r), max(l, r));
    let u = m_at(x, y - 1);
    let d = m_at(x, y + 1);
    let ev = (u + d) * 0.5 + (2.0 * c - m_at(x, y - 2) - m_at(x, y + 2)) * 0.25;
    gv[i] = clamp(ev, min(u, d), max(u, d));
}

fn g_dir(v: bool, j: u32) -> f32 {
    if (v) {
        return gv[j];
    }
    return gh[j];
}

// 2. One full candidate (colour-difference interpolation of R/B around its green plane).
fn candidate(v: bool, x: i32, y: i32) -> vec3<f32> {
    let i = idx(x, y);
    let own = colour_at(x, y);
    let gg = g_dir(v, i);
    // no dynamic writes into vector components: FXC (DX12) rejects them ("not natively addressable")
    if (own == 1u) {
        let ch = colour_at(x + 1, y);
        let jl = idx(x - 1, y);
        let jr = idx(x + 1, y);
        let ju = idx(x, y - 1);
        let jd = idx(x, y + 1);
        let hval = gg + ((mos[jl] - g_dir(v, jl)) + (mos[jr] - g_dir(v, jr))) * 0.5;
        let vval = gg + ((mos[ju] - g_dir(v, ju)) + (mos[jd] - g_dir(v, jd))) * 0.5;
        // the row neighbour's colour takes the horizontal estimate, the other one the vertical
        return vec3<f32>(select(vval, hval, ch == 0u), gg, select(vval, hval, ch == 2u));
    }
    let a = idx(x - 1, y - 1);
    let b = idx(x + 1, y - 1);
    let c = idx(x - 1, y + 1);
    let d = idx(x + 1, y + 1);
    let s = (mos[a] - g_dir(v, a)) + (mos[b] - g_dir(v, b)) + (mos[c] - g_dir(v, c)) + (mos[d] - g_dir(v, d));
    let other = gg + s * 0.25;
    let own_v = mos[i];
    return vec3<f32>(select(other, own_v, own == 0u), gg, select(other, own_v, own == 2u));
}

fn lab_f(t: f32) -> f32 {
    if (t > 0.008856) {
        // pow(t, 1/3) refined by one Newton step: closer to the CPU's correctly rounded cbrt, so
        // fewer homogeneity ties are decided differently
        let y = pow(t, 1.0 / 3.0);
        return y - (y * y * y - t) / (3.0 * y * y);
    }
    return 7.787 * t + 16.0 / 116.0;
}

// Linear camera RGB (as sRGB / D65 for the metric only) → CIELab (`ahd::lab`).
fn to_lab(p: vec3<f32>) -> vec3<f32> {
    let x = (0.4124 * p.x + 0.3576 * p.y + 0.1805 * p.z) / 0.95047;
    let yy = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
    let z = (0.0193 * p.x + 0.1192 * p.y + 0.9505 * p.z) / 1.08883;
    let fx = lab_f(x);
    let fy = lab_f(yy);
    let fz = lab_f(z);
    return vec3<f32>(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

@compute @workgroup_size(256)
fn ahd_lab(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let p = band_xy(g, nw);
    if (p.z == 0) {
        return;
    }
    let i = idx(p.x, p.y);
    let a = to_lab(candidate(false, p.x, p.y));
    let b = to_lab(candidate(true, p.x, p.y));
    lab_h[3u * i] = a.x;
    lab_h[3u * i + 1u] = a.y;
    lab_h[3u * i + 2u] = a.z;
    lab_v[3u * i] = b.x;
    lab_v[3u * i + 1u] = b.y;
    lab_v[3u * i + 2u] = b.z;
}

fn lab_of(v: bool, j: u32) -> vec3<f32> {
    if (v) {
        return vec3<f32>(lab_v[3u * j], lab_v[3u * j + 1u], lab_v[3u * j + 2u]);
    }
    return vec3<f32>(lab_h[3u * j], lab_h[3u * j + 1u], lab_h[3u * j + 2u]);
}

fn dl(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return abs(a.x - b.x);
}

fn dc(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let u = a.y - b.y;
    let v = a.z - b.z;
    return u * u + v * v;
}

// 3. Homogeneity: neighbours within the adaptive luminance / chrominance tolerances.
@compute @workgroup_size(256)
fn ahd_hom(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let p = band_xy(g, nw);
    if (p.z == 0) {
        return;
    }
    let x = p.x;
    let y = p.y;
    let i = idx(x, y);
    let n = array<u32, 4>(idx(x - 1, y), idx(x + 1, y), idx(x, y - 1), idx(x, y + 1));
    let hi = lab_of(false, i);
    let vi = lab_of(true, i);
    let hl = lab_of(false, n[0]);
    let hr = lab_of(false, n[1]);
    let vu = lab_of(true, n[2]);
    let vd = lab_of(true, n[3]);
    let eps_l = min(max(dl(hi, hl), dl(hi, hr)), max(dl(vi, vu), dl(vi, vd)));
    let eps_c = min(max(dc(hi, hl), dc(hi, hr)), max(dc(vi, vu), dc(vi, vd)));
    var ch = 0u;
    var cv = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let hj = lab_of(false, n[k]);
        let vj = lab_of(true, n[k]);
        if (dl(hi, hj) <= eps_l && dc(hi, hj) <= eps_c) {
            ch = ch + 1u;
        }
        if (dl(vi, vj) <= eps_l && dc(vi, vj) <= eps_c) {
            cv = cv + 1u;
        }
    }
    hom[i] = ch | (cv << 8u);
}

// 4. Pick the more homogeneous candidate over a 3 × 3 window (average on ties).
@compute @workgroup_size(256)
fn ahd_out(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let p = band_xy(g, nw);
    if (p.z == 0) {
        return;
    }
    let x = p.x;
    let y = p.y;
    var sh = 0u;
    var sv = 0u;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let v = hom[idx(x + dx, y + dy)];
            sh = sh + (v & 255u);
            sv = sv + (v >> 8u);
        }
    }
    var c: vec3<f32>;
    if (sh > sv) {
        c = candidate(false, x, y);
    } else if (sv > sh) {
        c = candidate(true, x, y);
    } else {
        c = (candidate(false, x, y) + candidate(true, x, y)) * 0.5;
    }
    let o = 3u * u32(y * rw() + x);
    out[o] = c.x;
    out[o + 1u] = c.y;
    out[o + 2u] = c.z;
}
