// Highlight reconstruction on the GPU (`lightcraft_raw::highlight::reconstruct`), on demosaiced
// camera RGB before white balance. Bindings: img (RGB, in place), pyr (the chromaticity pyramid:
// 4 floats per entry = r, g, b, valid; levels back to back, level 0 = half resolution).
// Common P: P[0] = w, P[1] = h (full image), P[2..5] = wb, P[5] = clip.
// Per kernel: P[6] = level width, P[7] = level height, P[8] = level offset (entries),
// P[9] = next (coarser) level width, P[10] = its height, P[11] = its offset,
// P[12] = offset of the 1 × 1 top level (hl_apply: is anything valid at all).

fn pyr_get(off: u32, i: u32) -> vec4<f32> {
    let k = 4u * (off + i);
    return vec4<f32>(pyr[k], pyr[k + 1u], pyr[k + 2u], pyr[k + 3u]);
}

fn pyr_set(off: u32, i: u32, v: vec4<f32>) {
    let k = 4u * (off + i);
    pyr[k] = v.x;
    pyr[k + 1u] = v.y;
    pyr[k + 2u] = v.z;
    pyr[k + 3u] = v.w;
}

fn wbv() -> vec3<f32> {
    return vec3<f32>(pf(2u), pf(3u), pf(4u));
}

fn px(i: u32) -> vec3<f32> {
    return vec3<f32>(img[3u * i], img[3u * i + 1u], img[3u * i + 2u]);
}

// `chroma_of`: white-balanced chromaticity of an unclipped pixel; w = 0 when clipped or too dark.
fn chroma(p: vec3<f32>) -> vec4<f32> {
    let clip = pf(5u);
    if (p.x >= clip || p.y >= clip || p.z >= clip) {
        return vec4<f32>(0.0);
    }
    let q = p * wbv();
    let s = q.x + q.y + q.z;
    if (!(s > 1e-4)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(max(q.x, 0.0) / s, max(q.y, 0.0) / s, max(q.z, 0.0) / s, 1.0);
}

// Level 0 (half resolution) straight from the image: mean chroma of the valid pixels of each 2 × 2 block.
@compute @workgroup_size(256)
fn hl_first(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let cw = pu(6u);
    if (i >= cw * pu(7u)) {
        return;
    }
    let x = i % cw;
    let y = i / cw;
    var s = vec3<f32>(0.0);
    var n = 0u;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {
            let sx = 2u * x + dx;
            let sy = 2u * y + dy;
            if (sx < pu(0u) && sy < pu(1u)) {
                let c = chroma(px(sy * pu(0u) + sx));
                if (c.w > 0.5) {
                    s = s + c.xyz;
                    n = n + 1u;
                }
            }
        }
    }
    if (n > 0u) {
        pyr_set(pu(8u), i, vec4<f32>(s / f32(n), 1.0));
    } else {
        pyr_set(pu(8u), i, vec4<f32>(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0));
    }
}

// One pyramid step (`highlight::downsample`): level (P[6..9]) → coarser level (P[9..12]).
@compute @workgroup_size(256)
fn hl_down(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let cw = pu(9u);
    if (i >= cw * pu(10u)) {
        return;
    }
    let x = i % cw;
    let y = i / cw;
    let w = pu(6u);
    let h = pu(7u);
    var s = vec3<f32>(0.0);
    var n = 0u;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {
            let sx = 2u * x + dx;
            let sy = 2u * y + dy;
            if (sx < w && sy < h) {
                let v = pyr_get(pu(8u), sy * w + sx);
                if (v.w > 0.5) {
                    s = s + v.xyz;
                    n = n + 1u;
                }
            }
        }
    }
    if (n > 0u) {
        pyr_set(pu(11u), i, vec4<f32>(s / f32(n), 1.0));
    } else {
        pyr_set(pu(11u), i, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}

// `highlight::upsample`: bilinear sample of a coarse level at fine pixel (x, y).
fn upsample(off: u32, cw: u32, ch: u32, x: u32, y: u32) -> vec3<f32> {
    let fx = clamp((f32(x) + 0.5) / 2.0 - 0.5, 0.0, f32(cw - 1u));
    let fy = clamp((f32(y) + 0.5) / 2.0 - 0.5, 0.0, f32(ch - 1u));
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, cw - 1u);
    let y1 = min(y0 + 1u, ch - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let g00 = pyr_get(off, y0 * cw + x0).xyz;
    let g10 = pyr_get(off, y0 * cw + x1).xyz;
    let g01 = pyr_get(off, y1 * cw + x0).xyz;
    let g11 = pyr_get(off, y1 * cw + x1).xyz;
    let top = g00 + (g10 - g00) * tx;
    let bot = g01 + (g11 - g01) * tx;
    return top + (bot - top) * ty;
}

// Fill the invalid entries of a level (P[6..9]) from the coarser, already filled level (P[9..12]).
@compute @workgroup_size(256)
fn hl_fill(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    let w = pu(6u);
    if (i >= w * pu(7u)) {
        return;
    }
    let v = pyr_get(pu(8u), i);
    if (v.w > 0.5) {
        return;
    }
    pyr_set(pu(8u), i, vec4<f32>(upsample(pu(11u), pu(9u), pu(10u), i % w, i / w), 1.0));
}

// The per-pixel rebuild of clipped channels (`reconstruct`'s final pass). Level 0 is P[6..9].
@compute @workgroup_size(256)
fn hl_apply(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = pu(13u) + lin_index(g, nw);
    let w = pu(0u);
    if (i >= pu(14u)) {
        return;
    }
    let p = px(i);
    let clip = pf(5u);
    let cl = vec3<bool>(p.x >= clip, p.y >= clip, p.z >= clip);
    if (!any(cl)) {
        return;
    }
    let wb = wbv();
    var r = vec3<f32>(1.0 / 3.0);
    if (pyr_get(pu(12u), 0u).w > 0.5) {
        r = upsample(pu(8u), pu(6u), pu(7u), i % w, i / w);
    }
    let q = p * wb;
    var sum = 0.0;
    var k = 0u;
    if (!cl.x && r.x > 1e-3) {
        sum = sum + q.x / r.x;
        k = k + 1u;
    }
    if (!cl.y && r.y > 1e-3) {
        sum = sum + q.y / r.y;
        k = k + 1u;
    }
    if (!cl.z && r.z > 1e-3) {
        sum = sum + q.z / r.z;
        k = k + 1u;
    }
    var o = q;
    if (k == 0u) {
        let max_level = max(max(wb.x, wb.y), wb.z) * clip;
        let v = max(max(max(q.x, q.y), q.z), max_level);
        o = vec3<f32>(v);
    } else {
        let m = sum / f32(k);
        o = vec3<f32>(select(q.x, max(q.x, m * r.x), cl.x), select(q.y, max(q.y, m * r.y), cl.y), select(q.z, max(q.z, m * r.z), cl.z));
    }
    let res = o / wb;
    img[3u * i] = res.x;
    img[3u * i + 1u] = res.y;
    img[3u * i + 2u] = res.z;
}
