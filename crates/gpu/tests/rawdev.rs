//! GPU raw development ↔ the CPU reference: AHD demosaicing (`lightcraft_raw::demosaic`) and the
//! camera colour step. Skips (passes with a note) when no GPU adapter exists, e.g. on CI.

use lightcraft_raw::{Cfa, Method, Normalized, demosaic};

fn gpu() -> bool {
    let ok = lightcraft_gpu::available();
    if !ok {
        eprintln!("skipped: no GPU adapter");
    }
    ok
}

/// A smooth colourful scene with sharp edges and fine stripes, mosaicked through `pattern`.
fn mosaic(w: usize, h: usize, pattern: &str) -> Normalized {
    let cfa = Cfa::bayer(pattern).unwrap();
    let mut data = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let edge = if (x / 37 + y / 53) % 2 == 0 { 1.0 } else { 0.35 };
            let stripe = if (x / 2) % 2 == 0 && y > h / 2 { 0.8 } else { 1.0 };
            let rgb = [0.2 + 0.7 * fx * edge, (0.15 + 0.6 * fy) * stripe, 0.1 + 0.5 * (1.0 - fx) * fy + 0.2 * ((x * y) % 7) as f32 / 7.0];
            data[y * w + x] = rgb[cfa.color_at(x, y) as usize];
        }
    }
    Normalized { width: w, height: h, cpp: 1, data, cfa: Some(cfa) }
}

fn compare(a: &[[f32; 3]], b: &[[f32; 3]]) -> (f64, f32, usize) {
    let (mut sum, mut max, mut big) = (0f64, 0f32, 0usize);
    for (p, q) in a.iter().zip(b) {
        for c in 0..3 {
            let d = (p[c] - q[c]).abs();
            sum += d as f64;
            max = max.max(d);
            if d > 1e-3 {
                big += 1;
            }
        }
    }
    (sum / (3 * a.len()) as f64, max, big)
}

#[test]
fn ahd_matches_the_cpu() {
    if !gpu() {
        return;
    }
    for pattern in ["RGGB", "GRBG", "BGGR", "GBRG"] {
        let (w, h) = (2050, 2048);
        let n = mosaic(w, h, pattern);
        let cpu = demosaic(&n, Method::Ahd);
        let cfa = n.cfa.as_ref().unwrap();
        let pat: [u8; 4] = cfa.pattern.as_slice().try_into().unwrap();
        let gpu = lightcraft_gpu::demosaic_ahd(w, h, &n.data, pat).expect("gpu demosaic");
        let (mean, max, big) = compare(&cpu.data, &gpu);
        eprintln!("{pattern}: mean |Δ| {mean:.2e}, max {max:.2e}, {big} values > 1e-3 of {}", 3 * w * h);
        // f32 on both sides: rounding differences, and a rare homogeneity tie decided the other way
        assert!(mean < 1e-5, "{pattern}: mean {mean}");
        assert!((big as f64) < 1e-4 * (3 * w * h) as f64, "{pattern}: {big} values differ by more than 1e-3");
    }
}

#[test]
fn small_or_odd_inputs_stay_on_the_cpu() {
    assert!(lightcraft_gpu::demosaic_ahd(64, 64, &[0.0; 64 * 64], [0, 1, 1, 2]).is_none());
    assert!(lightcraft_gpu::demosaic_ahd(4096, 1024, &[0.0; 10], [0, 1, 1, 2]).is_none());
    assert!(lightcraft_gpu::demosaic_ahd(4096, 1024, &vec![0.0; 4096 * 1024], [0, 1, 3, 2]).is_none());
}

#[test]
fn camera_colour_matches_the_cpu_formula() {
    if !gpu() {
        return;
    }
    let n = 2048 * 2048;
    let img: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            [(t * 37.0).fract() * 1.3, (t * 11.0).fract(), (t * 5.0).fract() * 0.7]
        })
        .collect();
    let wb = [2.05f32, 1.0, 1.4];
    let m = [[1.6f32, -0.5, -0.1], [-0.2, 1.4, -0.2], [0.0, -0.4, 1.4]];
    let gain = 1.25f32;
    // a 72 × 5 × 5 table with a smooth hue shift / saturation / value scale
    let (hd, sd, vd) = (72usize, 5usize, 5usize);
    let table: Vec<[f32; 3]> = (0..hd * sd * vd)
        .map(|k| {
            let h = (k / sd) % hd;
            let s = k % sd;
            [((h as f32) * 0.3).sin() * 4.0, 1.0 + 0.05 * s as f32, 0.97 + 0.01 * (k / (sd * hd)) as f32]
        })
        .collect();
    let hsv = lightcraft_raw::profile::HsvTable { hue_divisions: hd, sat_divisions: sd, val_divisions: vd, data: table.clone(), srgb_value: true };
    let to = [[0.8f32, 0.15, 0.05], [0.05, 0.9, 0.05], [0.0, 0.1, 0.9]];
    let from = invert(to);
    let lum = |c: [f32; 3]| c[0] * 0.2627 + c[1] * 0.6780 + c[2] * 0.0593;
    let mul = |m: &[[f32; 3]; 3], v: [f32; 3]| -> [f32; 3] { std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]) };
    let cpu: Vec<[f32; 3]> = img
        .iter()
        .map(|p| {
            let c = [p[0] * wb[0], p[1] * wb[1], p[2] * wb[2]];
            let rgb = mul(&m, c).map(|v| v * gain);
            let out = mul(&from, hsv.apply(mul(&to, rgb)));
            let (b, a) = (lum(rgb), lum(out));
            let rgb = if b > 0.0 && a > 0.0 && out.iter().all(|v| v.is_finite()) { out.map(|v| v * b / a) } else { rgb };
            rgb.map(|v| v.max(0.0))
        })
        .collect();
    let colour = lightcraft_gpu::CameraColour {
        wb,
        matrix: m,
        gain,
        hue_sat: Some(lightcraft_gpu::HueSatTable {
            to_prophoto: to,
            from_prophoto: from,
            hue_divisions: hd,
            sat_divisions: sd,
            val_divisions: vd,
            srgb_value: true,
            data: table.as_flattened(),
        }),
    };
    let gpu = lightcraft_gpu::camera_colour(&img, &colour).expect("gpu colour");
    let (mean, max, big) = compare(&cpu, &gpu);
    eprintln!("colour: mean |Δ| {mean:.2e}, max {max:.2e}, {big} values > 1e-3");
    assert!(mean < 1e-5 && (big as f64) < 1e-4 * (3 * n) as f64, "mean {mean}, max {max}, {big} big");
}

fn invert(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let m = m.map(|r| r.map(f64::from));
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let inv = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    inv.map(|r| r.map(|v| (v / det) as f32))
}
