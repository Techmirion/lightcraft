//! Raw development on the GPU (`wgsl/raw.wgsl`): AHD demosaicing of a full-size Bayer mosaic and the
//! camera colour transform. The CPU twins (`lightcraft_raw::demosaic`, the colour step of
//! `lightcraft_engine::files`) stay the reference; `tests/rawdev.rs` compares them.

use crate::ctx::{Buf, Gpu, groups1};

/// Rows per dispatch: every submission stays a few million invocations (watchdogs, see render.rs).
const BAND_PIXELS: usize = 4 << 20;

fn bands(w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    let rows = (BAND_PIXELS / w.max(1)).max(1);
    (0..h.div_ceil(rows)).map(move |b| (b * rows, rows.min(h - b * rows)))
}

/// AHD-demosaic a normalised `w × h` mosaic whose 2 × 2 Bayer tile has colours `cfa` (row-major,
/// 0 = R, 1 = G, 2 = B). Returns interleaved RGB.
pub(crate) fn ahd(g: &Gpu, w: usize, h: usize, mosaic: &[f32], cfa: [u8; 4]) -> Vec<[f32; 3]> {
    let n = w * h;
    let mos = g.upload(mosaic);
    let gh = g.buffer(n);
    let gv = g.buffer(n);
    let lab_h = g.buffer(3 * n);
    let lab_v = g.buffer(3 * n);
    let hom = g.buffer(n);
    let out = g.buffer(3 * n);
    let bufs: [Option<&Buf>; 7] = [Some(&mos), Some(&gh), Some(&gv), Some(&lab_h), Some(&lab_v), Some(&hom), Some(&out)];
    let params = |y0: usize, rows: usize| -> Vec<u32> {
        let mut p = vec![w as u32, h as u32, y0 as u32, rows as u32];
        p.extend(cfa.iter().map(|&c| u32::from(c)));
        p
    };
    for kernel in ["ahd_green", "ahd_lab", "ahd_hom", "ahd_out"] {
        for (y0, rows) in bands(w, h) {
            let mut enc = g.encoder();
            g.run(&mut enc, kernel, &params(y0, rows), &bufs, groups1(w * rows));
            g.submit([enc.finish()]);
        }
    }
    drop((gh, gv, lab_h, lab_v, hom));
    g.finish_and_read(g.encoder(), &out, 3 * n)
}

/// The camera colour transform (see [`crate::CameraColour`]) applied to interleaved RGB.
pub(crate) fn colour(g: &Gpu, rgb: &[[f32; 3]], c: &crate::CameraColour<'_>) -> Vec<[f32; 3]> {
    let n = rgb.len();
    let img = g.upload(rgb);
    let f = f32::to_bits;
    let mut p = vec![n as u32];
    p.extend(c.wb.iter().map(|v| f(*v)));
    p.extend(c.matrix.iter().flatten().map(|v| f(*v)));
    p.push(f(c.gain));
    match &c.hue_sat {
        Some(t) => {
            p.push(1);
            p.extend(t.to_prophoto.iter().flatten().map(|v| f(*v)));
            p.extend(t.from_prophoto.iter().flatten().map(|v| f(*v)));
            p.extend([t.hue_divisions as u32, t.sat_divisions as u32, t.val_divisions as u32, u32::from(t.srgb_value)]);
        }
        None => {
            p.push(0);
            p.extend(std::iter::repeat_n(0u32, 22));
        }
    }
    let table = match &c.hue_sat {
        Some(t) => g.upload(t.data),
        None => g.buffer(3),
    };
    let bufs: [Option<&Buf>; 2] = [Some(&table), Some(&img)];
    p.push(0); // P[37]: first pixel of the band
    for start in (0..n).step_by(BAND_PIXELS) {
        let len = BAND_PIXELS.min(n - start);
        if let Some(s) = p.get_mut(37) {
            *s = start as u32;
        }
        if let Some(e) = p.get_mut(0) {
            *e = (start + len) as u32;
        }
        let mut enc = g.encoder();
        g.run(&mut enc, "camera_colour", &p, &bufs, groups1(len));
        g.submit([enc.finish()]);
    }
    g.finish_and_read(g.encoder(), &img, 3 * n)
}
