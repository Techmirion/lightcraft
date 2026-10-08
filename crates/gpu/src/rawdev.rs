//! Raw development on the GPU: AHD demosaicing of a full-size Bayer mosaic (`wgsl/raw.wgsl`),
//! highlight reconstruction (`wgsl/raw_hl.wgsl`) and the camera colour transform
//! (`wgsl/raw_colour.wgsl`). The CPU twins (`lightcraft_raw::demosaic`, `lightcraft_raw::highlight`,
//! the colour step of `lightcraft_engine::files`) stay the reference; `tests/rawdev.rs` compares them.

use crate::ctx::{Buf, Gpu, groups1};

/// Pixels per dispatch: every submission stays a few million invocations (watchdogs, see render.rs).
const BAND_PIXELS: usize = 4 << 20;

fn bands(w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    let rows = (BAND_PIXELS / w.max(1)).max(1);
    (0..h.div_ceil(rows)).map(move |b| (b * rows, rows.min(h - b * rows)))
}

fn run_one(g: &Gpu, kernel: &str, p: &[u32], bufs: &[Option<&Buf>], n: usize) {
    let mut enc = g.encoder();
    g.run(&mut enc, kernel, p, bufs, groups1(n));
    g.submit([enc.finish()]);
}

/// AHD-demosaic into a new device buffer (interleaved RGB). `cfa`: the 2 × 2 tile's colours.
fn ahd_buf(g: &Gpu, w: usize, h: usize, mosaic: &[f32], cfa: [u8; 4]) -> Buf {
    let n = w * h;
    let mos = g.upload(mosaic);
    let gh = g.buffer(n);
    let gv = g.buffer(n);
    let lab_h = g.buffer(3 * n);
    let lab_v = g.buffer(3 * n);
    let hom = g.buffer(n);
    let out = g.buffer(3 * n);
    {
        let bufs: [Option<&Buf>; 7] = [Some(&mos), Some(&gh), Some(&gv), Some(&lab_h), Some(&lab_v), Some(&hom), Some(&out)];
        for kernel in ["ahd_green", "ahd_lab", "ahd_hom", "ahd_out"] {
            for (y0, rows) in bands(w, h) {
                let mut p = vec![w as u32, h as u32, y0 as u32, rows as u32];
                p.extend(cfa.iter().map(|&c| u32::from(c)));
                run_one(g, kernel, &p, &bufs, w * rows);
            }
        }
    }
    out
}

/// AHD demosaic only (see [`crate::demosaic_ahd`]).
pub(crate) fn ahd(g: &Gpu, w: usize, h: usize, mosaic: &[f32], cfa: [u8; 4]) -> Vec<[f32; 3]> {
    let out = ahd_buf(g, w, h, mosaic, cfa);
    g.finish_and_read(g.encoder(), &out, 3 * w * h)
}

/// Highlight reconstruction in place on `img` (`w × h` camera RGB), as `lightcraft_raw::highlight::reconstruct`.
fn highlights(g: &Gpu, img: &Buf, w: usize, h: usize, wb: [f32; 3], clip: f32) {
    // the chromaticity pyramid: level 0 at half resolution, halving down to 1 × 1
    let mut levels = vec![(w.div_ceil(2), h.div_ceil(2), 0usize)];
    while let Some(&(lw, lh, off)) = levels.last() {
        if lw * lh <= 1 {
            break;
        }
        levels.push((lw.div_ceil(2), lh.div_ceil(2), off + lw * lh));
    }
    let total = levels.last().map_or(1, |&(lw, lh, off)| off + lw * lh);
    let top = levels.last().map_or(0, |l| l.2);
    let pyr = g.buffer(4 * total);
    let bufs: [Option<&Buf>; 2] = [Some(img), Some(&pyr)];
    let f = f32::to_bits;
    let params = |lvl: (usize, usize, usize), next: (usize, usize, usize), band: (usize, usize)| -> Vec<u32> {
        vec![
            w as u32,
            h as u32,
            f(wb[0]),
            f(wb[1]),
            f(wb[2]),
            f(clip),
            lvl.0 as u32,
            lvl.1 as u32,
            lvl.2 as u32,
            next.0 as u32,
            next.1 as u32,
            next.2 as u32,
            top as u32,
            band.0 as u32,
            band.1 as u32,
        ]
    };
    let Some(&l0) = levels.first() else { return };
    run_one(g, "hl_first", &params(l0, l0, (0, 0)), &bufs, l0.0 * l0.1);
    for pair in levels.windows(2) {
        if let [a, b] = pair {
            run_one(g, "hl_down", &params(*a, *b, (0, 0)), &bufs, b.0 * b.1);
        }
    }
    for pair in levels.windows(2).rev() {
        if let [a, b] = pair {
            run_one(g, "hl_fill", &params(*a, *b, (0, 0)), &bufs, a.0 * a.1);
        }
    }
    let n = w * h;
    for start in (0..n).step_by(BAND_PIXELS) {
        let len = BAND_PIXELS.min(n - start);
        run_one(g, "hl_apply", &params(l0, l0, (start, start + len)), &bufs, len);
    }
}

/// The camera colour transform in place on `img` (`n` pixels).
fn colour_on(g: &Gpu, img: &Buf, n: usize, c: &crate::CameraColour<'_>) {
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
    let bufs: [Option<&Buf>; 2] = [Some(&table), Some(img)];
    p.push(0); // P[37]: first pixel of the band
    for start in (0..n).step_by(BAND_PIXELS) {
        let len = BAND_PIXELS.min(n - start);
        if let Some(s) = p.get_mut(37) {
            *s = start as u32;
        }
        if let Some(e) = p.get_mut(0) {
            *e = (start + len) as u32;
        }
        run_one(g, "camera_colour", &p, &bufs, len);
    }
}

/// The camera colour transform (see [`crate::CameraColour`]) applied to interleaved RGB.
pub(crate) fn colour(g: &Gpu, rgb: &[[f32; 3]], c: &crate::CameraColour<'_>) -> Vec<[f32; 3]> {
    let img = g.upload(rgb);
    colour_on(g, &img, rgb.len(), c);
    g.finish_and_read(g.encoder(), &img, 3 * rgb.len())
}

/// Demosaic, highlights and colour in one pass on the device: the mosaic goes up, the developed
/// image comes back (see [`crate::develop_raw`]).
pub(crate) fn develop(
    g: &Gpu,
    w: usize,
    h: usize,
    mosaic: &[f32],
    cfa: [u8; 4],
    hl: Option<([f32; 3], f32)>,
    c: &crate::CameraColour<'_>,
) -> Vec<[f32; 3]> {
    let img = ahd_buf(g, w, h, mosaic, cfa);
    if let Some((wb, clip)) = hl {
        highlights(g, &img, w, h, wb, clip);
    }
    colour_on(g, &img, w * h, c);
    g.finish_and_read(g.encoder(), &img, 3 * w * h)
}

/// Highlight reconstruction alone, for the equivalence tests.
pub(crate) fn highlights_only(g: &Gpu, rgb: &[[f32; 3]], w: usize, h: usize, wb: [f32; 3], clip: f32) -> Vec<[f32; 3]> {
    let img = g.upload(rgb);
    highlights(g, &img, w, h, wb, clip);
    g.finish_and_read(g.encoder(), &img, 3 * rgb.len())
}
