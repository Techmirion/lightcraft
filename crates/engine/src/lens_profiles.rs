//! Lens profiles for raws that carry no lens corrections of their own (personal fork, Marios).
//!
//! DNG files bring `WarpRectilinear` / `FixVignetteRadial` opcodes ([`crate::files::embedded_lens`]); Nikon DSLR
//! NEFs bring nothing. A lens profile here describes one lens with the three lensfun-style models, per focal
//! length (and aperture for vignetting):
//!
//! * distortion `ptlens`: `Rd = Ru · (a·Ru³ + b·Ru² + c·Ru + 1 − a − b − c)`, radii in units of half the image's
//!   short side (`Ru` corrected, `Rd` source);
//! * lateral CA `poly3` (linear part): red / blue source radius scaled by `vr` / `vb`;
//! * vignetting `pa`: `source = corrected · (1 + k1·r² + k2·r⁴ + k3·r⁶)`, `r` in units of half the diagonal.
//!
//! The models are interpolated to the photo's focal length and f-number and re-expressed as the DNG opcode
//! polynomials the pipeline already applies ([`EmbeddedWarp`], [`EmbeddedVignette`]) by least squares, so the
//! CPU and GPU paths need no change and "Enable Profile Corrections" turns them on and off.
//!
//! Profiles: `<config>/lens-profiles/*.json` (`LIGHTCRAFT_LENS_PROFILES` overrides the folder), then the ones built
//! in from `assets/lens-profiles/`. The built-in ones are converted from the lensfun database (CC BY-SA 3.0, see
//! `assets/ATTRIBUTION.md`): data only, no lensfun code.

use std::path::PathBuf;
use std::sync::OnceLock;

use lightcraft_develop::{EmbeddedLens, EmbeddedVignette, EmbeddedWarp};
use lightcraft_geom::Point;
use serde::{Deserialize, Serialize};

const VERSION: u32 = 1;
const MAX_FILE: u64 = 1 << 20;

pub const BUNDLED: &[(&str, &str)] = &[
    ("nikon-afp-dx-18-55-vr", include_str!("../../../assets/lens-profiles/nikon-afp-dx-18-55-vr.json")),
    ("nikon-afp-dx-70-300-vr", include_str!("../../../assets/lens-profiles/nikon-afp-dx-70-300-vr.json")),
    ("nikon-afs-dx-35-f18g", include_str!("../../../assets/lens-profiles/nikon-afs-dx-35-f18g.json")),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LensProfile {
    pub version: u32,
    pub name: String,
    #[serde(default)]
    pub source: String,
    pub r#match: Match,
    #[serde(default)]
    pub distortion: Vec<Distortion>,
    #[serde(default)]
    pub tca: Vec<Tca>,
    #[serde(default)]
    pub vignetting: Vec<Vignetting>,
}

/// Which photos a profile applies to: the lens spec (min/max focal length, f-number at each) within tolerance,
/// every `model_contains` string found in the lens name, and the camera make if given.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Match {
    pub lens_spec: [f64; 4],
    #[serde(default)]
    pub model_contains: Vec<String>,
    #[serde(default)]
    pub camera_make: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Distortion {
    pub focal: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tca {
    pub focal: f64,
    pub vr: f64,
    pub vb: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vignetting {
    pub focal: f64,
    pub aperture: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
}

impl LensProfile {
    fn valid(&self) -> bool {
        let fin = |v: f64| v.is_finite();
        let [f0, f1, a0, a1] = self.r#match.lens_spec;
        self.version == VERSION
            && f0 > 0.0
            && f1 >= f0
            && a0 > 0.0
            && a1 > 0.0
            && self.distortion.iter().all(|d| d.focal > 0.0 && [d.a, d.b, d.c].iter().all(|v| fin(*v) && v.abs() < 1.0))
            && self.tca.iter().all(|t| t.focal > 0.0 && (0.98..1.02).contains(&t.vr) && (0.98..1.02).contains(&t.vb))
            && self.vignetting.iter().all(|v| v.focal > 0.0 && v.aperture > 0.0 && [v.k1, v.k2, v.k3].iter().all(|k| fin(*k) && k.abs() < 10.0))
    }

    fn matches(&self, make: &str, lens: &str, spec: [f64; 4]) -> bool {
        let m = &self.r#match;
        let close = m.lens_spec.iter().zip(spec).enumerate().all(|(i, (a, b))| (a - b).abs() <= if i < 2 { 0.6 } else { 0.15 });
        let lens = lens.to_ascii_lowercase();
        let named = m.model_contains.iter().all(|s| lens.contains(&s.to_ascii_lowercase()));
        let make_ok = m.camera_make.as_deref().is_none_or(|want| make.to_ascii_lowercase().contains(&want.to_ascii_lowercase()));
        close && named && make_ok
    }
}

pub fn parse(json: &[u8], origin: &str) -> Result<LensProfile, String> {
    let p: LensProfile = serde_json::from_slice(json).map_err(|e| format!("{origin}: {e}"))?;
    if !p.valid() {
        return Err(format!("{origin}: not a usable lens profile ({:?})", p.name));
    }
    Ok(p)
}

pub fn dir() -> Option<PathBuf> {
    std::env::var_os("LIGHTCRAFT_LENS_PROFILES").map(PathBuf::from).or_else(|| crate::camera_profiles::config_dir().map(|d| d.join("lens-profiles")))
}

/// All profiles: the user's folder first (so a local file wins), then the built-in ones. Read once per process.
pub fn all() -> &'static [LensProfile] {
    static ALL: OnceLock<Vec<LensProfile>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut out = Vec::new();
        if let Some(entries) = dir().and_then(|d| std::fs::read_dir(d).ok()) {
            let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            paths.sort();
            for path in paths {
                let small = std::fs::metadata(&path).map(|m| m.len() <= MAX_FILE).unwrap_or(false);
                let read = if small { std::fs::read(&path).map_err(|e| e.to_string()) } else { Err("too large".into()) };
                match read.and_then(|b| parse(&b, &path.display().to_string())) {
                    Ok(p) => out.push(p),
                    Err(e) => eprintln!("lightcraft: ignoring lens profile {e}"),
                }
            }
        }
        for (name, json) in BUNDLED {
            match parse(json.as_bytes(), &format!("built-in lens profile {name}")) {
                Ok(p) => out.push(p),
                Err(e) => eprintln!("lightcraft: {e}"),
            }
        }
        out
    })
}

/// The profile for a lens, if any.
pub fn find(make: &str, lens: &str, spec: [f64; 4]) -> Option<&'static LensProfile> {
    all().iter().find(|p| p.matches(make, lens, spec))
}

/// Lens corrections for a raw without embedded ones, for its default-cropped image `cw × ch` (any orientation:
/// the result is centred and rotation-symmetric).
pub fn for_raw(raw: &lightcraft_raw::RawInfo, cw: f64, ch: f64) -> Option<EmbeddedLens> {
    let m = &raw.metadata;
    let spec = m.lens_spec?;
    let profile = find(m.make.as_deref().unwrap_or(""), m.lens_model.as_deref().unwrap_or(""), spec)?;
    let focal = m.focal_length.filter(|f| f.is_finite() && *f > 0.0).unwrap_or(spec[0]);
    let aperture = m.f_number.filter(|f| f.is_finite() && *f > 0.0).unwrap_or(spec[2]);
    corrections(profile, focal, aperture, cw, ch)
}

/// The profile's corrections at `focal` / `aperture` for a `w × h` image.
pub fn corrections(p: &LensProfile, focal: f64, aperture: f64, w: f64, h: f64) -> Option<EmbeddedLens> {
    if !(w.is_finite() && h.is_finite() && w >= 2.0 && h >= 2.0) {
        return None;
    }
    let long = w.max(h);
    let half_diag = w.hypot(h) / 2.0;
    // our radius unit (half diagonal) in lensfun distortion units (half the short side)
    let s = half_diag / (w.min(h) / 2.0);
    let center = Point::new(0.5, 0.5);
    let radius = half_diag / long;
    let mut lens = EmbeddedLens::default();

    let dist = interp_distortion(&p.distortion, focal);
    let (vr, vb) = interp_tca(&p.tca, focal).unwrap_or((1.0, 1.0));
    if dist.is_some() || (vr, vb) != (1.0, 1.0) {
        let (a, b, c) = dist.unwrap_or((0.0, 0.0, 0.0));
        let d = 1.0 - a - b - c;
        // source/corrected radius ratio at r (half-diagonal units), fitted by kr0 + kr1·r² + kr2·r⁴ + kr3·r⁶
        let ratio = |r: f64| {
            let u = s * r;
            a * u * u * u + b * u * u + c * u + d
        };
        let k = fit_even(ratio, 4, true)?;
        let plane = |scale: f64| [k[0] * scale, k[1] * scale, k[2] * scale, k[3] * scale, 0.0, 0.0];
        lens.warp = Some(EmbeddedWarp { planes: [plane(vr), plane(1.0), plane(vb)], center, radius });
    }
    if let Some((k1, k2, k3)) = interp_vignetting(&p.vignetting, focal, aperture) {
        // correction gain = 1 / (1 + k1 r² + k2 r⁴ + k3 r⁶), fitted by 1 + g0 r² + … + g4 r¹⁰
        let falloff = |r: f64| {
            let r2 = r * r;
            1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        };
        if (0..=20).all(|i| falloff(i as f64 / 20.0) > 0.05) {
            let g = fit_even(|r| 1.0 / falloff(r) - 1.0, 5, false)?;
            lens.vignette = Some(EmbeddedVignette { k: [g[0], g[1], g[2], g[3], g[4]], center, radius });
        }
    }
    (lens.warp.is_some() || lens.vignette.is_some()).then_some(lens)
}

/// Linear interpolation weight of `x` between `x0` and `x1` (clamped).
fn t_of(x: f64, x0: f64, x1: f64) -> f64 {
    if (x1 - x0).abs() < 1e-9 { 0.0 } else { ((x - x0) / (x1 - x0)).clamp(0.0, 1.0) }
}

/// The two entries bracketing `x` by `key` (the same one twice at the ends).
fn bracket<T: Copy>(items: &[T], key: impl Fn(&T) -> f64, x: f64) -> Option<(T, T)> {
    let mut v: Vec<T> = items.to_vec();
    v.sort_by(|a, b| key(a).total_cmp(&key(b)));
    let first = *v.first()?;
    let last = *v.last()?;
    if x <= key(&first) {
        return Some((first, first));
    }
    if x >= key(&last) {
        return Some((last, last));
    }
    v.windows(2).find_map(|w| match w {
        [a, b] if key(a) <= x && x <= key(b) => Some((*a, *b)),
        _ => None,
    })
}

fn interp_distortion(d: &[Distortion], focal: f64) -> Option<(f64, f64, f64)> {
    let (lo, hi) = bracket(d, |e| e.focal, focal)?;
    let t = t_of(focal, lo.focal, hi.focal);
    let l = |a: f64, b: f64| a + (b - a) * t;
    Some((l(lo.a, hi.a), l(lo.b, hi.b), l(lo.c, hi.c)))
}

fn interp_tca(d: &[Tca], focal: f64) -> Option<(f64, f64)> {
    let (lo, hi) = bracket(d, |e| e.focal, focal)?;
    let t = t_of(focal, lo.focal, hi.focal);
    let l = |a: f64, b: f64| a + (b - a) * t;
    Some((l(lo.vr, hi.vr), l(lo.vb, hi.vb)))
}

/// Vignetting at one focal length, interpolated over aperture (in stops).
fn vig_at_focal(entries: &[Vignetting], aperture: f64) -> Option<(f64, f64, f64)> {
    let stops = |a: f64| 2.0 * a.max(0.1).log2();
    let (lo, hi) = bracket(entries, |e| stops(e.aperture), stops(aperture))?;
    let t = t_of(stops(aperture), stops(lo.aperture), stops(hi.aperture));
    let l = |a: f64, b: f64| a + (b - a) * t;
    Some((l(lo.k1, hi.k1), l(lo.k2, hi.k2), l(lo.k3, hi.k3)))
}

fn interp_vignetting(v: &[Vignetting], focal: f64, aperture: f64) -> Option<(f64, f64, f64)> {
    let mut focals: Vec<f64> = v.iter().map(|e| e.focal).collect();
    focals.sort_by(f64::total_cmp);
    focals.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let (f_lo, f_hi) = bracket(&focals, |f| *f, focal)?;
    let at = |f: f64| {
        let entries: Vec<Vignetting> = v.iter().copied().filter(|e| (e.focal - f).abs() < 1e-9).collect();
        vig_at_focal(&entries, aperture)
    };
    let (a, b) = (at(f_lo)?, at(f_hi)?);
    let t = t_of(focal, f_lo, f_hi);
    let l = |x: f64, y: f64| x + (y - x) * t;
    Some((l(a.0, b.0), l(a.1, b.1), l(a.2, b.2)))
}

/// Least-squares fit of `f(r)` on `r ∈ [0, 1]` by `n` even powers: `c0 + c1 r² + …` when `with_constant`, else
/// `c0 r² + c1 r⁴ + …`. Samples are weighted by `r²`: the residual times `r` is a displacement in pixels.
fn fit_even(f: impl Fn(f64) -> f64, n: usize, with_constant: bool) -> Option<Vec<f64>> {
    if n == 0 || n > 6 {
        return None;
    }
    let mut ata = [[0.0f64; 6]; 6];
    let mut atb = [0.0f64; 6];
    let samples = 400;
    for i in 0..=samples {
        let r = i as f64 / samples as f64;
        let w = r * r + 1e-4;
        let y = f(r);
        if !y.is_finite() {
            return None;
        }
        let mut basis = [0.0f64; 6];
        let mut p = if with_constant { 1.0 } else { r * r };
        for b in basis.iter_mut().take(n) {
            *b = p;
            p *= r * r;
        }
        for (row, bi) in ata.iter_mut().zip(basis).take(n) {
            for (cell, bj) in row.iter_mut().zip(basis).take(n) {
                *cell += w * bi * bj;
            }
        }
        for (cell, bi) in atb.iter_mut().zip(basis).take(n) {
            *cell += w * bi * y;
        }
    }
    solve(ata, atb, n)
}

/// Gaussian elimination with partial pivoting on the leading `n × n` block.
fn solve(mut a: [[f64; 6]; 6], mut b: [f64; 6], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mag = |i: usize| a.get(i).and_then(|r| r.get(col)).map_or(0.0, |v| v.abs());
        let pivot = (col..n).max_by(|&i, &j| mag(i).total_cmp(&mag(j)))?;
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = *a.get(col)?.get(col)?;
        if p.abs() < 1e-14 {
            return None;
        }
        let pivot_row = *a.get(col)?;
        let pivot_b = *b.get(col)?;
        for row in (col + 1)..n {
            let factor = a.get(row)?.get(col)? / p;
            let r = a.get_mut(row)?;
            for (x, pv) in r.iter_mut().zip(pivot_row).take(n) {
                *x -= factor * pv;
            }
            *b.get_mut(row)? -= factor * pivot_b;
        }
    }
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut s = *b.get(row)?;
        for col in (row + 1)..n {
            s -= a.get(row)?.get(col)? * x.get(col)?;
        }
        *x.get_mut(row)? = s / a.get(row)?.get(row)?;
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> LensProfile {
        parse(BUNDLED[0].1.as_bytes(), "test").unwrap()
    }

    #[test]
    fn bundled_profiles_parse() {
        for (name, json) in BUNDLED {
            parse(json.as_bytes(), name).unwrap();
        }
    }

    #[test]
    fn matches_afp_18_55_not_afs() {
        let p = profile();
        assert!(p.matches("NIKON CORPORATION", "Nikon AF-P 18-55mm f/3.5-5.6G VR", [18.0, 55.0, 3.5, 5.6]));
        assert!(!p.matches("NIKON CORPORATION", "Nikon 18-55mm f/3.5-5.6G VR", [18.0, 55.0, 3.5, 5.6]));
        assert!(!p.matches("NIKON CORPORATION", "Nikon AF-P 70-300mm f/4.5-6.3G VR", [70.0, 300.0, 4.5, 6.3]));
    }

    #[test]
    fn wide_end_corrects_barrel_and_vignetting() {
        let p = profile();
        let lens = corrections(&p, 18.0, 3.5, 6000.0, 4000.0).unwrap();
        let w = lens.warp.unwrap();
        let f = |r: f64| {
            let k = w.planes[1];
            k[0] + k[1] * r * r + k[2] * r.powi(4) + k[3] * r.powi(6)
        };
        // barrel: the corner of the corrected image samples from inside the source corner
        assert!(f(1.0) < 0.99, "corner ratio {}", f(1.0));
        // matches the ptlens model within a pixel-ish at the corner (half diagonal ≈ 3606 px)
        let s = (6000.0f64).hypot(4000.0) / 4000.0;
        let (a, b, c) = (0.01368, -0.06635, 0.05634);
        let exact = a * s.powi(3) + b * s * s + c * s + 1.0 - a - b - c;
        assert!(((f(1.0) - exact) * 3606.0).abs() < 3.0, "fit {} exact {exact}", f(1.0));
        let v = lens.vignette.unwrap();
        let gain = 1.0 + v.k.iter().sum::<f64>();
        assert!(gain > 1.5, "corner gain {gain}");
    }

    #[test]
    fn solve_small_system() {
        let x = fit_even(|r| 1.0 + 2.0 * r * r - 0.5 * r.powi(4), 3, true).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9 && (x[1] - 2.0).abs() < 1e-9 && (x[2] + 0.5).abs() < 1e-9);
    }
}

#[cfg(test)]
mod sample_tests {
    /// `LC_SAMPLE_NEF=/path/to/DSC.NEF cargo test -p lightcraft-engine --lib sample_nef -- --nocapture`
    #[test]
    fn sample_nef() {
        let Some(path) = std::env::var_os("LC_SAMPLE_NEF") else { return };
        let bytes = std::fs::read(path).unwrap();
        let info = crate::files::probe_bytes("x.NEF", &bytes).unwrap();
        println!("lens {:?} focal {:?} f {:?}", info.meta.lens, info.meta.focal_mm, info.meta.aperture);
        println!("embedded_lens {:?}", info.embedded_lens);
    }
}
