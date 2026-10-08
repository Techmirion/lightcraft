# The Techmirion fork of LightCraft

This repository is [Techmirion](https://github.com/Techmirion)'s fork of
[storytold/lightcraft](https://github.com/storytold/lightcraft). We use it day to day on a Windows
workstation with an RTX 5090 and a Nikon D3500. The fork adds faster raw development on the GPU,
exposure stacking, support for that camera and its lenses, and quicker browsing of large libraries.
The ArtCraft team does not maintain this fork and has not endorsed it. Upstream's licence (MIT OR
Apache-2.0), its brand terms (`docs/brand/`) and its clean-room rules all apply unchanged.

## Branches

| Branch | What it holds |
|---|---|
| `techmirion` (default) | Our work, rebased on upstream `main` |
| `main` | An untouched mirror of upstream `main`, kept so we can sync |

## What the fork adds

The timings below were measured on an RTX 5090 / Windows 11 machine with Nikon D3500 NEFs (24 MP).

**Raw development on the GPU**
- Full-size AHD demosaic, highlight reconstruction and the camera colour transform (white balance,
  matrix, HueSat table) run on the GPU through wgpu (DX12 / Vulkan / Metal) as one pass: one upload
  and one readback. The CPU path is still the reference, and `crates/gpu/tests/rawdev.rs` checks that
  both agree to within one 8-bit level. If a GPU step fails, development falls back to the CPU instead
  of stopping. Loading a full-size raw went from 1.2 s to 0.4 s.
- Batch export is pipelined: the next photos are prepared on worker threads while the current one is
  written. 16 full-size JPEGs took 9.9 s, down from 31 s.
  `LIGHTCRAFT_EXPORT_AHEAD` sets how many photos are prepared ahead (default 2).

**Exposure stacking** (Photo → Photo Merge → HDR → *Method*; CLI: `lightcraft-cli merge hdr --mode …`)
- `hdr` (the default): the existing merge of a bracket.
- `average`: averages the frames to cut noise. With 7 frames, noise drops to about 1/2.7.
- `sigmaClip`: averages after rejecting outliers (2.5 × robust σ), so anything moving through the
  frame (people, cars) is removed.
- `median`, `lighten` (star trails, light painting) and `darken`.
- All frames are aligned and normalised for exposure first, and the result is a DNG named after the
  method, for example `…-Stack-Median.dng`. A merge takes up to 256 frames, and frames are demosaiced
  on the GPU (a 7-frame burst merges in about 7 s, down from 17 s).

**Nikon D3500 and lenses**
- The NEF maker note now provides the lens name and specification (for example "Nikon AF-P 18-55mm
  f/3.5-5.6G VR"), so the lens can be filtered on and the lens profile is picked automatically.
- Lens profiles for the AF-P DX 18-55 VR, AF-P DX 70-300 VR and AF-S DX 35 mm f/1.8G
  (`assets/lens-profiles/`): the distortion, vignetting and chromatic aberration models were converted
  from [lensfun](https://lensfun.github.io/) **data** (CC BY-SA 3.0, credited in
  `assets/ATTRIBUTION.md`). The data is fitted to LightCraft's own correction model, and no lensfun
  code is used. Because of this data, the fork's lens commit cannot be sent upstream as it is.
- On import, a JPEG that sits next to a raw of the same name (RAW+JPEG shooting) is skipped, as
  Lightroom does. Set `LIGHTCRAFT_RAW_JPEG_SEPARATE=1` to import both.

**Browsing large libraries**
- Grid tiles are built from the smallest embedded JPEG that is big enough. Only the file header and
  that JPEG are read, not the whole raw, so a tile that is not in the cache takes 6–8 ms instead of
  50–100 ms.
- Library → Build Previews works on several photos at once. `LIGHTCRAFT_PREVIEW_WORKERS` sets how many
  (default: half the cores, 2–8).
- Background render workers grow with the CPU (up to 12). `LIGHTCRAFT_RENDER_THREADS` overrides the
  number.
- Previews keep their detail. A render that goes through a crop, a rotation or a lens correction is
  resampled at twice the size on screen and then reduced with a Catmull-Rom filter (CPU and GPU alike),
  instead of being resampled bilinearly at screen size. On D3500 photos, a loupe preview went from
  about half to about 80 % of the fine detail of a full-size render reduced to the same size, and grid
  thumbnails went to 93–100 %. Grid tiles also reuse the larger thumbnails that Build Previews makes.

## Building

As upstream (see the README's *Quick start*):

```sh
cargo build --release -p lightcraft -p lightcraft-cli
```

On Windows, wgpu uses DX12. If no suitable GPU is found, everything still works on the CPU.
`LIGHTCRAFT_PROFILE=1` prints a timing for each stage.

## Keeping up with upstream

```sh
git remote add upstream https://github.com/storytold/lightcraft.git   # once
git fetch upstream
git switch main && git merge --ff-only upstream/main && git push origin main
git switch techmirion && git rebase main     # resolve, run `cargo xtask ci`, then:
git push --force-with-lease origin techmirion
```

Each change is kept as its own commit, so anything that upstream's rules allow can still be offered
back as a pull request on its own.
