//! NeMo `AudioToMelSpectrogramPreprocessor`-compatible log-mel, for the Persian
//! model (ADR-0047). Nothing else in this service computes features: `parakeet-rs`
//! does its own internally and exposes none of it (`mod audio` is private), so a
//! model driven through `ort` directly has to bring its own.
//!
//! The parameters are not guessed. They are `model_config.yaml` out of the source
//! `.nemo`, which `export_manifest.json` names as the origin of *both* published
//! exports:
//!
//! ```text
//! sample_rate 16000  normalize NA   window hann
//! window_size 0.025  window_stride 0.01   features 80   n_fft 512
//! dither 1.0e-05     pad_to 0
//! ```
//!
//! `preemph` and `log_zero_guard_value` are absent from that config, so NeMo's
//! defaults apply: 0.97 and 2^-24. `dither` is deliberately NOT implemented — it is
//! additive training noise, and at inference it would make the same audio decode
//! differently on every call.
//!
//! ## The one trap that actually bites
//!
//! `normalize: NA` means **no per-feature normalisation**. NeMo's usual default is
//! `per_feature` CMVN, and `parakeet-rs`'s own pipeline applies it — so the obvious
//! thing to copy is the wrong thing. Measured against the real model: applying CMVN
//! takes character error rate from 0.033 to **1.000 with empty output**. That is at
//! least a loud failure rather than a quiet one, but it is the failure this file
//! exists to avoid, which is why the absence is commented rather than merely absent.
//!
//! For contrast, two things that were *expected* to be traps and measurably are not:
//! placing the 400-tap window at offset 0 instead of centring it in the 512-point
//! FFT scores 0.042, and dropping preemphasis entirely scores 0.056, against a 0.033
//! baseline. Both are implemented correctly here anyway — matching NeMo costs
//! nothing — but neither is load-bearing, and a future reader should not treat them
//! as though they were.

use anyhow::{bail, Context, Result};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const N_MELS: usize = 80;
pub const N_FFT: usize = 512;
pub const WIN_LEN: usize = 400;
pub const HOP: usize = 160;
/// `center: true` with `pad_mode: reflect`; torch uses `n_fft / 2`.
pub const CENTER_PAD: usize = N_FFT / 2;
const N_BINS: usize = N_FFT / 2 + 1; // 257
const PREEMPH: f32 = 0.97;
/// `log_zero_guard_type: add`, NeMo default value 2^-24.
const GUARD: f32 = 5.960_464_5e-8;

/// The Slaney filterbank, shipped by the model author rather than recomputed.
///
/// `librosa.filters.mel(htk=False, norm='slaney')` is about forty lines of ramp and
/// area-normalisation arithmetic whose failure mode is a plausible-looking matrix
/// that is subtly wrong — exactly the class of bug this file is trying not to have.
/// The authors publish the exact matrix, so the arithmetic is theirs and already
/// validated. `include_str!` follows the pattern `service.rs` already uses for the
/// web assets, and keeps this runnable with no PVC and no network.
const MEL_FILTERS_JSON: &str = include_str!("../assets/mel_filters_slaney_80x257.json");

/// Streaming log-mel. One per session: it carries the unconsumed samples and the
/// preemphasis state across chunk boundaries.
pub struct Fbank {
    /// Row-major `[N_MELS][N_BINS]`, flattened. Flat rather than nested because the
    /// inner loop walks one row against the whole power spectrum.
    filters: Vec<f32>,
    /// Hann(400, periodic=false) already zero-padded to `N_FFT` and centred.
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    /// Post-emphasis samples not yet consumed by a frame. Before the first frame is
    /// emitted this also accumulates the utterance opening, because the centre pad
    /// mirrors it and cannot be built until enough of it has arrived.
    ///
    /// Carries `N_FFT - HOP` between calls in steady state — **not** `WIN_LEN - HOP`:
    /// a frame consumes `N_FFT` samples even though only `WIN_LEN` of them are
    /// weighted, so the overlap is 352 and a 240 here would silently drop audio.
    pending: Vec<f32>,
    /// Last *raw* sample of the previous chunk; `None` until the utterance starts.
    /// Preemphasis is a one-tap IIR, so without this every chunk boundary re-applies
    /// the utterance-start special case and injects a discontinuity every 560 ms.
    prev_raw: Option<f32>,
    /// Whether the left reflect-pad has been emitted.
    started: bool,
    scratch_in: Vec<f32>,
    scratch_out: Vec<Complex<f32>>,
}

impl Fbank {
    pub fn new() -> Result<Self> {
        let rows: Vec<Vec<f32>> =
            serde_json::from_str(MEL_FILTERS_JSON).context("parsing the bundled mel filterbank")?;
        if rows.len() != N_MELS || rows.iter().any(|r| r.len() != N_BINS) {
            bail!(
                "bundled mel filterbank is {}x{}, expected {N_MELS}x{N_BINS}",
                rows.len(),
                rows.first().map_or(0, Vec::len)
            );
        }

        // Hann, periodic = false — i.e. symmetric, divided by len-1 rather than len.
        // Centred in the FFT the way torch.stft pads a short window.
        let mut window = vec![0.0f32; N_FFT];
        let off = (N_FFT - WIN_LEN) / 2;
        for (i, w) in window[off..off + WIN_LEN].iter_mut().enumerate() {
            let phase = std::f32::consts::TAU * i as f32 / (WIN_LEN - 1) as f32;
            *w = 0.5 - 0.5 * phase.cos();
        }

        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let scratch_out = fft.make_output_vec();
        Ok(Self {
            filters: rows.into_iter().flatten().collect(),
            window,
            fft,
            pending: Vec::new(),
            prev_raw: None,
            started: false,
            scratch_in: Vec::with_capacity(N_FFT),
            scratch_out,
        })
    }

    /// Feed one chunk, get back whole frames.
    ///
    /// `last` closes the utterance and applies the right-hand reflect pad. It is
    /// legal — and normal — to call with an empty slice and `last: true`: the
    /// service's bare-close path does exactly that, so this must not panic.
    ///
    /// The result does not depend on how the audio is divided across calls. That is
    /// a contract, not an accident: the offline path feeds engine-sized slices while
    /// the browser sends 560 ms ones, and both must decode identically.
    pub fn push(&mut self, samples: &[f32], last: bool) -> Vec<[f32; N_MELS]> {
        let emphasised = self.preemphasise(samples);

        // Taken out of `self` so `frame` can hold `&mut self` for its scratch
        // buffers while this stays borrowed.
        let mut buf = std::mem::take(&mut self.pending);
        buf.extend_from_slice(&emphasised);

        if !self.started {
            // The centre reflect-pad mirrors the first CENTER_PAD samples of the
            // UTTERANCE, so it cannot be built from a chunk shorter than that.
            // Building it early zero-fills instead of mirroring and corrupts the
            // first one or two frames — silently, which is the whole thing this
            // module exists to avoid. Wait instead; it costs 16 ms of latency once.
            if buf.len() <= CENTER_PAD && !last {
                self.pending = buf;
                return Vec::new();
            }
            let pad = reflect_pad_front(&buf, CENTER_PAD);
            buf.splice(0..0, pad);
            self.started = true;
        }

        if last {
            let pad = reflect_pad_back(&buf, CENTER_PAD);
            buf.extend(pad);
        }

        let mut out = Vec::new();
        let mut start = 0;
        while start + N_FFT <= buf.len() {
            out.push(self.frame(&buf[start..start + N_FFT]));
            start += HOP;
        }

        // Keep what a future frame will still need. On `last` there is no future.
        self.pending = if last {
            Vec::new()
        } else {
            buf[start..].to_vec()
        };
        out
    }

    /// One-tap preemphasis, `y[i] = x[i] - 0.97*x[i-1]`, applied to the raw signal
    /// before any padding to match NeMo's ordering.
    ///
    /// `prev_raw` is an Option rather than a float plus a flag because "no
    /// predecessor" is a genuinely different case from "predecessor was silence":
    /// NeMo leaves the utterance's very first sample unfiltered.
    fn preemphasise(&mut self, samples: &[f32]) -> Vec<f32> {
        let out: Vec<f32> = samples
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let prev = if i == 0 {
                    self.prev_raw
                } else {
                    Some(samples[i - 1])
                };
                match prev {
                    None => s,
                    Some(p) => s - PREEMPH * p,
                }
            })
            .collect();
        if let Some(&lastraw) = samples.last() {
            self.prev_raw = Some(lastraw);
        }
        out
    }

    fn frame(&mut self, samples: &[f32]) -> [f32; N_MELS] {
        self.scratch_in.clear();
        self.scratch_in
            .extend(samples.iter().zip(&self.window).map(|(s, w)| s * w));
        // Both lengths are compile-time constants agreed with the planner, so a
        // failure here is a programming error — and this module's whole argument is
        // that a wrong number must never pass silently. Swallowing it would hand
        // back a zeroed spectrum and a plausible transcript.
        self.fft
            .process(&mut self.scratch_in, &mut self.scratch_out)
            .expect("FFT length agreed with the planner at construction");

        // `magnitude_power_2_no_fft_normalization` — squared magnitude, and NO 1/N
        // scaling. realfft's forward transform is already unnormalised, so this is
        // the absence of a division rather than the presence of one.
        let mut mels = [0.0f32; N_MELS];
        // as_chunks over chunks_exact: N_BINS is a constant, so the remainder is
        // provably empty. Same idiom as engine.rs.
        let (rows, remainder) = self.filters.as_chunks::<N_BINS>();
        debug_assert!(remainder.is_empty(), "filters validated as N_MELS*N_BINS");
        for (m, row) in mels.iter_mut().zip(rows) {
            let energy: f32 = row
                .iter()
                .zip(&self.scratch_out)
                .map(|(f, c)| f * (c.re * c.re + c.im * c.im))
                .sum();
            *m = (energy + GUARD).ln();
        }
        // normalize: NA. There is deliberately no CMVN here — see the module docs.
        mels
    }
}

/// `np.pad(x, n, mode="reflect")` on the left edge: mirrors about `x[0]` without
/// repeating it.
///
/// Reflect needs `n + 1` samples to mirror. `push` will not call this with fewer
/// except on a `last` that ends the utterance early, so the degraded zero-fill is
/// reachable only for a recording shorter than 257 samples (16 ms), where there is
/// no correct answer to give.
fn reflect_pad_front(x: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for k in 0..n.min(x.len().saturating_sub(1)) {
        out[n - 1 - k] = x[k + 1];
    }
    out
}

fn reflect_pad_back(x: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    let len = x.len();
    for k in 0..n.min(len.saturating_sub(1)) {
        out[k] = x[len - 2 - k];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden frames, generated by a numpy reference that was itself validated
    /// end-to-end: it decoded six Persian clips through the real ONNX model at a
    /// mean character error rate of 0.022, five of them character-exact. That
    /// provenance is the whole point — a fixture written from this file's own prose
    /// would only prove the prose and the code agree.
    const GOLDEN: &str = include_str!("../assets/golden_mel.json");

    fn golden() -> serde_json::Value {
        serde_json::from_str(GOLDEN).expect("golden fixture parses")
    }

    /// The golden input is broadband noise, NOT `synthetic_audio`'s sine sweep, and
    /// the reason is worth keeping. A sweep puts all its energy in a handful of low
    /// mel bins, leaving the high ones holding nothing but spectral leakage — where
    /// f32 and f64 disagree by up to 0.24 log units. Comparing those bins tests
    /// float noise, not this pipeline. Noise gives every bin real energy, so the
    /// comparison means something across all 80.
    ///
    /// Integer LCG so Rust and the Python oracle produce bit-identical input.
    fn lcg_noise(n: usize) -> Vec<f32> {
        let mut st: u32 = 12345;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                (st >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    fn one_shot(audio: &[f32]) -> Vec<[f32; N_MELS]> {
        Fbank::new().unwrap().push(audio, true)
    }

    fn chunked(audio: &[f32], n: usize) -> Vec<[f32; N_MELS]> {
        let mut fb = Fbank::new().unwrap();
        let mut out = Vec::new();
        let mut it = audio.chunks(n).peekable();
        while let Some(c) = it.next() {
            out.extend(fb.push(c, it.peek().is_none()));
        }
        out
    }

    fn assert_close(a: &[[f32; N_MELS]], b: &[[f32; N_MELS]], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: frame count differs");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            for bin in 0..N_MELS {
                assert!(
                    (x[bin] - y[bin]).abs() < 1e-3,
                    "{what}: frame {i} bin {bin}: {} vs {}",
                    x[bin],
                    y[bin]
                );
            }
        }
    }

    #[test]
    fn matches_the_golden_frames() {
        let g = golden();
        let mels = one_shot(&lcg_noise(48_000));
        assert_eq!(
            mels.len(),
            g["shape"][1].as_u64().unwrap() as usize,
            "frame count changed"
        );

        for (name, idx) in [
            ("frame_0", 0usize),
            ("frame_100", 100),
            ("frame_last", mels.len() - 1),
        ] {
            let want: Vec<f64> = serde_json::from_value(g[name].clone()).unwrap();
            for (bin, w) in want.iter().enumerate() {
                let got = mels[idx][bin] as f64;
                assert!(
                    (got - w).abs() < 1e-3,
                    "{name} bin {bin}: got {got}, want {w}"
                );
            }
        }

        // Three sampled frames can miss drift the whole matrix would show, and the
        // fixture already carries these.
        let n = (mels.len() * N_MELS) as f64;
        let mean = mels.iter().flatten().map(|v| *v as f64).sum::<f64>() / n;
        let min = mels.iter().flatten().fold(f32::MAX, |a, b| a.min(*b)) as f64;
        let max = mels.iter().flatten().fold(f32::MIN, |a, b| a.max(*b)) as f64;
        for (name, got) in [("mean", mean), ("min", min), ("max", max)] {
            let want = g["stats"][name].as_f64().unwrap();
            assert!(
                (got - want).abs() < 1e-3,
                "stats.{name}: got {got}, want {want}"
            );
        }
    }

    /// The `normalize: NA` tripwire. Cheap, and it is the ONE feature-pipeline
    /// mistake measured to actually destroy the transcript (CER 1.000, empty
    /// output). It cannot catch a wrong window or a missing preemphasis — those were
    /// measured at 0.042 and 0.056 against a 0.033 baseline, i.e. nearly harmless —
    /// so do not mistake this for coverage of the whole pipeline. The golden frames
    /// above are what does that.
    #[test]
    fn output_is_not_normalised() {
        let mels = one_shot(&crate::engine::synthetic_audio(3.0));
        let n = (mels.len() * N_MELS) as f32;
        let mean = mels.iter().flatten().sum::<f32>() / n;
        assert!(
            mean < -2.0,
            "global mean {mean} looks normalised; preprocessor says normalize: NA, \
             so per-feature CMVN must NOT be applied"
        );
    }

    /// The bare close. `service.rs` closes a session with `last: true` and no audio,
    /// so an empty final push is a normal event, not an edge case.
    #[test]
    fn empty_and_short_pushes_do_not_panic() {
        // 512 samples of centre-padding around nothing is still one whole frame.
        // The property here is that it does not panic; an empty utterance producing
        // a frame of silence is harmless.
        let mut fb = Fbank::new().unwrap();
        assert!(fb.push(&[], true).len() <= 1);

        let mut fb = Fbank::new().unwrap();
        fb.push(&[0.1; 400], false);
        fb.push(&[], true); // close with nothing left to add

        let mut fb = Fbank::new().unwrap();
        fb.push(&[0.1, -0.2, 0.3], true); // shorter than one reflect pad
    }

    /// Chunking must not change the answer, at ANY chunk size.
    ///
    /// This is not hypothetical. An earlier version built the left reflect-pad from
    /// whatever the first `push` happened to carry, so a first chunk under 256
    /// samples zero-filled the pad instead of mirroring and corrupted frames 0-1 by
    /// up to 3.57 log units — silently. The browser sends 8960 at a time and would
    /// never have shown it; the offline path feeds arbitrary slices and would.
    #[test]
    fn chunking_does_not_change_the_result() {
        let audio = crate::engine::synthetic_audio(2.0);
        let base = one_shot(&audio);
        assert!(base.len() > 100, "sanity: expected a couple hundred frames");

        for n in [
            1, 7, 159, 160, 161, 255, 256, 257, 1000, 8960, 32_000, 99_999,
        ] {
            assert_close(&chunked(&audio, n), &base, &format!("chunk size {n}"));
        }
    }
}
