//! The model, the GPU assertion, and the audio plumbing they share.
//!
//! WHY THE ASSERTION EXISTS AND WHY IT CRASHES
//! parakeet-rs registers providers like this (execution.rs, 0.3.7):
//!
//!     ExecutionProvider::Cuda => builder.with_execution_providers([
//!         ort::ep::CUDA::default().build(),
//!         CPUExecutionProvider::default().build().error_on_failure(),
//!     ])?
//!
//! `error_on_failure()` is on the CPU provider, not CUDA. If CUDA fails to
//! initialise, ORT falls through to CPU and the model loads, transcribes, and
//! returns correct text — roughly 30x slower, with nothing in the logs that
//! reads as an error. On 2026-08-31 that is exactly what happened: a 0 MiB GPU
//! delta at a real-time factor of 0.081, which is 12x realtime and looks
//! healthy. Only the memory assertion caught it.
//!
//! ADR-0044 Decision 3. A CrashLoopBackOff is loud and attributable; a ready
//! pod decoding at 30x on CPU is the failure this service is most likely to
//! suffer and least likely to have noticed. Downtime is already accepted
//! (ADR-0044 Context), so crashing costs nothing that was promised.

use anyhow::{bail, Context, Result};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, Transcriber};
use std::{path::Path, process::Command, time::Instant};

/// A CPU fallback allocates no GPU memory. The threshold is deliberately not
/// zero: nvidia-smi reports whole-GPU usage, so a few MiB of noise from
/// anything else on the card must not count as success. A real load is ~3.4 GiB.
pub const MIN_DELTA_MIB: u64 = 128;

pub const SAMPLE_RATE: u32 = 16_000;

/// `ParakeetTDT` must be `Send` for the server to hold it behind a mutex and
/// hand it to `spawn_blocking`. It is — nothing it retains is `Rc`/`RefCell`,
/// `ort::session::Session` is `Send + Sync`, and realfft declares
/// `RealToComplex: Send + Sync`. This was an open question in the design; it is
/// now a compile error if it ever stops being true.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<ParakeetTDT>();
};

/// Whole-GPU used memory in MiB, via the `nvidia-smi` the container toolkit
/// injects at runtime.
///
/// ponytail: whole-GPU, not per-process. nvidia-smi cannot map our PID through
/// the container's PID namespace, so a per-process query returns nothing useful
/// here. Correct on a single-tenant GPU, which ADR-0011 guarantees this is.
/// Switch to NVML per-process if this GPU is ever shared.
pub fn gpu_used_mib() -> Result<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .context("running nvidia-smi — is this pod using runtimeClassName: nvidia?")?;
    if !out.status.success() {
        bail!(
            "nvidia-smi exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let first = s.lines().next().unwrap_or_default().trim();
    first
        .parse::<u64>()
        .with_context(|| format!("parsing nvidia-smi memory.used from {first:?}"))
}

/// A 3s synthetic sweep through the speech band. Used to warm the graph when no
/// real audio is available.
///
/// The assertion is a GPU memory delta, driven by session creation plus a
/// forward pass — which synthetic input exercises exactly as well as speech
/// does. What it cannot do is prove the transcript is *correct*, so callers
/// that care assert that separately.
///
/// Deliberately not a baked-in speech clip: every candidate meant guessing at a
/// URL or dataset path that can rot, and a startup check that fails because a
/// download moved tells you nothing about CUDA.
pub fn synthetic_audio(seconds: f32) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let f = 120.0 + 400.0 * (t / seconds);
            0.25 * (std::f32::consts::TAU * f * t).sin()
        })
        .collect()
}

/// Mono little-endian signed 16-bit PCM to f32 in [-1, 1).
///
/// An odd trailing byte is a truncated frame, which means the caller's framing
/// is wrong. Rejected rather than dropped: silently discarding it produces a
/// transcript that is subtly wrong at the tail and no way to notice.
pub fn pcm_s16le_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!(
            "audio is {} bytes, which is not a whole number of 16-bit samples — \
             expected mono little-endian s16 PCM",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect())
}

/// 16 kHz mono WAV to f32 samples. Only used by `--selftest`.
pub fn read_wav_16k_mono(path: &Path) -> Result<(Vec<f32>, f32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        bail!(
            "expected 16kHz mono, got {} Hz / {} channel(s). Convert first: \
             ffmpeg -i in -ar 16000 -ac 1 out.wav",
            spec.sample_rate,
            spec.channels
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
    Ok((samples, seconds))
}

/// Load the model on the GPU and refuse to return one that is secretly on CPU.
///
/// Returns the loaded model. Errors if CUDA did not engage — the caller is
/// expected to propagate that all the way out of `main`.
pub fn load_and_assert_cuda(model_dir: &Path) -> Result<ParakeetTDT> {
    let baseline = gpu_used_mib()?;
    tracing::info!("gpu.used before load: {baseline} MiB");

    // The load-time half of the CUDA decision. Passing None here would silently
    // give a CPU session even with the `cuda` cargo feature compiled in.
    let cfg = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);

    let t_load = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(model_dir, Some(cfg))
        .with_context(|| format!("loading the TDT model from {}", model_dir.display()))?;
    tracing::info!("model loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // Warm up before measuring: the first decode pays lazy CUDA context
    // creation and cuDNN algorithm selection. It is also what makes the first
    // real request fast instead of paying that cost on a user's latency.
    model
        .transcribe_samples(synthetic_audio(3.0), SAMPLE_RATE, 1, None)
        .context("warmup decode")?;

    let after = gpu_used_mib()?;
    let delta = after.saturating_sub(baseline);
    tracing::info!("gpu.used after warmup: {after} MiB (delta {delta} MiB)");

    if delta < MIN_DELTA_MIB {
        bail!(
            "CUDA did not engage: GPU memory grew by only {delta} MiB (< {MIN_DELTA_MIB} MiB) \
             across model load and warmup. ORT fell back to CPU — parakeet-rs does this \
             silently by design. Read the ort=debug lines above first; they name the provider \
             that was declined. Causes found once already, both in the Dockerfile: \
             libonnxruntime_providers_cuda.so missing from /usr/local/bin (ORT dlopens it next \
             to the binary), and a CUDA major mismatch — ort-sys ships no CUDA 12 build for \
             Linux, so the base image must be CUDA 13. Then check the pod has \
             runtimeClassName: nvidia and an nvidia.com/gpu limit."
        );
    }

    tracing::info!("CUDA engaged ({delta} MiB resident)");
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ponytail: the pure helpers only. Loading a model needs a GPU and 2.4GB of
    // weights, which is what the startup assertion and --selftest are for —
    // mocking them here would test the mock.

    #[test]
    fn synthetic_audio_has_the_right_length_and_range() {
        let s = synthetic_audio(3.0);
        assert_eq!(s.len(), 48_000, "3s at 16kHz");
        assert!(
            s.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
            "samples must stay in range or the model sees clipping, not speech"
        );
        // Not silence: a zero signal would still exercise the graph, but it
        // would make an empty transcript ambiguous.
        assert!(s.iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn pcm_round_trips_and_rejects_a_truncated_frame() {
        // -32768, 0, 32767 little-endian.
        let bytes = [0x00, 0x80, 0x00, 0x00, 0xff, 0x7f];
        let f = pcm_s16le_to_f32(&bytes).unwrap();
        assert_eq!(f.len(), 3);
        assert!((f[0] + 1.0).abs() < 1e-6, "got {}", f[0]);
        assert_eq!(f[1], 0.0);
        assert!((f[2] - 0.999_97).abs() < 1e-4, "got {}", f[2]);

        // An odd byte count means the caller's framing is wrong. Truncating it
        // would corrupt the tail of every transcript with nothing to see.
        assert!(pcm_s16le_to_f32(&[0x00, 0x80, 0x00]).is_err());
    }

    #[test]
    fn wav_reader_rejects_the_wrong_sample_rate() {
        // The model is 16kHz mono. Accepting 44.1k silently would produce a
        // plausible-looking but wrong transcript, and a wrong RTF with it.
        let dir = std::env::temp_dir().join("ukubi_stt_test_44k.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&dir, spec).unwrap();
        for _ in 0..100 {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();

        let err = read_wav_16k_mono(&dir).unwrap_err().to_string();
        assert!(err.contains("16kHz mono"), "got: {err}");
        let _ = std::fs::remove_file(&dir);
    }
}
