//! Gate 0 (ADR-0044 Decision 1): does parakeet-rs's CUDA execution provider
//! actually engage on the RTX 2070 SUPER in k8s-worker-01?
//!
//! This binary exists to answer exactly that and nothing else. The gRPC service
//! is deliberately not written yet: an engine swap invalidates the streaming
//! proto, so committing to a proto before this passes would be building on an
//! unverified foundation.
//!
//! WHY A RUNTIME CHECK AND NOT A BUILD-TIME ONE
//! The build-runner LXC is pinned to `server1`, which has no GPU, and the PCI
//! passthrough is exclusive so it cannot be moved to `.165`. There is no point
//! in the build where a GPU is reachable.
//!
//! WHY IT HAS TO BE CHECKED AT ALL
//! parakeet-rs registers the providers like this (execution.rs, 0.3.7):
//!
//!     ExecutionProvider::Cuda => builder.with_execution_providers([
//!         ort::ep::CUDA::default().build(),
//!         CPUExecutionProvider::default().build().error_on_failure(),
//!     ])?
//!
//! `error_on_failure()` is on the CPU provider, not CUDA. If CUDA fails to
//! initialise, ORT falls through to CPU and the model loads, transcribes, and
//! returns correct text — roughly 30x slower, with nothing in the logs that
//! looks like an error. A ready pod quietly running on CPU is the failure this
//! service is most likely to suffer and least likely to notice.

use anyhow::{bail, Context, Result};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT};
use std::{path::PathBuf, process::Command, time::Instant};

/// Whole-GPU used-memory in MiB, via the `nvidia-smi` the container toolkit
/// injects at runtime.
///
/// ponytail: whole-GPU, not per-process. nvidia-smi cannot map our PID through
/// the container's PID namespace, so a per-process query returns nothing useful
/// here. Correct on a single-tenant GPU, which ADR-0011 guarantees this is.
/// Switch to NVML per-process if this GPU is ever shared.
fn gpu_used_mib() -> Result<u64> {
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

/// 16 kHz mono f32, which is what the TDT model expects.
fn read_wav_16k_mono(path: &PathBuf) -> Result<(Vec<f32>, f32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
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
    let seconds = samples.len() as f32 / spec.sample_rate as f32;
    Ok((samples, seconds))
}

/// A 3s synthetic sweep, used when no real audio is supplied.
///
/// The gate's actual assertion is the GPU memory delta, and that is driven by
/// session creation plus a forward pass through the full encoder/decoder graph
/// — which synthetic input exercises exactly as well as speech does. What it
/// cannot do is prove the transcript is *correct*, so the text assertion is
/// skipped in this mode and said so out loud.
///
/// Deliberately not a baked-in speech clip: every candidate meant guessing at a
/// URL or dataset path that can rot, and a gate that fails because a download
/// moved tells you nothing about CUDA.
fn synthetic_audio(seconds: f32, sample_rate: u32) -> Vec<f32> {
    let n = (seconds * sample_rate as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f = 120.0 + 400.0 * (t / seconds); // rising sweep through the speech band
            0.25 * (std::f32::consts::TAU * f * t).sin()
        })
        .collect()
}

fn main() -> Result<()> {
    let model_dir: PathBuf = std::env::var("STT_MODEL_DIR")
        .unwrap_or_else(|_| "/models/tdt".into())
        .into();

    // Optional. `kubectl cp some.wav stt-gate:/tmp/a.wav` and pass /tmp/a.wav to
    // upgrade this from "CUDA engaged" to "CUDA engaged and the model works".
    let audio_arg = std::env::args().nth(1).map(PathBuf::from);

    let baseline = gpu_used_mib()?;
    println!("gpu.used before load : {baseline} MiB");

    // The load-time half of the CUDA decision. Passing None here would silently
    // give a CPU session even with the `cuda` cargo feature compiled in.
    let cfg = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);

    let t_load = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(&model_dir, Some(cfg))
        .with_context(|| format!("loading the TDT model from {}", model_dir.display()))?;
    println!("model loaded in      : {:.1}s", t_load.elapsed().as_secs_f32());

    let (samples, audio_seconds, real_speech) = match &audio_arg {
        Some(p) => {
            let (s, secs) = read_wav_16k_mono(p)?;
            println!("audio                : {}", p.display());
            (s, secs, true)
        }
        None => {
            println!("audio                : synthetic sweep (no file given — transcript check skipped)");
            (synthetic_audio(3.0, 16_000), 3.0, false)
        }
    };

    // Warm up before measuring: the first decode pays lazy CUDA context creation
    // and cuDNN algorithm selection, which would flatter neither number.
    let _ = model
        .transcribe_samples(samples.clone(), 16_000, 1, None)
        .context("warmup decode")?;

    let after_warmup = gpu_used_mib()?;
    let delta = after_warmup.saturating_sub(baseline);

    let t = Instant::now();
    let result = model
        .transcribe_samples(samples, 16_000, 1, None)
        .context("measured decode")?;
    let decode_seconds = t.elapsed().as_secs_f32();
    let rtf = decode_seconds / audio_seconds;

    println!("gpu.used after warmup: {after_warmup} MiB (delta {delta} MiB)");
    println!("audio_seconds        : {audio_seconds:.2}");
    println!("decode_seconds       : {decode_seconds:.2}");
    println!("real-time factor     : {rtf:.3}  (lower is faster; <1 is faster than realtime)");
    println!("transcript           : {:?}", result.text);

    if real_speech && result.text.trim().is_empty() {
        bail!("GATE FAILED: empty transcript from real audio — the model ran but produced nothing");
    }

    // The actual gate. A CPU fallback allocates no GPU memory, so this is the
    // one assertion that distinguishes "working" from "working on the wrong
    // device". The threshold is deliberately not 0: nvidia-smi reports whole-GPU
    // usage, so a few MiB of noise from anything else on the card should not
    // count as success.
    const MIN_DELTA_MIB: u64 = 128;
    if delta < MIN_DELTA_MIB {
        bail!(
            "GATE FAILED: GPU memory grew by only {delta} MiB (< {MIN_DELTA_MIB} MiB) across \
             model load and warmup. CUDA did not engage and ORT fell back to CPU — parakeet-rs \
             does this silently by design. Check that the pod has runtimeClassName: nvidia and \
             an nvidia.com/gpu limit, that the image's CUDA/cuDNN majors match what ort \
             2.0.0-rc.13 links, and that ExecutionProvider::Cuda reached from_pretrained."
        );
    }

    println!("\nGATE PASSED: CUDA engaged ({delta} MiB resident), RTF {rtf:.3}");
    if !real_speech {
        println!("NOTE: transcript not asserted — rerun with a 16kHz mono WAV argument for the full check.");
    }
    Ok(())
}
