//! `ukubi-stt` — GPU speech-to-text for ukubi-cluster. ADR-0044.
//!
//! Two modes, one binary:
//!
//!   ukubi-stt                    serve gRPC on :9090, page + health on :8080
//!   ukubi-stt --selftest [wav]   load the batch model, assert CUDA, transcribe
//!   ukubi-stt --selftest-stream  load BOTH models and report total GPU use
//!
//! `--selftest` is the Gate 0 check kept as a manual tool (ADR-0044 Decision 3).
//! It is not a build gate and cannot be: the build-runner LXC is pinned to
//! `server1`, which has no GPU, and PCI passthrough is exclusive by
//! construction so it cannot be moved to `.165`. There is no point in the build
//! where a GPU is reachable.

mod engine;
mod fbank;
mod persian;
mod service;

use anyhow::{Context, Result};
use service::pb::stt_server::SttServer;
use std::{net::SocketAddr, path::PathBuf};

const GRPC_ADDR: &str = "0.0.0.0:9090";
const HEALTH_ADDR: &str = "0.0.0.0:8080";

fn main() -> Result<()> {
    // ORT names the provider it registered and the one it declined, with the
    // reason, at debug level. Without a subscriber those events are dropped and
    // a CUDA failure is indistinguishable from a CUDA success that used no
    // memory — which is exactly how 2026-08-31 was spent. Verbose by
    // construction; RUST_LOG still overrides.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=debug")),
        )
        .with_writer(std::io::stderr)
        .init();

    let model_dir: PathBuf = std::env::var("STT_MODEL_DIR")
        .unwrap_or_else(|_| "/models/tdt".into())
        .into();

    let stream_dir: PathBuf = std::env::var("STT_STREAM_MODEL_DIR")
        .unwrap_or_else(|_| "/models/nemotron".into())
        .into();

    let fa_dir: PathBuf = std::env::var("STT_FA_MODEL_DIR")
        .unwrap_or_else(|_| "/models/shenava".into())
        .into();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--selftest") => selftest(&model_dir, args.next().map(PathBuf::from)),
        Some("--selftest-stream") => selftest_stream(&model_dir, &stream_dir),
        Some("--selftest-fa") => {
            let wav = args.next().map(PathBuf::from);
            let reference = args.next();
            selftest_fa(&fa_dir, wav, reference)
        }
        Some(other) => anyhow::bail!(
            "unknown argument {other:?} — expected --selftest [wav], --selftest-stream, \
             or --selftest-fa [wav] [reference text]"
        ),
        None => serve(&model_dir, stream_dir, fa_dir),
    }
}

/// Load BOTH models and report what the card is holding.
///
/// This exists because ADR-0046 Decision 2 rests on an estimate: the batch model
/// measures 3367 MiB live and the streaming model is an identically-sized fp32
/// export, so both resident is *about* 6.8GB of an 8GB card. "About" is why the
/// streaming model loads lazily in the server — and this is how to replace the
/// estimate with a number, on the GPU node, without risking the running service.
fn selftest_stream(model_dir: &std::path::Path, stream_dir: &std::path::Path) -> Result<()> {
    let empty = engine::gpu_used_mib()?;
    let _batch = engine::load_and_assert_cuda(model_dir)?;
    let after_batch = engine::gpu_used_mib()?;
    let _stream = engine::load_streaming_and_assert_cuda(stream_dir)?;
    let after_both = engine::gpu_used_mib()?;

    println!("gpu.used empty      : {empty} MiB");
    println!(
        "gpu.used batch only : {after_batch} MiB (+{})",
        after_batch - empty
    );
    println!(
        "gpu.used both       : {after_both} MiB (+{} for streaming)",
        after_both - after_batch
    );
    println!("\nBOTH MODELS RESIDENT. If this printed, the card holds them together.");
    Ok(())
}

/// Persian gate: decode a clip and report what it produced, how fast, and — when a
/// reference transcript is supplied — how wrong.
///
/// Runs on CPU unless `STT_FA_DEVICE=cuda`. That default is not a preference, it is
/// the state of an open question: the model's published "83.9 ms per 1.12 s chunk"
/// is a *tract* measurement of unknown thread count, so it predicts neither ORT
/// provider, and the card already holds 6880 MiB of 8192 with two models resident.
/// This subcommand exists to replace that guess with a number on both devices.
///
/// CER rather than WER: Persian word-level error is dominated by ezāfe and ZWNJ
/// variants, which move for reasons unrelated to whether the pipeline is correct.
fn selftest_fa(
    fa_dir: &std::path::Path,
    audio: Option<PathBuf>,
    reference: Option<String>,
) -> Result<()> {
    let device = match std::env::var("STT_FA_DEVICE").as_deref() {
        Ok("cuda") => persian::Device::Cuda,
        Ok("cpu") | Err(_) => persian::Device::Cpu,
        Ok(other) => anyhow::bail!("STT_FA_DEVICE={other:?} — expected \"cpu\" or \"cuda\""),
    };
    println!("model dir      : {}", fa_dir.display());
    println!("device         : {device:?}");

    let baseline = engine::gpu_used_mib().ok();
    let model = std::sync::Arc::new(std::sync::Mutex::new(persian::PersianModel::load(
        fa_dir, device,
    )?));
    if let (Some(before), Some(after)) = (baseline, engine::gpu_used_mib().ok()) {
        println!("gpu delta      : {} MiB", after.saturating_sub(before));
    }

    let Some(path) = audio else {
        println!("\nLOADED. Pass a 16 kHz mono WAV to decode one.");
        return Ok(());
    };
    let (samples, audio_seconds) = engine::read_wav_16k_mono(&path)?;
    println!("audio          : {} ({audio_seconds:.2}s)", path.display());

    // Fed in 560 ms chunks, which is exactly what the browser sends. Decoding it in
    // one call would exercise a path no client uses.
    let mut stream = persian::PersianStream::new(std::sync::Arc::clone(&model));
    let started = std::time::Instant::now();
    let mut text = String::new();
    let mut steps = 0usize;
    let mut chunks = samples.chunks(8960).peekable();
    while let Some(chunk) = chunks.next() {
        let out = stream.push(chunk, chunks.peek().is_none())?;
        if !out.is_empty() {
            steps += 1;
        }
        text.push_str(&out);
    }
    let decode_seconds = started.elapsed().as_secs_f32();
    let hypothesis = persian::tidy(&text);

    println!("decode_seconds : {decode_seconds:.2}");
    println!("real-time factor: {:.3}", decode_seconds / audio_seconds);
    if steps > 0 {
        // The number that decides CPU vs CUDA. Per model step, not per request:
        // steps are a fixed 121 frames wide, so this is comparable across runs
        // where a per-request RTF is not.
        println!(
            "ms per step    : {:.1}",
            decode_seconds * 1000.0 / steps as f32
        );
    }
    println!("unk tokens     : {}", stream.unks());
    println!("transcript     : {hypothesis}");

    if hypothesis.is_empty() {
        anyhow::bail!(
            "empty transcript. If the audio was real speech this is the signature of a \
             wrong feature pipeline — per-feature CMVN in particular takes this model \
             from 0.033 CER to empty output."
        );
    }
    if stream.unks() > 0 {
        println!(
            "\nWARNING: {} <unk> tokens. Persian decoded through a model that cannot \
             write it looks exactly like this.",
            stream.unks()
        );
    }
    if let Some(want) = reference {
        let cer = persian::cer(&want, &hypothesis);
        println!("reference      : {want}");
        println!("CER            : {cer:.3}");
        // 0.15 is the gate. Above 0.30 the pipeline is wrong; between the two the
        // pipeline is right and the export is worse than advertised, which is a
        // product decision rather than a failure.
        if cer > 0.30 {
            anyhow::bail!(
                "CER {cer:.3} — this is a pipeline fault, not a weak model. Suspects, in \
                 order: log-mel framing, the blank id, prev_token collapse across step \
                 boundaries, cache_last_channel_len threading."
            );
        }
        if cer > 0.15 {
            println!(
                "\nWARNING: CER {cer:.3} is above the 0.15 gate but below 0.30 — the \
                      pipeline looks right and the export looks weak."
            );
        }
    }
    println!("\nSELFTEST PASSED");
    Ok(())
}

/// Load, assert CUDA, transcribe, print, exit. Human-readable on purpose: this
/// is the thing someone runs on the node when they want to know whether the GPU
/// path still works.
fn selftest(model_dir: &std::path::Path, audio: Option<PathBuf>) -> Result<()> {
    let mut model = engine::load_and_assert_cuda(model_dir)?;

    let (samples, audio_seconds, real_speech) = match &audio {
        Some(p) => {
            let (s, secs) = engine::read_wav_16k_mono(p)?;
            println!("audio          : {}", p.display());
            (s, secs, true)
        }
        None => {
            println!("audio          : synthetic sweep — transcript not asserted");
            (engine::synthetic_audio(3.0), 3.0, false)
        }
    };

    let started = std::time::Instant::now();
    let result = {
        use parakeet_rs::Transcriber;
        model
            .transcribe_samples(samples, engine::SAMPLE_RATE, 1, None)
            .context("measured decode")?
    };
    let decode_seconds = started.elapsed().as_secs_f32();

    println!("audio_seconds  : {audio_seconds:.2}");
    println!("decode_seconds : {decode_seconds:.2}");
    println!("real-time factor: {:.3}", decode_seconds / audio_seconds);
    println!("transcript     : {:?}", result.text);

    if real_speech && result.text.trim().is_empty() {
        anyhow::bail!("empty transcript from real audio — the model ran but produced nothing");
    }
    println!("\nSELFTEST PASSED");
    Ok(())
}

/// Blocking `main` on purpose: the model loads and CUDA is asserted *before*
/// any runtime or listener exists. There is no window in which this process is
/// reachable but cannot decode.
fn serve(model_dir: &std::path::Path, stream_dir: PathBuf, fa_dir: PathBuf) -> Result<()> {
    // Read before the expensive part. A credential problem is a config error and
    // should not cost a 2.4GB model load and a CUDA context to discover.
    let auth = service::BearerAuth::from_env()?;

    let model = engine::load_and_assert_cuda(model_dir)?;

    let grpc: SocketAddr = GRPC_ADDR.parse()?;
    let health: SocketAddr = HEALTH_ADDR.parse()?;

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            tokio::spawn(async move {
                if let Err(e) = service::serve_http(health).await {
                    tracing::error!("http listener died: {e}");
                }
            });

            // Shared so a background task can warm the streaming model while the
            // server is already accepting traffic. from_arc rather than new()
            // exists for exactly this.
            let svc = std::sync::Arc::new(service::SttService::new(model, stream_dir, fa_dir));
            let warm = std::sync::Arc::clone(&svc);
            tokio::spawn(async move { warm.warm_streaming().await });

            let inner =
                SttServer::from_arc(svc).max_decoding_message_size(service::MAX_DECODE_BYTES);
            let authed = tonic::service::interceptor::InterceptedService::new(inner, auth);

            // Reflection, so an in-cluster caller can discover the API without
            // being handed a .proto first. Registered UNAUTHENTICATED and that
            // is deliberate rather than an oversight: the IngressRoute matches
            // PathPrefix(`/stt.v1.Stt/`) while reflection answers on
            // `/grpc.reflection.v1.*`, so Traefik 404s it and only the pod
            // network can reach it. The schema is public on GitHub anyway; what
            // it must not do is become a second externally-reachable surface.
            //
            // v1 and v1alpha both: grpcurl and older tooling disagree on which
            // they ask for, and serving one of them is the kind of thing that
            // looks fine until someone else's client fails.
            let reflection_v1 = tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(service::FILE_DESCRIPTOR_SET)
                .build_v1()?;
            let reflection_v1alpha = tonic_reflection::server::Builder::configure()
                .register_encoded_file_descriptor_set(service::FILE_DESCRIPTOR_SET)
                .build_v1alpha()?;

            tracing::info!("gRPC listening on {grpc}");
            tonic::transport::Server::builder()
                .add_service(authed)
                .add_service(reflection_v1)
                .add_service(reflection_v1alpha)
                // Recreate is mandatory for this Deployment (the new pod would
                // request the only nvidia.com/gpu while the old one holds it),
                // so a clean SIGTERM shutdown is what keeps the gap short.
                .serve_with_shutdown(grpc, async {
                    let _ = tokio::signal::ctrl_c().await;
                    tracing::info!("shutting down");
                })
                .await
                .map_err(anyhow::Error::from)
        })
}
