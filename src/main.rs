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
// Nothing calls this yet — the ORT session that consumes it lands next, and CI runs
// clippy with -D warnings. Delete this attribute in that change; if it is still here
// once src/persian.rs exists, something did not get wired up.
#[allow(dead_code)]
mod fbank;
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

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--selftest") => selftest(&model_dir, args.next().map(PathBuf::from)),
        Some("--selftest-stream") => selftest_stream(&model_dir, &stream_dir),
        Some(other) => anyhow::bail!(
            "unknown argument {other:?} — expected --selftest [wav] or --selftest-stream"
        ),
        None => serve(&model_dir, stream_dir),
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
fn serve(model_dir: &std::path::Path, stream_dir: PathBuf) -> Result<()> {
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
            let svc = std::sync::Arc::new(service::SttService::new(model, stream_dir));
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
