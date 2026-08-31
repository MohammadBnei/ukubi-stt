//! `ukubi-stt` — GPU speech-to-text for ukubi-cluster. ADR-0044.
//!
//! Two modes, one binary:
//!
//!   ukubi-stt                  serve gRPC on :9090, health on :8080
//!   ukubi-stt --selftest [wav] load the model, assert CUDA, transcribe, exit
//!
//! `--selftest` is the Gate 0 check kept as a manual tool (ADR-0044 Decision 3).
//! It is not a build gate and cannot be: the build-runner LXC is pinned to
//! `server1`, which has no GPU, and PCI passthrough is exclusive by
//! construction so it cannot be moved to `.165`. There is no point in the build
//! where a GPU is reachable.

mod engine;
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

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--selftest") => selftest(&model_dir, args.next().map(PathBuf::from)),
        Some(other) => anyhow::bail!("unknown argument {other:?} — expected --selftest [wav]"),
        None => serve(&model_dir),
    }
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
fn serve(model_dir: &std::path::Path) -> Result<()> {
    // Read before the expensive part. A missing token is a config error and
    // should not cost a 2.4GB model load and a CUDA context to discover.
    let token = std::env::var("STT_AUTH_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .context(
            "STT_AUTH_TOKEN is unset or empty. It comes from Infisical via common-app-chart's \
             `infisical` block. Refusing to start rather than serving a GPU on a public \
             hostname with no authentication — Certificate Transparency publishes that \
             hostname within minutes of issuance (ADR-0044 Consequences).",
        )?;

    let model = engine::load_and_assert_cuda(model_dir)?;

    let grpc: SocketAddr = GRPC_ADDR.parse()?;
    let health: SocketAddr = HEALTH_ADDR.parse()?;

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            tokio::spawn(async move {
                if let Err(e) = service::serve_health(health).await {
                    tracing::error!("health listener died: {e}");
                }
            });

            let inner = SttServer::new(service::SttService::new(model))
                .max_decoding_message_size(service::MAX_DECODE_BYTES);
            let authed = tonic::service::interceptor::InterceptedService::new(
                inner,
                service::BearerAuth::new(token),
            );

            tracing::info!("gRPC listening on {grpc}");
            tonic::transport::Server::builder()
                .add_service(authed)
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
