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

mod config;
mod engine;
mod errkeys;
mod fbank;
mod persian;
mod service;

use anyhow::{Context, Result};
use service::pb::stt_server::SttServer;
use std::{net::SocketAddr, path::PathBuf};
use tracing_subscriber::fmt::writer::BoxMakeWriter;

const GRPC_ADDR: &str = "0.0.0.0:9090";
const HEALTH_ADDR: &str = "0.0.0.0:8080";

fn main() -> Result<()> {
    let config = config::Config::from_process()?;

    // ORT names the provider it registered and the one it declined, with the
    // reason, at debug level. Without a subscriber those events are dropped and
    // a CUDA failure is indistinguishable from a CUDA success that used no
    // memory — which is exactly how 2026-08-31 was spent. Verbose by
    // construction; RUST_LOG still overrides, and an invalid one falls back.
    // Serve logs JSON lines on stdout. A selftest's stdout is its report, so its
    // logs go to stderr (still JSON).
    let filter = config
        .log_filter
        .as_deref()
        .and_then(|s| tracing_subscriber::EnvFilter::try_new(s).ok())
        .unwrap_or_else(|| tracing_subscriber::EnvFilter::new("info,ort=debug"));
    let writer = match config.mode {
        config::Mode::Serve { .. } => BoxMakeWriter::new(std::io::stdout),
        config::Mode::Selftest { .. }
        | config::Mode::SelftestStream
        | config::Mode::SelftestFa { .. } => BoxMakeWriter::new(std::io::stderr),
    };
    // init(), not set_global_default: it also installs the `log` bridge.
    logs(filter, writer).init();

    let config::Config {
        mode,
        model_dir,
        stream_dir,
        fa_dir,
        fa_device,
        max_sessions,
        log_filter: _,
    } = config;
    match mode {
        config::Mode::Selftest { wav } => selftest(&model_dir, wav),
        config::Mode::SelftestStream => selftest_stream(&model_dir, &stream_dir),
        config::Mode::SelftestFa { wav, reference } => {
            selftest_fa(&fa_dir, fa_device, wav, reference)
        }
        config::Mode::Serve { credentials } => {
            serve(&model_dir, stream_dir, fa_dir, credentials, max_sessions)
        }
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
    device: persian::Device,
    audio: Option<PathBuf>,
    reference: Option<String>,
) -> Result<()> {
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

/// One JSON object per line: `timestamp`, `level`, `message` and the event's
/// fields at the top level, then `target` and the enclosing spans. Loki reads
/// `level` as detected_level, which is what log-error-rate-spike counts.
fn logs<W>(
    filter: tracing_subscriber::EnvFilter,
    writer: W,
) -> tracing_subscriber::fmt::SubscriberBuilder<
    tracing_subscriber::fmt::format::JsonFields,
    tracing_subscriber::fmt::format::Format<tracing_subscriber::fmt::format::Json>,
    tracing_subscriber::EnvFilter,
    W,
>
where
    W: for<'w> tracing_subscriber::fmt::MakeWriter<'w> + Send + Sync + 'static,
{
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_env_filter(filter)
        .with_writer(writer)
}

/// Blocking `main` on purpose: the model loads and CUDA is asserted *before*
/// any listener exists. There is no window in which this process is reachable
/// but cannot decode.
fn serve(
    model_dir: &std::path::Path,
    stream_dir: PathBuf,
    fa_dir: PathBuf,
    credentials: config::Credentials,
    max_sessions: usize,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    // Before the load, not after: as PID 1 the kernel drops any signal that has
    // no handler, so a SIGTERM during the multi-second load would otherwise be
    // lost and the pod would sit out the grace period. Registered here, it is
    // held and the server stops as soon as it starts.
    let shutdown = {
        let _rt = rt.enter();
        shutdown_signal()?
    };

    // Credentials were validated in Config::load, before anything expensive: a
    // credential problem should not cost a 2.4GB model load and a CUDA context
    // to discover. Built here so the log line keeps its place.
    let auth = service::BearerAuth::new(credentials);

    let model = engine::load_and_assert_cuda(model_dir)?;

    let grpc: SocketAddr = GRPC_ADDR.parse()?;
    let health: SocketAddr = HEALTH_ADDR.parse()?;

    rt.block_on(async move {
        tokio::spawn(async move {
            if let Err(e) = service::serve_http(health).await {
                tracing::error!("http listener died: {e}");
            }
        });

        // Shared so a background task can warm the streaming model while the
        // server is already accepting traffic. from_arc rather than new()
        // exists for exactly this.
        let svc = std::sync::Arc::new(service::SttService::new(
            model,
            stream_dir,
            fa_dir,
            max_sessions,
        ));
        let warm = std::sync::Arc::clone(&svc);
        tokio::spawn(async move { warm.warm_streaming().await });

        let inner = SttServer::from_arc(svc).max_decoding_message_size(service::MAX_DECODE_BYTES);
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
            // ponytail: the drain is unbounded. Idle connections and unary
            // RPCs finish at once; a live bidi stream holds it open until
            // the client stops or SIGKILL lands. Add a timeout if that bites.
            .serve_with_shutdown(grpc, shutdown)
            .await
            .map_err(anyhow::Error::from)
    })
}

/// Resolves on SIGTERM or SIGINT, then logs which. Both handlers are installed
/// by this call, not on first poll: the binary is PID 1 (exec-form ENTRYPOINT),
/// and the kernel ignores SIGTERM for PID 1 while no handler is installed — so
/// waiting on `ctrl_c()` alone meant every pod stop ran out the grace period and
/// ended in SIGKILL. Call inside a runtime context.
// ponytail: unix only. The binary ships in a Linux image; dev is macOS.
fn shutdown_signal() -> std::io::Result<impl std::future::Future<Output = ()>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;
    Ok(async move {
        let signal = tokio::select! {
            _ = term.recv() => "SIGTERM",
            _ = int.recv() => "SIGINT",
        };
        tracing::info!(signal, "shutting down");
    })
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "tests spawn and configure child processes"
)]
mod tests {
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};

    const CHILD_PORT: &str = "UKUBI_STT_SHUTDOWN_CHILD_PORT";

    /// Kills and reaps the child on every exit path, so a failed assertion
    /// cannot leave an orphan waiting for a signal.
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[derive(Clone, Default)]
    struct Buf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for Buf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The format Loki parses, through the real log sites: one JSON object per
    /// line with its level, the message words unchanged, fields at the top level
    /// and the session span kept on an ERROR raised inside it.
    #[test]
    fn logs_are_json_lines_with_their_level() {
        let buf = Buf::default();
        let sink = buf.clone();
        let subscriber = super::logs(tracing_subscriber::EnvFilter::new("info"), move || {
            sink.clone()
        })
        .finish();
        let credentials =
            match crate::config::Config::load([], [("STT_AUTH_TOKEN".into(), "x".into())])
                .unwrap()
                .mode
            {
                crate::config::Mode::Serve { credentials } => credentials,
                other => panic!("{other:?}"),
            };
        tracing::subscriber::with_default(subscriber, || {
            // Callsite interest is cached process-wide. A parallel test that hit
            // the same WARN with no subscriber cached "never"; recompute it now
            // that this one is registered.
            tracing::callsite::rebuild_interest_cache();
            tracing::info_span!("decode_chunk", session = "s1")
                .in_scope(|| crate::errkeys::internal("what", &"boom"));
            // A name no other test uses: warn_due is process-global.
            crate::errkeys::model_unavailable("log format test model", &"gone");
            crate::service::BearerAuth::new(credentials);
            tracing::debug!("filtered out");
        });
        let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
            .collect();
        for l in &lines {
            assert!(l["timestamp"].is_string(), "{l}");
        }
        let levels: Vec<_> = lines.iter().map(|l| l["level"].as_str().unwrap()).collect();
        // Exactly three: the DEBUG line is gone, so the filter passed in is applied.
        assert_eq!(levels, ["ERROR", "WARN", "INFO"], "{text}");
        assert_eq!(lines[0]["message"], "what: boom");
        assert_eq!(lines[0]["span"]["session"], "s1");
        assert_eq!(
            lines[1]["message"],
            "log format test model failed to load: gone"
        );
        assert_eq!(lines[2]["message"], "bearer credentials loaded");
        assert_eq!(lines[2]["clients"], r#"["default"]"#);
    }

    /// Not a test on its own: the body of the child process that
    /// `signal_stops_a_serving_server` spawns. It is a real tonic server
    /// (reflection only, so no model) wired exactly as `serve()` wires it.
    /// Without the env var it returns at once, so `--include-ignored` is safe.
    #[test]
    #[ignore]
    fn shutdown_child() {
        let Ok(port) = std::env::var(CHILD_PORT) else {
            return;
        };
        let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let shutdown = {
            let _rt = rt.enter();
            super::shutdown_signal().unwrap()
        };
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(crate::service::FILE_DESCRIPTOR_SET)
            .build_v1()
            .unwrap();
        rt.block_on(
            tonic::transport::Server::builder()
                .add_service(reflection)
                .serve_with_shutdown(addr, shutdown),
        )
        .unwrap();
    }

    /// The kernel only ignores a signal for PID 1 while no handler is installed,
    /// so a handler that resolves and lets the server exit is what fixes the pod
    /// stop. An idle h2 connection is held open across the signal: graceful
    /// shutdown has to drain it, and a drain that hangs is the other way to fail.
    #[test]
    fn signal_stops_a_serving_server() {
        // The client side of the idle connection needs a runtime that keeps
        // running (to answer GOAWAY) while this thread polls the child.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        for sig in ["TERM", "INT"] {
            let port = std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port();
            let mut child = Reap(
                Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "tests::shutdown_child", "--ignored"])
                    .env(CHILD_PORT, port.to_string())
                    .spawn()
                    .unwrap(),
            );

            // A served connection means shutdown_signal() already ran: it is
            // called before the server is built.
            let endpoint =
                tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
                    .unwrap();
            let deadline = Instant::now() + Duration::from_secs(20);
            let _conn = loop {
                match rt.block_on(endpoint.connect()) {
                    Ok(c) => break c,
                    Err(e) => {
                        assert!(Instant::now() < deadline, "child never served: {e}");
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            };

            // sh's builtin kill: slim images ship no /usr/bin/kill.
            let kill = format!("kill -{sig} {}", child.0.id());
            assert!(Command::new("sh")
                .args(["-c", &kill])
                .status()
                .unwrap()
                .success());

            let deadline = Instant::now() + Duration::from_secs(10);
            let status = loop {
                if let Some(s) = child.0.try_wait().unwrap() {
                    break s;
                }
                assert!(
                    Instant::now() < deadline,
                    "SIG{sig}: still running after 10s"
                );
                std::thread::sleep(Duration::from_millis(20));
            };
            assert!(status.success(), "SIG{sig}: child ended with {status}");
        }
    }
}
