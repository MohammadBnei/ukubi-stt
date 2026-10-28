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
mod traces;

use anyhow::{Context, Result};
use service::pb::stt_server::SttServer;
use std::{net::SocketAddr, path::PathBuf};
use tracing_subscriber::{
    fmt::writer::BoxMakeWriter, layer::SubscriberExt as _, util::SubscriberInitExt,
};

const GRPC_ADDR: &str = "0.0.0.0:9090";
const HEALTH_ADDR: &str = "0.0.0.0:8080";

fn main() -> Result<()> {
    let config = config::Config::from_process()?;
    start(config, run);
    Ok(())
}

/// Everything after the config is read: the logger, traces when set, `run`,
/// and the flush. The traces test's child calls this too, with its own `run`,
/// so the install and the shutdown it proves are the ones main runs. `run`
/// gets whether traces are on. Exits 1 on its error, after the flush.
fn start(config: config::Config, run: impl FnOnce(config::Config, bool) -> Result<()>) {
    // ORT names the provider it registered and the one it declined, with the
    // reason, at debug level. Without a subscriber those events are dropped and
    // a CUDA failure is indistinguishable from a CUDA success that used no
    // memory — which is exactly how 2026-08-31 was spent. Verbose by
    // construction; RUST_LOG still overrides, and an invalid one falls back.
    // Serve logs JSON lines on stdout. A selftest's stdout is its report, so its
    // logs go to stderr (still JSON).
    const DEFAULT_FILTER: &str = "info,ort=debug";
    let (filter, bad_filter) = match config
        .log_filter
        .as_deref()
        .map(tracing_subscriber::EnvFilter::try_new)
    {
        Some(Ok(f)) => (f, None),
        Some(Err(e)) => (tracing_subscriber::EnvFilter::new(DEFAULT_FILTER), Some(e)),
        None => (tracing_subscriber::EnvFilter::new(DEFAULT_FILTER), None),
    };
    let writer = if logs_to_stdout(&config.mode) {
        BoxMakeWriter::new(std::io::stdout)
    } else {
        BoxMakeWriter::new(std::io::stderr)
    };
    // Before the logger, because the OTel layer is part of it. Its error waits
    // for the logger to be reported.
    let (traces, traces_failed) = match config.traces_endpoint.as_deref().map(traces::Tracing::new)
    {
        Some(Ok(t)) => (Some(t), None),
        Some(Err(e)) => (None, Some(e)),
        None => (None, None),
    };
    // init(), not set_global_default: it also installs the `log` bridge.
    tracing_subscriber::registry()
        .with(logs(filter, writer))
        .with(traces.as_ref().map(traces::Tracing::layer))
        .init();
    if let Some(e) = bad_filter {
        tracing::warn!("RUST_LOG is invalid, using {DEFAULT_FILTER}: {e}");
    }
    for w in &config.warnings {
        tracing::warn!("{w}");
    }
    if let Some(e) = traces_failed {
        tracing::warn!("tracing is off: {e:#}");
    }

    // A fatal error is the line that explains a crash loop, so it is a JSON
    // ERROR like every other line, not anyhow's plain text on stderr.
    let traced = traces.is_some();
    let result = run(config, traced);
    if let Err(e) = &result {
        tracing::error!("{e:#}");
    }
    // After run, so after the server has drained: the span of a call still in
    // flight at SIGTERM is closed by now. exit(1) skips destructors, so the
    // flush has to come before it. ponytail: a panic out of run skips it, and
    // the spans still queued are lost.
    if let Some(t) = traces {
        t.shutdown();
    }
    if result.is_err() {
        std::process::exit(1);
    }
}

fn run(config: config::Config, traced: bool) -> Result<()> {
    let config::Config {
        mode,
        model_dir,
        stream_dir,
        fa_dir,
        fa_device,
        max_sessions,
        log_filter: _,
        traces_endpoint: _,
        warnings: _,
    } = config;
    match mode {
        config::Mode::Selftest { wav } => selftest(&model_dir, wav),
        config::Mode::SelftestStream => selftest_stream(&model_dir, &stream_dir),
        config::Mode::SelftestFa { wav, reference } => {
            selftest_fa(&fa_dir, fa_device, wav, reference)
        }
        config::Mode::Serve { credentials } => serve(
            &model_dir,
            stream_dir,
            fa_dir,
            credentials,
            max_sessions,
            traced,
        ),
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

/// Serve logs on stdout. A selftest's stdout is its report, so its logs go to
/// stderr. Exhaustive so a new mode has to choose.
fn logs_to_stdout(mode: &config::Mode) -> bool {
    match mode {
        config::Mode::Serve { .. } => true,
        config::Mode::Selftest { .. }
        | config::Mode::SelftestStream
        | config::Mode::SelftestFa { .. } => false,
    }
}

/// One JSON object per line: `timestamp`, `level`, `message` and the event's
/// fields at the top level, then `target` and the current span. Loki reads
/// `level` as detected_level, which is what log-error-rate-spike counts.
///
/// The filter is this layer's own, not the subscriber's, so the OTel layer
/// beside it keeps its own; and this layer never sees the gRPC call span, so
/// a line logged inside a traced call is the line logged without traces.
fn logs<S>(
    filter: tracing_subscriber::EnvFilter,
    writer: BoxMakeWriter,
) -> impl tracing_subscriber::Layer<S> + Send + Sync
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    use tracing_subscriber::{filter::FilterExt as _, Layer as _};
    let not_the_call_span =
        tracing_subscriber::filter::filter_fn(|m| m.target() != traces::SPAN_TARGET);
    // ponytail: one span level (decode_chunk), so `span` alone; the list repeats it.
    tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_span_list(false)
        .with_writer(writer)
        .with_filter(filter.and(not_the_call_span))
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
    traced: bool,
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

        serve_grpc(svc, auth, grpc, shutdown, traced).await
    })
}

/// `svc` behind the bearer check, with reflection beside it, until `shutdown`
/// has fired and every call has finished. The traces test serves a stand-in
/// `Stt` through this, so what it proves is this wiring.
async fn serve_grpc<S: service::pb::stt_server::Stt>(
    svc: std::sync::Arc<S>,
    auth: service::BearerAuth,
    grpc: SocketAddr,
    shutdown: impl std::future::Future<Output = ()>,
    traced: bool,
) -> Result<()> {
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
    let routes = tonic::service::Routes::new(authed)
        .add_service(reflection_v1)
        .add_service(reflection_v1alpha);

    tracing::info!("gRPC listening on {grpc}");
    // Recreate is mandatory for this Deployment (the new pod would
    // request the only nvidia.com/gpu while the old one holds it),
    // so a clean SIGTERM shutdown is what keeps the gap short.
    // ponytail: the drain is unbounded. Idle connections and unary
    // RPCs finish at once; a live bidi stream holds it open until
    // the client stops or SIGKILL lands. Add a timeout if that bites.
    //
    // Two arms, not option_layer: the traced response body is another type,
    // and with traces unset the server must have no layer at all. Every
    // tower-http log event is off (its default logs ERROR for any non-OK
    // gRPC status, which log-error-rate-spike would count).
    let mut server = tonic::transport::Server::builder();
    if traced {
        let layer = tower_http::trace::TraceLayer::new_for_grpc()
            .make_span_with(traces::make_span)
            .on_request(())
            .on_response(traces::on_response)
            .on_body_chunk(())
            .on_eos(traces::on_eos)
            .on_failure(());
        server
            .layer(layer)
            .add_routes(routes)
            .serve_with_shutdown(grpc, shutdown)
            .await?;
    } else {
        server
            .add_routes(routes)
            .serve_with_shutdown(grpc, shutdown)
            .await?;
    }
    Ok(())
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

    /// README: "Serve writes them to stdout; the `--selftest*` modes write them
    /// to stderr" — a selftest's stdout is its report.
    #[test]
    fn only_serve_logs_to_stdout() {
        for (args, stdout) in [
            (&[][..], true),
            (&["--selftest"], false),
            (&["--selftest-stream"], false),
            (&["--selftest-fa"], false),
        ] {
            let mode = crate::config::Config::load(
                args.iter().map(Into::into),
                [("STT_AUTH_TOKEN".into(), "x".into())],
            )
            .unwrap()
            .mode;
            assert_eq!(super::logs_to_stdout(&mode), stdout, "{args:?}");
        }
    }

    /// The format Loki parses, through the real log sites: one JSON object per
    /// line with its level, the message words unchanged, fields at the top level
    /// and the session span kept on an ERROR raised inside it.
    #[test]
    fn logs_are_json_lines_with_their_level() {
        let buf = Buf::default();
        let sink = buf.clone();
        let subscriber = tracing_subscriber::layer::SubscriberExt::with(
            tracing_subscriber::registry(),
            super::logs(
                tracing_subscriber::EnvFilter::new("info"),
                tracing_subscriber::fmt::writer::BoxMakeWriter::new(move || sink.clone()),
            ),
        );
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
        assert!(lines[0].get("spans").is_none(), "{text}");
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

    const TRACES_CHILD: &str = "UKUBI_STT_TRACES_CHILD";
    const CALLER_TRACE: &str = "0af7651916cd43dd8448eb211c80319c";
    const CALLER_SPAN: &str = "b7ad6b7169203331";

    /// A stand-in for SttService, so the child needs no model. It logs inside
    /// the call (once outside any span of ours, once inside a decode_chunk
    /// span) so the fmt and OTel filters both have something to drop, and
    /// takes its behaviour from `session_id`.
    struct StandIn;

    #[tonic::async_trait]
    impl crate::service::pb::stt_server::Stt for StandIn {
        async fn recognize(
            &self,
            request: tonic::Request<crate::service::pb::RecognizeRequest>,
        ) -> Result<tonic::Response<crate::service::pb::RecognizeResponse>, tonic::Status> {
            tracing::info!("stand-in recognize");
            tracing::info_span!("decode_chunk", session = "s")
                .in_scope(|| tracing::info!("stand-in chunk"));
            match request.get_ref().session_id.as_str() {
                "internal" => return Err(tonic::Status::internal("stand-in fault")),
                "slow" => {
                    tracing::info!("stand-in slow call started");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                _ => {}
            }
            Ok(tonic::Response::new(Default::default()))
        }
    }

    /// Not a test on its own: the child that `traces_reach_a_collector_and_unset_is_inert`
    /// spawns. It runs main's `start` (logger, traces, flush) around the real
    /// `serve_grpc` and `serve_http`, with the stand-in for the model.
    #[test]
    #[ignore]
    fn traces_child() {
        let Ok(ports) = std::env::var(TRACES_CHILD) else {
            return;
        };
        let (grpc, health) = ports.split_once(',').unwrap();
        let grpc: std::net::SocketAddr = format!("127.0.0.1:{grpc}").parse().unwrap();
        let health: std::net::SocketAddr = format!("127.0.0.1:{health}").parse().unwrap();
        let config = crate::config::Config::load([], std::env::vars_os()).unwrap();
        super::start(config, |config, traced| {
            let crate::config::Mode::Serve { credentials } = config.mode else {
                panic!("not serve");
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let shutdown = {
                let _rt = rt.enter();
                super::shutdown_signal()?
            };
            let auth = crate::service::BearerAuth::new(credentials);
            rt.block_on(async move {
                tokio::spawn(crate::service::serve_http(health));
                super::serve_grpc(std::sync::Arc::new(StandIn), auth, grpc, shutdown, traced).await
            })
        });
    }

    type Got = std::sync::Arc<
        std::sync::Mutex<
            Vec<opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest>,
        >,
    >;

    /// An OTLP/HTTP collector: POST /v1/traces, protobuf in, empty answer out.
    async fn collector(got: Got) -> std::net::SocketAddr {
        let app = axum::Router::new().route(
            "/v1/traces",
            axum::routing::post(move |body: axum::body::Bytes| {
                let got = got.clone();
                async move {
                    got.lock()
                        .unwrap()
                        .push(prost::Message::decode(body).unwrap());
                    (
                        [(axum::http::header::CONTENT_TYPE, "application/x-protobuf")],
                        Vec::<u8>::new(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    }

    /// One unary call; the code it ended with.
    async fn unary(
        channel: tonic::transport::Channel,
        path: &'static str,
        token: &str,
        session: &str,
        traceparent: Option<&str>,
    ) -> tonic::Code {
        let mut grpc = tonic::client::Grpc::new(channel);
        grpc.ready().await.unwrap();
        let mut request = tonic::Request::new(crate::service::pb::RecognizeRequest {
            session_id: session.into(),
            ..Default::default()
        });
        let md = request.metadata_mut();
        md.insert("authorization", format!("Bearer {token}").parse().unwrap());
        if let Some(tp) = traceparent {
            md.insert("traceparent", tp.parse().unwrap());
        }
        let codec = tonic_prost::ProstCodec::<
            crate::service::pb::RecognizeRequest,
            crate::service::pb::RecognizeResponse,
        >::default();
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(path);
        match grpc.unary(request, path, codec).await {
            Ok(_) => tonic::Code::Ok,
            Err(s) => s.code(),
        }
    }

    /// One reflection `list`, read to its end so the stream's trailers arrive.
    async fn reflection_list(channel: tonic::transport::Channel) {
        use tonic_reflection::pb::v1::{
            server_reflection_request::MessageRequest, ServerReflectionRequest,
            ServerReflectionResponse,
        };
        let mut grpc = tonic::client::Grpc::new(channel);
        grpc.ready().await.unwrap();
        let request = tonic::Request::new(tonic::codegen::tokio_stream::once(
            ServerReflectionRequest {
                host: String::new(),
                message_request: Some(MessageRequest::ListServices(String::new())),
            },
        ));
        let codec =
            tonic_prost::ProstCodec::<ServerReflectionRequest, ServerReflectionResponse>::default();
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
        );
        let mut answers = grpc
            .streaming(request, path, codec)
            .await
            .unwrap()
            .into_inner();
        let mut n = 0;
        while answers.message().await.unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 1);
    }

    fn http_get(port: u16, path: &str) -> String {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            s,
            "GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    /// Runs one child with `env`, makes the same calls to it, and returns its
    /// log lines. A slow call is in flight when SIGTERM lands; nothing may
    /// reach the collector before that (the batch delay is an hour, so only
    /// the shutdown flush exports).
    fn serve_child(rt: &tokio::runtime::Runtime, got: &Got, env: &[(&str, String)]) -> Vec<String> {
        let port = || {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        };
        let (grpc, health) = (port(), port());
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "tests::traces_child", "--ignored"]);
        // The shell's own OTEL_* settings must not reach the child.
        for (k, _) in std::env::vars_os() {
            if k.to_string_lossy().starts_with("OTEL_") {
                cmd.env_remove(k);
            }
        }
        cmd.env(TRACES_CHILD, format!("{grpc},{health}"))
            .env("STT_AUTH_TOKEN", "good")
            .env("OTEL_BSP_SCHEDULE_DELAY", "3600000")
            .envs(env.iter().map(|(k, v)| (k, v)))
            .stdout(std::process::Stdio::piped());
        let mut child = Reap(cmd.spawn().unwrap());
        let stdout = child.0.stdout.take().unwrap();
        let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let reader = {
            let lines = lines.clone();
            std::thread::spawn(move || {
                use std::io::BufRead;
                for l in std::io::BufReader::new(stdout).lines() {
                    lines.lock().unwrap().push(l.unwrap());
                }
            })
        };

        let endpoint =
            tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{grpc}")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let channel = loop {
            match rt.block_on(endpoint.connect()) {
                Ok(c) => break c,
                Err(e) => {
                    assert!(Instant::now() < deadline, "child never served: {e}");
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        };
        let traceparent = format!("00-{CALLER_TRACE}-{CALLER_SPAN}-01");
        let recognize = "/stt.v1.Stt/Recognize";
        let c = || channel.clone();
        rt.block_on(async {
            assert_eq!(
                unary(c(), recognize, "good", "ok", Some(&traceparent)).await,
                tonic::Code::Ok
            );
            assert_eq!(
                unary(c(), recognize, "bad", "ok", None).await,
                tonic::Code::Unauthenticated
            );
            assert_eq!(
                unary(c(), recognize, "good", "internal", None).await,
                tonic::Code::Internal
            );
            assert_eq!(
                unary(c(), "/stt.v1.Stt/Nope", "good", "ok", None).await,
                tonic::Code::Unimplemented
            );
            reflection_list(c()).await;
        });
        for path in ["/healthz", "/livez"] {
            let answer = http_get(health, path);
            assert!(answer.starts_with("HTTP/1.1 200"), "{path}: {answer}");
        }

        let slow = rt.spawn(unary(c(), recognize, "good", "slow", None));
        let deadline = Instant::now() + Duration::from_secs(20);
        while !lines
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.contains("stand-in slow call started"))
        {
            assert!(Instant::now() < deadline, "the slow call never started");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            got.lock().unwrap().is_empty(),
            "exported before shutdown: the flush is not what is being proved"
        );
        let kill = format!("kill -TERM {}", child.0.id());
        assert!(Command::new("sh")
            .args(["-c", &kill])
            .status()
            .unwrap()
            .success());
        assert_eq!(rt.block_on(slow).unwrap(), tonic::Code::Ok, "drained");

        let deadline = Instant::now() + Duration::from_secs(30);
        let status = loop {
            if let Some(s) = child.0.try_wait().unwrap() {
                break s;
            }
            assert!(Instant::now() < deadline, "still running after SIGTERM");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert!(status.success(), "child ended with {status}");
        reader.join().unwrap();
        let lines = lines.lock().unwrap().clone();
        // The test harness's own lines are not logs.
        lines.into_iter().filter(|l| l.starts_with('{')).collect()
    }

    /// A log line with its timestamp dropped and its ports blanked, so two
    /// children's lines compare.
    fn masked(line: &str) -> String {
        let mut v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
        v.as_object_mut().unwrap().remove("timestamp");
        let text = v.to_string();
        let mut parts = text.split("127.0.0.1:");
        let mut out = parts.next().unwrap().to_string();
        for p in parts {
            out.push_str("127.0.0.1:<port>");
            out.push_str(p.trim_start_matches(|c: char| c.is_ascii_digit()));
        }
        out
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Standard §7, through main's own `start` and the real `serve_grpc` and
    /// `serve_http`: with the variable set, one server span per gRPC call and
    /// none for an unknown method or the probes; each span a new root linked
    /// to the caller; exported by the shutdown flush, after a call in flight
    /// at SIGTERM has finished; log lines the same as with it unset. With only
    /// OTEL_EXPORTER_OTLP_ENDPOINT set, nothing is exported at all (the traced
    /// child, against the same collector, is that check's positive control).
    #[test]
    fn traces_reach_a_collector_and_unset_is_inert() {
        use opentelemetry_proto::tonic::common::v1::any_value::Value;
        use opentelemetry_proto::tonic::trace::v1::{span::SpanKind, status::StatusCode};
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let got = Got::default();
        let addr = rt.block_on(collector(got.clone()));

        let traced = serve_child(
            &rt,
            &got,
            &[(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                format!("http://{addr}/v1/traces"),
            )],
        );
        let exports = std::mem::take(&mut *got.lock().unwrap());
        let inert = serve_child(
            &rt,
            &got,
            &[("OTEL_EXPORTER_OTLP_ENDPOINT", format!("http://{addr}"))],
        );
        assert!(got.lock().unwrap().is_empty(), "unset is not inert");

        // Logs: the same lines with traces on and off. No tower-http or OTel
        // line, no ERROR (log-error-rate-spike), and no `span` key on a line
        // logged inside the call but outside decode_chunk.
        for lines in [&traced, &inert] {
            assert!(
                lines.iter().any(|l| l.contains("shutting down")),
                "{lines:?}"
            );
            for l in lines.iter() {
                let v: serde_json::Value = serde_json::from_str(l).unwrap();
                let target = v["target"].as_str().unwrap();
                assert!(target.starts_with("ukubi_stt"), "{l}");
                assert_ne!(v["level"], "ERROR", "{l}");
                if v["message"] == "stand-in recognize" {
                    assert!(v.get("span").is_none(), "{l}");
                }
                if v["message"] == "stand-in chunk" {
                    assert_eq!(v["span"]["session"], "s", "{l}");
                }
            }
        }
        let sorted = |lines: &[String]| {
            let mut m: Vec<String> = lines.iter().map(|l| masked(l)).collect();
            m.sort();
            m
        };
        assert_eq!(sorted(&traced), sorted(&inert), "log lines changed");

        // Spans.
        let mut spans = Vec::new();
        for export in &exports {
            for rs in &export.resource_spans {
                let resource = rs.resource.as_ref().unwrap();
                let attrs: Vec<_> = resource
                    .attributes
                    .iter()
                    .map(|a| {
                        (
                            a.key.as_str(),
                            a.value.as_ref().and_then(|v| v.value.clone()),
                        )
                    })
                    .collect();
                assert_eq!(
                    attrs,
                    [("service.name", Some(Value::StringValue("ukubi-stt".into())))]
                );
                for ss in &rs.scope_spans {
                    spans.extend(ss.spans.iter());
                }
            }
        }
        let mut names: Vec<&str> = spans.iter().map(|s| s.name.as_str()).collect();
        names.sort();
        assert_eq!(
            names,
            [
                "grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
                "stt.v1.Stt/Recognize",
                "stt.v1.Stt/Recognize",
                "stt.v1.Stt/Recognize",
                "stt.v1.Stt/Recognize",
            ],
            "one span per call; none for the unknown method, the probes or decode_chunk"
        );
        let attr = |s: &opentelemetry_proto::tonic::trace::v1::Span, key: &str| {
            s.attributes
                .iter()
                .find(|a| a.key == key)
                .and_then(|a| a.value.as_ref()?.value.clone())
        };
        let mut trace_ids = std::collections::BTreeSet::new();
        let mut seen = Vec::new();
        for s in &spans {
            assert_eq!(s.kind, SpanKind::Server as i32, "{s:?}");
            assert!(s.parent_span_id.is_empty(), "a root: {s:?}");
            assert_ne!(s.trace_id, hex(CALLER_TRACE), "a new trace: {s:?}");
            assert!(s.events.is_empty(), "no events: {s:?}");
            trace_ids.insert(s.trace_id.clone());
            let mut keys: Vec<&str> = s.attributes.iter().map(|a| a.key.as_str()).collect();
            keys.sort();
            assert_eq!(
                keys,
                [
                    "rpc.grpc.status_code",
                    "rpc.method",
                    "rpc.service",
                    "rpc.system"
                ],
                "{s:?}"
            );
            assert_eq!(
                attr(s, "rpc.system"),
                Some(Value::StringValue("grpc".into()))
            );
            let (service, method) = s.name.split_once('/').unwrap();
            assert_eq!(
                attr(s, "rpc.service"),
                Some(Value::StringValue(service.into()))
            );
            assert_eq!(
                attr(s, "rpc.method"),
                Some(Value::StringValue(method.into()))
            );
            let Some(Value::IntValue(code)) = attr(s, "rpc.grpc.status_code") else {
                panic!("{s:?}");
            };
            let status = s.status.as_ref().map_or(0, |st| st.code);
            for l in &s.links {
                assert_eq!(
                    (l.trace_id.clone(), l.span_id.clone()),
                    (hex(CALLER_TRACE), hex(CALLER_SPAN))
                );
            }
            seen.push((method, code, status, s.links.len()));
        }
        assert_eq!(trace_ids.len(), spans.len(), "every call its own trace");
        seen.sort();
        let (unset, error) = (StatusCode::Unset as i32, StatusCode::Error as i32);
        assert_eq!(
            seen,
            [
                ("Recognize", 0, unset, 0),  // the slow call, in flight at SIGTERM
                ("Recognize", 0, unset, 1),  // the caller's traceparent, linked
                ("Recognize", 13, error, 0), // INTERNAL is the server's fault
                ("Recognize", 16, unset, 0), // UNAUTHENTICATED is the caller's
                ("ServerReflectionInfo", 0, unset, 0),
            ]
        );
    }
}
