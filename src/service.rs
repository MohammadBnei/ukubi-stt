//! The gRPC surface: one unary RPC, a bearer-token interceptor, and an
//! unauthenticated health listener on a separate port.

use crate::engine::{
    is_multilingual, load_streaming_and_assert_cuda, pcm_s16le_to_f32, SAMPLE_RATE,
};
use parakeet_rs::{Nemotron, NemotronHandle, ParakeetTDT, Transcriber};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};
use subtle::ConstantTimeEq;
use tonic::{service::Interceptor, Request, Response, Status};

pub mod pb {
    tonic::include_proto!("stt.v1");
}

/// The compiled proto, served over gRPC reflection. See build.rs.
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("stt_descriptor");

/// Which credential a request arrived on. Attached by the interceptor and read
/// back for logging, so "who is calling this" is answerable without guessing
/// from traffic shape.
#[derive(Clone, Debug)]
pub struct ClientName(pub String);

/// ~8 minutes of 16 kHz mono s16. tonic's default is 4 MB, which is ~2 minutes
/// — short enough that a normal batch upload fails with a confusing error.
/// This is also a defence, not just a limit: ADR-0044's grey-cloud hostname has
/// no Cloudflare WAF in front of it, so the body cap is ours to set.
pub const MAX_DECODE_BYTES: usize = 16 * 1024 * 1024;

/// A cap on concurrent streaming sessions. The binding constraint is not the
/// ~7.5MB of decoder state each one holds — it is the GPU, which each stream
/// occupies for 20-50ms out of every 560ms. Eight streams is roughly half the
/// card's duty cycle, with the batch path still needing room.
const MAX_SESSIONS: usize = 8;

/// Sessions idle longer than this are swept. Browsers close tabs without
/// sending `last: true`, and that is the normal case rather than the edge — an
/// unbounded map keyed by a client-supplied string is a memory leak with an
/// attacker-chosen key. ADR-0046 Decision 5.
const SESSION_IDLE: Duration = Duration::from_secs(120);

struct Session {
    recognizer: Arc<Mutex<Nemotron>>,
    last_used: Instant,
}

/// Recover rather than propagate a poisoned mutex.
///
/// A decode that panicked poisons its lock. Refusing to touch it afterwards
/// means one bad request bricks the service until someone notices a pod that is
/// Ready and answers every call with the same panic.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("mutex was poisoned by a previous panic; recovering");
        poisoned.into_inner()
    })
}

/// Two recognizers behind one RPC, with deliberately different concurrency.
///
/// **Offline** (`session_id` empty) keeps one decode at a time behind a
/// `Semaphore(1)`. One 8GB GPU running one model is honest about what it can do,
/// and a batch caller submitting eight minutes of audio *should* be serialised.
///
/// **Streaming** (`session_id` set) must not use that permit, and ADR-0044
/// Decision 4 predicted why: holding a single permit for a stream's lifetime
/// lets one browser tab starve every batch caller, and the rate limiter cannot
/// help because the connection is already established. Instead each session
/// gets its own recognizer over a shared ONNX session, and the model's internal
/// lock is held only during inference — 20-50ms per 560ms chunk.
pub struct SttService {
    model: Arc<Mutex<ParakeetTDT>>,
    permits: Arc<tokio::sync::Semaphore>,
    stream_dir: PathBuf,
    /// `None` until the first streaming request. See
    /// [`crate::engine::load_streaming_and_assert_cuda`] for why this is lazy.
    /// A `tokio` mutex rather than a `std` one because it is held across the
    /// load `await` — which also means concurrent first-requests wait on one
    /// load instead of racing to start several.
    stream_handle: Arc<tokio::sync::Mutex<Option<NemotronHandle>>>,
    sessions: Arc<Mutex<HashMap<String, Session>>>,
}

impl SttService {
    pub fn new(model: ParakeetTDT, stream_dir: PathBuf) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            permits: Arc::new(tokio::sync::Semaphore::new(1)),
            stream_dir,
            stream_handle: Arc::new(tokio::sync::Mutex::new(None)),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Load the streaming model now, in the background, so the first streaming
    /// request does not pay for it.
    ///
    /// ADR-0046 Decision 2 made this lazy because "both models resident is
    /// ~6.8GB of an 8GB card" was an ESTIMATE, and under `strategy: Recreate` a
    /// pod that cannot start is an outage of the working batch service. It has
    /// since been measured at 6880 MiB, so the unknown the hedge protected
    /// against is gone.
    ///
    /// Warming in a background task rather than before `serve()` keeps the good
    /// half of the hedge: the listener is already up, so a model that fails to
    /// load costs streaming requests a FAILED_PRECONDITION and costs batch
    /// nothing at all. The cliff it removes is real — a cold pod made the first
    /// chunk take ~5s, and because chunks arrive faster than that backlog
    /// drains, the first several seconds of a session ran badly behind.
    pub async fn warm_streaming(&self) {
        match self.handle().await {
            Ok(_) => tracing::info!("streaming model warm"),
            Err(e) => tracing::warn!(
                "streaming model failed to warm ({e}); batch is unaffected and \
                 streaming requests will retry the load"
            ),
        }
    }

    /// The shared streaming model, loaded on first use.
    async fn handle(&self) -> Result<NemotronHandle, Status> {
        let mut guard = self.stream_handle.lock().await;
        if let Some(h) = guard.as_ref() {
            return Ok(h.clone());
        }
        let dir = self.stream_dir.clone();
        let handle = tokio::task::spawn_blocking(move || load_streaming_and_assert_cuda(&dir))
            .await
            .map_err(|e| Status::internal(format!("streaming model load task failed: {e}")))?
            // failed_precondition, not internal: either the model is missing or
            // the card could not hold both, and neither is fixed by retrying.
            .map_err(|e| Status::failed_precondition(format!("{e:#}")))?;
        *guard = Some(handle.clone());
        Ok(handle)
    }

    /// Find or create the recognizer for a session, sweeping idle ones first.
    ///
    /// The sweep can drop a session that a request is still decoding on. That is
    /// safe — the `Arc` keeps the recognizer alive for the in-flight call — and
    /// only means the next chunk starts fresh. Preferable to holding the map
    /// lock across a decode, which would serialise every stream against every
    /// other one.
    fn session(
        &self,
        id: &str,
        handle: &NemotronHandle,
        language: &str,
    ) -> Result<Arc<Mutex<Nemotron>>, Status> {
        let mut map = lock(&self.sessions);
        let now = Instant::now();
        let before = map.len();
        map.retain(|_, s| now.duration_since(s.last_used) < SESSION_IDLE);
        if map.len() != before {
            tracing::info!("swept {} idle streaming session(s)", before - map.len());
        }

        if let Some(existing) = map.get_mut(id) {
            existing.last_used = now;
            return Ok(Arc::clone(&existing.recognizer));
        }
        if map.len() >= MAX_SESSIONS {
            return Err(Status::resource_exhausted(format!(
                "{MAX_SESSIONS} streaming sessions already active — there is one GPU. Retry, or \
                 send `last: true` on sessions you have finished with."
            )));
        }

        let mut recognizer = Nemotron::from_shared(handle);
        // Naming the language is strictly more accurate than letting the model
        // guess, but only the multilingual export can be told — on the
        // English-only build the call is a no-op, so it is gated rather than
        // attempted and swallowed.
        if is_multilingual(handle) && !language.is_empty() && language != "auto" {
            recognizer.set_target_lang(language).map_err(|e| {
                Status::invalid_argument(format!(
                    "language {language:?} is not one this model knows: {e}"
                ))
            })?;
        }

        let recognizer = Arc::new(Mutex::new(recognizer));
        map.insert(
            id.to_string(),
            Session {
                recognizer: Arc::clone(&recognizer),
                last_used: now,
            },
        );
        tracing::info!(
            session = id,
            sessions = map.len(),
            "streaming session opened"
        );
        Ok(recognizer)
    }

    /// Whole-utterance decode against the batch model, one at a time.
    async fn decode_offline(&self, samples: Vec<f32>) -> Result<String, Status> {
        // Acquire BEFORE spawning: the point is to reject the caller, not to
        // park a blocking thread waiting for the GPU.
        //
        // ponytail: try_acquire and reject, never a queue. A queue on a single
        // GPU converts overload into unbounded latency, which is harder to
        // diagnose than a clean RESOURCE_EXHAUSTED.
        let _permit = self.permits.clone().try_acquire_owned().map_err(|_| {
            Status::resource_exhausted(
                "the GPU is busy with another batch decode — one at a time, by design. Retry.",
            )
        })?;

        let model = Arc::clone(&self.model);
        let result = tokio::task::spawn_blocking(move || {
            lock(&model).transcribe_samples(samples, SAMPLE_RATE, 1, None)
        })
        .await
        .map_err(|e| Status::internal(format!("decode task failed: {e}")))?
        .map_err(|e| Status::internal(format!("transcription failed: {e}")))?;
        Ok(result.text)
    }

    /// One chunk of a continuing stream. Returns only the NEW text — this model
    /// does not revise past output, which is why the response carries no
    /// `is_final` and clients simply concatenate.
    async fn decode_chunk(
        &self,
        session_id: &str,
        language: &str,
        last: bool,
        mut samples: Vec<f32>,
    ) -> Result<String, Status> {
        let handle = self.handle().await?;
        let recognizer = self.session(session_id, &handle, language)?;

        // FLUSH THE TAIL. The encoder emits only on a COMPLETE chunk, so a
        // final partial one is buffered and never decoded — measured live on
        // 2026-08-31, where a 9.23s utterance ended "...on an NVIDIA G" and lost
        // its last 270ms. Every utterance would lose its ending.
        //
        // Padding with silence to the next chunk boundary makes the buffered
        // audio a whole chunk, and the extra full chunk after it pushes the
        // model's right-context window past the real speech. Silence decodes to
        // nothing, so the cost is one ~25ms decode and no spurious text.
        //
        // Done here rather than in the client because `last` already means
        // exactly this, and a client that forgot would lose words with no
        // symptom other than a slightly short transcript.
        if last {
            let chunk = handle.chunk_samples();
            let remainder = samples.len() % chunk;
            if remainder != 0 {
                samples.resize(samples.len() + (chunk - remainder), 0.0);
            }
            samples.resize(samples.len() + chunk, 0.0);
        }

        let text =
            tokio::task::spawn_blocking(move || lock(&recognizer).transcribe_chunk(&samples))
                .await
                .map_err(|e| Status::internal(format!("decode task failed: {e}")))?
                .map_err(|e| Status::internal(format!("streaming transcription failed: {e}")))?;

        if last {
            let mut map = lock(&self.sessions);
            map.remove(session_id);
            tracing::info!(
                session = session_id,
                sessions = map.len(),
                "streaming session closed"
            );
        }
        Ok(text)
    }
}

#[tonic::async_trait]
impl pb::stt_server::Stt for SttService {
    async fn recognize(
        &self,
        request: Request<pb::RecognizeRequest>,
    ) -> Result<Response<pb::RecognizeResponse>, Status> {
        let client = request
            .extensions()
            .get::<ClientName>()
            .map(|c| c.0.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let req = request.into_inner();
        let cfg = req.config.unwrap_or_default();

        // 0 means "unset" in proto3 and is treated as "the only rate we
        // support". Any other value is rejected rather than resampled: a
        // silently resampled request returns a plausible transcript and a
        // meaningless real-time factor, and the RTF is how a caller detects
        // that this service has fallen back to CPU.
        if cfg.sample_rate_hertz != 0 && cfg.sample_rate_hertz != SAMPLE_RATE as i32 {
            return Err(Status::invalid_argument(format!(
                "sample_rate_hertz must be {SAMPLE_RATE} (or 0 for the default); got {}. \
                 Resample before sending: ffmpeg -i in -ar 16000 -ac 1 out.wav",
                cfg.sample_rate_hertz
            )));
        }
        // Empty audio is an error EXCEPT as a bare close. `last` means "flush
        // and release", and a client that stops recording on an exact chunk
        // boundary — 1 callback in 35, plus every Stop before speaking — has
        // nothing left to send but still needs the session flushed and closed.
        // Rejecting it leaked the recognizer until the idle sweep and dropped
        // the tail, which is the bug the padding below exists to prevent.
        if req.audio.is_empty() && !req.last {
            return Err(Status::invalid_argument("audio is empty"));
        }

        let samples =
            pcm_s16le_to_f32(&req.audio).map_err(|e| Status::invalid_argument(e.to_string()))?;
        let audio_seconds = samples.len() as f32 / SAMPLE_RATE as f32;

        let started = Instant::now();
        let text = if req.session_id.is_empty() {
            self.decode_offline(samples).await?
        } else {
            self.decode_chunk(&req.session_id, &cfg.language, req.last, samples)
                .await?
        };
        let decode_seconds = started.elapsed().as_secs_f32();

        tracing::info!(
            audio_seconds,
            decode_seconds,
            // A bare close carries no audio, and `x / 0.0` logs as `inf`.
            rtf = if audio_seconds > 0.0 {
                decode_seconds / audio_seconds
            } else {
                0.0
            },
            chars = text.len(),
            streaming = !req.session_id.is_empty(),
            client,
            "recognize"
        );

        Ok(Response::new(pb::RecognizeResponse {
            text,
            audio_seconds,
            decode_seconds,
        }))
    }
}

/// `authorization: Bearer <token>`, constant-time compared against every
/// configured client credential.
///
/// ADR-0044 Decision 5 specified ONE shared token, which was right when the
/// only callers were a browser and the owner's machines. Other services on the
/// cluster calling this changes that: one shared secret means revoking any
/// caller revokes all of them, and it is why ADR-0046 had to accept that a
/// caller can interleave audio into another's `session_id` — they all hold the
/// same credential, so the id is the only thing separating them.
///
/// Credentials are read from the environment as `STT_TOKEN_<NAME>`, one per
/// caller, plus the original `STT_AUTH_TOKEN` as `default` so nothing that
/// works today stops working. Adding a caller is a new Infisical secret;
/// revoking one is deleting it. Both are one action affecting one caller.
///
/// forwardAuth remains rejected on the ground ADR-0044 gave: authentik's proxy
/// provider answers an unauthenticated request with a 302 to a login page,
/// which a native gRPC client cannot follow.
#[derive(Clone)]
pub struct BearerAuth {
    clients: Arc<Vec<(String, Vec<u8>)>>,
}

impl BearerAuth {
    /// Collect every configured credential. Errors if there are none rather
    /// than starting an unauthenticated GPU on a hostname that Certificate
    /// Transparency publishes minutes after issuance.
    pub fn from_env() -> anyhow::Result<Self> {
        let mut clients: Vec<(String, Vec<u8>)> = std::env::vars()
            .filter(|(_, v)| !v.is_empty())
            .filter_map(|(k, v)| match k.strip_prefix("STT_TOKEN_") {
                Some(name) => Some((name.to_lowercase(), v.into_bytes())),
                None if k == "STT_AUTH_TOKEN" => Some(("default".to_string(), v.into_bytes())),
                None => None,
            })
            .collect();
        clients.sort_by(|a, b| a.0.cmp(&b.0));

        anyhow::ensure!(
            !clients.is_empty(),
            "no credentials configured. Set STT_AUTH_TOKEN, or one STT_TOKEN_<NAME> per caller, \
             via Infisical (project ukubi-stt-bhr-m). Refusing to serve a GPU with no \
             authentication on a publicly-resolvable hostname."
        );
        tracing::info!(
            clients = ?clients.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            "bearer credentials loaded"
        );
        Ok(Self {
            clients: Arc::new(clients),
        })
    }
}

impl Interceptor for BearerAuth {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        let value = request
            .metadata()
            .get("authorization")
            .ok_or_else(|| Status::unauthenticated("missing authorization metadata"))?
            .to_str()
            .map_err(|_| Status::unauthenticated("authorization is not valid ASCII"))?;

        let presented = value
            .strip_prefix("Bearer ")
            .ok_or_else(|| Status::unauthenticated("authorization must be `Bearer <token>`"))?
            .as_bytes();

        // Every credential is compared, with no early exit on the first match:
        // stopping early would make the response time depend on WHICH client is
        // calling. subtle's slice impl returns Choice(0) on a length mismatch,
        // so that case is covered too.
        let mut matched: Option<&str> = None;
        for (name, token) in self.clients.iter() {
            if bool::from(presented.ct_eq(token)) {
                matched = Some(name);
            }
        }

        match matched {
            Some(name) => {
                request
                    .extensions_mut()
                    .insert(ClientName(name.to_string()));
                Ok(request)
            }
            None => Err(Status::unauthenticated("invalid token")),
        }
    }
}

/// The browser test client, served from the SAME ORIGIN as the gRPC endpoint.
///
/// That is the whole reason it lives in this binary rather than anywhere else:
/// same-origin means no CORS on the RPC at all — no preflight carrying
/// `authorization`, no `grpcWeb.allowOrigins` to keep in step with wherever the
/// page is hosted, and no failure mode where the RPC works from grpcurl and not
/// from a browser. `include_str!` so it ships in the binary; a page with no
/// external assets needs no volume, no second image and no static file server.
async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../web/index.html"))
}

/// `/` (the test page), `/healthz` and `/livez`, on a port whose ONLY externally
/// routed path is `/` — the IngressRoute matches `Path(`/`)` exactly, so the
/// health endpoints stay unreachable from outside as ADR-0044 Decision 5
/// requires. Routing this port at all is new; routing the probes is not.
///
/// It starts only after the model has loaded and CUDA has been asserted, so
/// "the port answers" and "the service can actually decode" are the same fact.
/// A readiness probe that goes green before the GPU is proven is worse than no
/// probe at all.
pub async fn serve_http(addr: SocketAddr) -> anyhow::Result<()> {
    let app = axum::Router::new()
        .route("/", axum::routing::get(index))
        .route("/healthz", axum::routing::get(|| async { "ok" }))
        .route("/livez", axum::routing::get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("http (test page + health) listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
