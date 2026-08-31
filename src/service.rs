//! The gRPC surface: one unary RPC, a bearer-token interceptor, and an
//! unauthenticated health listener on a separate port.

use crate::engine::{is_multilingual, load_streaming_and_assert_cuda, pcm_s16le_to_f32, SAMPLE_RATE};
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
        tracing::info!(session = id, sessions = map.len(), "streaming session opened");
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
        samples: Vec<f32>,
    ) -> Result<String, Status> {
        let handle = self.handle().await?;
        let recognizer = self.session(session_id, &handle, language)?;

        let text = tokio::task::spawn_blocking(move || lock(&recognizer).transcribe_chunk(&samples))
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
        if req.audio.is_empty() {
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
            rtf = decode_seconds / audio_seconds,
            chars = text.len(),
            streaming = !req.session_id.is_empty(),
            "recognize"
        );

        Ok(Response::new(pb::RecognizeResponse {
            text,
            audio_seconds,
            decode_seconds,
        }))
    }
}

/// `authorization: Bearer <token>`, constant-time compared.
///
/// ADR-0044 Decision 5. authentik forwardAuth is rejected here on a technical
/// ground rather than a preference: its proxy provider answers an
/// unauthenticated request with a 302 to a login page, which a native gRPC
/// client cannot follow — it sees a non-`application/grpc` response and fails
/// opaquely.
#[derive(Clone)]
pub struct BearerAuth {
    expected: Arc<Vec<u8>>,
}

impl BearerAuth {
    pub fn new(token: String) -> Self {
        Self {
            expected: Arc::new(token.into_bytes()),
        }
    }
}

impl Interceptor for BearerAuth {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let value = request
            .metadata()
            .get("authorization")
            .ok_or_else(|| Status::unauthenticated("missing authorization metadata"))?
            .to_str()
            .map_err(|_| Status::unauthenticated("authorization is not valid ASCII"))?;

        let presented = value
            .strip_prefix("Bearer ")
            .ok_or_else(|| Status::unauthenticated("authorization must be `Bearer <token>`"))?;

        // subtle's slice impl returns Choice(0) for a length mismatch, so this
        // covers both halves. The token's *length* is not secret; its content is.
        if bool::from(presented.as_bytes().ct_eq(&self.expected)) {
            Ok(request)
        } else {
            Err(Status::unauthenticated("invalid token"))
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
