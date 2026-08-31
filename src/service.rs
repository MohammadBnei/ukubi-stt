//! The gRPC surface: one unary RPC, a bearer-token interceptor, and an
//! unauthenticated health listener on a separate port.

use crate::engine::{pcm_s16le_to_f32, SAMPLE_RATE};
use parakeet_rs::{ParakeetTDT, Transcriber};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
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

/// One decode at a time. One 8 GB GPU running one model is honest about what it
/// can do, and it doubles as abuse protection on a hostname that Certificate
/// Transparency publishes within minutes of issuance.
///
/// ponytail: `try_acquire` and reject, never queue. A queue on a single GPU
/// converts overload into unbounded latency, which is harder to diagnose than a
/// clean RESOURCE_EXHAUSTED. Phase E's streaming needs a different shape
/// entirely — a shared recognizer with per-session streams — because holding
/// this permit for a stream's lifetime would let one browser tab starve every
/// batch caller.
pub struct SttService {
    model: Arc<Mutex<ParakeetTDT>>,
    permits: Arc<tokio::sync::Semaphore>,
}

impl SttService {
    pub fn new(model: ParakeetTDT) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            permits: Arc::new(tokio::sync::Semaphore::new(1)),
        }
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

        // Acquire BEFORE spawning: the point is to reject the caller, not to
        // park a blocking thread waiting for the GPU.
        let _permit = self.permits.clone().try_acquire_owned().map_err(|_| {
            Status::resource_exhausted(
                "the GPU is busy with another decode — one at a time, by design. Retry.",
            )
        })?;

        let model = Arc::clone(&self.model);
        let started = Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            // A decode that panicked poisons this mutex. Recovering is
            // deliberate: the alternative is that one bad request bricks the
            // service until someone notices a pod that is Ready and answers
            // every call with the same panic.
            let mut model = model.lock().unwrap_or_else(|poisoned| {
                tracing::warn!("model mutex was poisoned by a previous panic; recovering");
                poisoned.into_inner()
            });
            model.transcribe_samples(samples, SAMPLE_RATE, 1, None)
        })
        .await
        .map_err(|e| Status::internal(format!("decode task failed: {e}")))?
        .map_err(|e| Status::internal(format!("transcription failed: {e}")))?;

        let decode_seconds = started.elapsed().as_secs_f32();
        tracing::info!(
            audio_seconds,
            decode_seconds,
            rtf = decode_seconds / audio_seconds,
            chars = result.text.len(),
            "recognize"
        );

        Ok(Response::new(pb::RecognizeResponse {
            text: result.text,
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

/// `/healthz` and `/livez` on a separate port, unauthenticated and never routed
/// externally — only :9090 gets an IngressRoute.
///
/// It starts only after the model has loaded and CUDA has been asserted, so
/// "the port answers" and "the service can actually decode" are the same fact.
/// A readiness probe that goes green before the GPU is proven is worse than no
/// probe at all.
pub async fn serve_health(addr: SocketAddr) -> anyhow::Result<()> {
    let app = axum::Router::new()
        .route("/healthz", axum::routing::get(|| async { "ok" }))
        .route("/livez", axum::routing::get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("health listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
