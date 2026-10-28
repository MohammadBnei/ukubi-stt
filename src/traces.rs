//! Opt-in traces: one OTLP/HTTP server span per gRPC call. Nothing here runs
//! unless `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set (config.rs checks it), so
//! with it unset the subscriber has no OTel layer and the gRPC server no layer.
//!
//! Everything is passed explicitly: no OTel global (provider, propagator or
//! error handler) is set. The SDK's internal-logs feature is off, so it writes
//! no log lines of its own; an export failure is one WARN from [`WarnOnFailure`].

use opentelemetry::propagation::TextMapPropagator as _;
use opentelemetry::trace::{TraceContextExt as _, TracerProvider as _};
use opentelemetry_otlp::{WithExportConfig as _, WithHttpConfig as _};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider, SpanData, SpanExporter};
use std::time::Duration;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;
use tracing_subscriber::{filter::Targets, registry::LookupSpan, Layer};

/// The target of the gRPC call span, and nothing else. The OTel layer sees
/// only this target, so decode_chunk and every event stay out of traces; the
/// fmt layer sees everything but this, so log lines do not change.
pub const SPAN_TARGET: &str = "ukubi_stt::grpc_call";

/// Every method this server answers, as the span is named: the path without
/// its leading slash. Any other path gets no span, so a request no route
/// takes is not traced. `tests::known_methods_cover_the_proto` holds the
/// stt.v1 entries to the proto.
pub const METHODS: [&str; 3] = [
    "stt.v1.Stt/Recognize",
    "grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
    "grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo",
];

pub struct Tracing(SdkTracerProvider);

impl Tracing {
    /// Builds the exporter and the provider. Call outside a tokio runtime: the
    /// blocking client runs its own. `endpoint` is the full traces URL, already
    /// checked by config.rs. An error here turns tracing off with a WARN.
    /// (An unsupported OTEL_EXPORTER_OTLP_COMPRESSION is not one: the SDK
    /// ignores it and sends uncompressed.)
    pub fn new(endpoint: &str) -> anyhow::Result<Self> {
        // no_proxy: reqwest reads HTTP_PROXY and friends even without its
        // system-proxy feature. The timeout bounds one request; the SDK's
        // OTEL_EXPORTER_OTLP_TIMEOUT only bounds its retries on our client.
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()?;
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_http_client(client)
            .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
            .with_endpoint(endpoint)
            .build()?;
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(WarnOnFailure(exporter))
            // Every span is a new root (callers are linked, never parents), so
            // there is no remote decision to follow. Overrides OTEL_TRACES_SAMPLER.
            .with_sampler(Sampler::AlwaysOn)
            // builder_empty: no env detectors, so OTEL_SERVICE_NAME and
            // OTEL_RESOURCE_ATTRIBUTES cannot rename the service.
            .with_resource(
                opentelemetry_sdk::Resource::builder_empty()
                    .with_service_name("ukubi-stt")
                    .build(),
            )
            .build();
        Ok(Self(provider))
    }

    /// The OTel layer, filtered to the gRPC call span.
    pub fn layer<S>(&self) -> impl Layer<S>
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        tracing_opentelemetry::layer()
            .with_tracer(self.0.tracer("ukubi-stt"))
            .with_location(false)
            .with_threads(false)
            .with_target(false)
            .with_tracked_inactivity(false)
            .with_filter(Targets::new().with_target(SPAN_TARGET, tracing::Level::INFO))
    }

    /// Exports what is queued and stops. Call after the server has drained, and
    /// outside a tokio runtime. Bounded by the SDK's own shutdown timeout.
    pub fn shutdown(self) {
        if let Err(e) = self.0.shutdown() {
            tracing::warn!("trace flush on shutdown failed: {e}");
        }
    }
}

/// The span of one gRPC call, or `Span::none()` for a path no route takes.
/// `parent: None`: a new root whatever the thread's current context. A
/// caller's `traceparent` is a link, never the parent, because it is read
/// before the bearer check and so is whatever a caller chose to send.
pub fn make_span<B>(request: &http::Request<B>) -> Span {
    let path = request.uri().path().strip_prefix('/').unwrap_or_default();
    let Some((method, (service, rpc))) = METHODS
        .iter()
        .find(|m| **m == path)
        .and_then(|m| Some((*m, m.split_once('/')?)))
    else {
        return Span::none();
    };
    let span = tracing::info_span!(
        target: SPAN_TARGET,
        parent: None,
        "grpc",
        otel.name = method,
        otel.kind = "server",
        otel.status_code = tracing::field::Empty,
        rpc.system = "grpc",
        rpc.service = service,
        rpc.method = rpc,
        rpc.grpc.status_code = tracing::field::Empty,
    );
    let caller = opentelemetry_sdk::propagation::TraceContextPropagator::new()
        .extract(&opentelemetry_http::HeaderExtractor(request.headers()));
    let caller = caller.span().span_context().clone();
    if caller.is_valid() {
        span.add_link(caller);
    }
    span
}

/// A trailers-only response (an error before any message) carries
/// `grpc-status` in its headers.
pub fn on_response<B>(response: &http::Response<B>, _: Duration, span: &Span) {
    record(response.headers(), span);
}

/// Otherwise it is in the trailers.
pub fn on_eos(trailers: Option<&http::HeaderMap>, _: Duration, span: &Span) {
    if let Some(trailers) = trailers {
        record(trailers, span);
    }
}

fn record(headers: &http::HeaderMap, span: &Span) {
    let Some(code) = headers
        .get("grpc-status")
        .and_then(|v| v.to_str().ok()?.parse::<i32>().ok())
    else {
        return;
    };
    span.record("rpc.grpc.status_code", code);
    // The server's faults, per the OTel gRPC semconv, are ERROR. A client's
    // own fault (UNAUTHENTICATED, INVALID_ARGUMENT, ...) stays Unset, and so
    // does FAILED_PRECONDITION (model_unavailable): its WARN line covers it.
    use tonic::Code;
    if matches!(
        Code::from_i32(code),
        Code::Unknown
            | Code::DeadlineExceeded
            | Code::Unimplemented
            | Code::Internal
            | Code::Unavailable
            | Code::DataLoss
    ) {
        span.record("otel.status_code", "ERROR");
    }
}

/// The OTLP exporter, with each failed export logged once at WARN (a
/// collector hiccup is not our failure, so never ERROR).
#[derive(Debug)]
struct WarnOnFailure(opentelemetry_otlp::SpanExporter);

impl SpanExporter for WarnOnFailure {
    fn export(
        &self,
        batch: Vec<SpanData>,
    ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
        let export = self.0.export(batch);
        async move {
            let result = export.await;
            if let Err(e) = &result {
                tracing::warn!("trace export failed: {e}");
            }
            result
        }
    }
    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }
    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }
    fn set_resource(&mut self, resource: &opentelemetry_sdk::Resource) {
        self.0.set_resource(resource);
    }
}

#[cfg(test)]
mod tests {
    /// Every method in the compiled proto has a span name, so a new RPC is
    /// traced the day it is added, not the day someone notices.
    #[test]
    fn known_methods_cover_the_proto() {
        use prost::Message as _;
        let set =
            prost_types::FileDescriptorSet::decode(crate::service::FILE_DESCRIPTOR_SET).unwrap();
        let mut names = Vec::new();
        for file in &set.file {
            for service in &file.service {
                for method in &service.method {
                    names.push(format!(
                        "{}.{}/{}",
                        file.package(),
                        service.name(),
                        method.name()
                    ));
                }
            }
        }
        assert!(!names.is_empty(), "positive control: the set decoded");
        for name in &names {
            assert!(super::METHODS.contains(&name.as_str()), "{name} untraced");
        }
    }
}
