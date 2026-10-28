//! The closed registry of error keys, and the only place a `tonic::Status` is
//! built (clippy.toml bans every other constructor).
//!
//! Every Status leaves with `grpc-status-details-bin`: a `google.rpc.Status`
//! whose code and message are the same as the headers' (grpc-go takes them from
//! here, and answers INTERNAL "grpc-status-details-bin mismatch" when they
//! differ) and whose one detail is `stt.v1.ErrorKey`. Clients branch on the code
//! and the key, never on the message.
//!
//! `source` is always empty: no message here carries anyone else's words. An
//! internal failure is logged and the caller gets a fixed text, so a model path
//! or a GPU figure never reaches the wire.
//!
//! Not keyed: errors tonic raises before a handler runs (body too large,
//! unknown method), reflection's, and tonic's own `From<io::Error>` /
//! `From<h2::Error>` conversions, which clippy cannot ban. No code here uses
//! those conversions; a tower layer that keys them is the upgrade if one appears.

use crate::service::pb::ErrorKey;
use prost::Message;
use std::fmt::Display;
use tonic::{Code, Status};

/// Keys whose message is written by this service, for the caller. The two keys
/// whose cause is internal have no variant here, so `status()` cannot send
/// internal text under them: only `internal()` and `model_unavailable()` can.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Unauthenticated,
    BadSampleRate,
    EmptyAudio,
    BadAudio,
    UnknownLanguage,
    EngineMismatch,
    SessionCapReached,
    GpuBusy,
}

const SERVER_INTERNAL: &str = "server.internal";
const MODEL_UNAVAILABLE: &str = "model.unavailable";

impl Key {
    /// The wire string. Stable: a client may switch on it.
    fn as_str(self) -> &'static str {
        match self {
            Key::Unauthenticated => "auth.unauthenticated",
            Key::BadSampleRate => "request.bad_sample_rate",
            Key::EmptyAudio => "request.empty_audio",
            Key::BadAudio => "request.bad_audio",
            Key::UnknownLanguage => "language.unknown",
            Key::EngineMismatch => "session.engine_mismatch",
            Key::SessionCapReached => "session.cap_reached",
            Key::GpuBusy => "gpu.busy",
        }
    }

    /// One code per key, so a key never travels under two codes.
    fn code(self) -> Code {
        match self {
            Key::Unauthenticated => Code::Unauthenticated,
            Key::BadSampleRate
            | Key::EmptyAudio
            | Key::BadAudio
            | Key::UnknownLanguage
            | Key::EngineMismatch => Code::InvalidArgument,
            Key::SessionCapReached | Key::GpuBusy => Code::ResourceExhausted,
        }
    }
}

/// A refusal in this service's own words.
pub fn status(key: Key, message: impl Into<String>) -> Status {
    build(key.code(), key.as_str(), message.into())
}

/// An internal failure: the full error is logged at ERROR, the caller gets only
/// "internal error".
pub fn internal(what: &str, err: &dyn Display) -> Status {
    tracing::error!("{what}: {err:#}");
    build(Code::Internal, SERVER_INTERNAL, "internal error".into())
}

/// A lazily loaded model that would not load. WARN, not ERROR: batch is
/// unaffected, and every streaming request retries the load, so this repeats per
/// request while the model is missing. The caller learns which model, not why.
pub fn model_unavailable(model: &'static str, err: &dyn Display) -> Status {
    tracing::warn!("{model} failed to load: {err:#}");
    build(
        Code::FailedPrecondition,
        MODEL_UNAVAILABLE,
        format!("the {model} is not available on this server"),
    )
}

/// `google.rpc.Status`, the payload of `grpc-status-details-bin`. Three fields;
/// not worth a googleapis dependency.
#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

const ERROR_KEY_TYPE_URL: &str = "type.googleapis.com/stt.v1.ErrorKey";

fn build(code: Code, key: &str, message: String) -> Status {
    let detail = ErrorKey {
        key: key.to_string(),
        source: String::new(),
    };
    let details = RpcStatus {
        code: code.into(),
        message: message.clone(),
        details: vec![prost_types::Any {
            type_url: ERROR_KEY_TYPE_URL.to_string(),
            value: detail.encode_to_vec(),
        }],
    };
    with_details(code, message, details.encode_to_vec())
}

// One expression wide: this expect also lifts the std::env ban, so it covers
// nothing else. Its being fulfilled is the ban's positive control.
#[expect(clippy::disallowed_methods, reason = "the one Status constructor")]
fn with_details(code: Code, message: String, details: Vec<u8>) -> Status {
    Status::with_details(code, message, details.into())
}

/// What a client decodes from a Status: (code, message, key, source), read back
/// through the wire headers exactly as tonic writes them.
#[cfg(test)]
pub fn read_back(status: Status) -> (Code, String, String, String) {
    let http = status.into_http::<()>();
    let back = Status::from_header_map(http.headers()).expect("grpc-status header");
    let rpc = RpcStatus::decode(back.details()).expect("google.rpc.Status");
    assert_eq!(
        rpc.code,
        i32::from(back.code()),
        "details code == grpc-status"
    );
    assert_eq!(
        rpc.message,
        back.message(),
        "details message == grpc-message"
    );
    assert_eq!(rpc.details.len(), 1, "one detail");
    assert_eq!(rpc.details[0].type_url, ERROR_KEY_TYPE_URL);
    let key = ErrorKey::decode(rpc.details[0].value.as_slice()).expect("stt.v1.ErrorKey");
    (back.code(), back.message().to_string(), key.key, key.source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Key; 8] = [
        Key::Unauthenticated,
        Key::BadSampleRate,
        Key::EmptyAudio,
        Key::BadAudio,
        Key::UnknownLanguage,
        Key::EngineMismatch,
        Key::SessionCapReached,
        Key::GpuBusy,
    ];

    #[test]
    fn keys_are_unique_and_well_formed() {
        let mut all: Vec<&str> = ALL.iter().map(|k| k.as_str()).collect();
        all.extend([SERVER_INTERNAL, MODEL_UNAVAILABLE]);
        for k in &all {
            let mut parts = k.split('.');
            let head = parts.next().unwrap();
            let rest: Vec<&str> = parts.collect();
            assert!(
                !head.is_empty() && head.bytes().all(|b| b.is_ascii_lowercase()),
                "{k}"
            );
            assert!(!rest.is_empty(), "{k} needs a dot");
            for p in rest {
                assert!(
                    !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
                    "{k}"
                );
            }
        }
        let n = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n, "a key string is used twice");
    }

    #[test]
    fn every_key_round_trips_with_matching_code_and_message() {
        for k in ALL {
            let (code, msg, key, source) = read_back(status(k, "own words"));
            assert_eq!((code, msg.as_str()), (k.code(), "own words"));
            assert_eq!((key.as_str(), source.as_str()), (k.as_str(), ""));
        }
    }

    #[test]
    fn internal_text_never_reaches_the_wire() {
        let err = anyhow::anyhow!("secret /models/x").context("loading");
        let (code, msg, key, _) = read_back(internal("decode task failed", &err));
        assert_eq!(
            (code, msg.as_str(), key.as_str()),
            (Code::Internal, "internal error", SERVER_INTERNAL)
        );

        let (code, msg, key, _) = read_back(model_unavailable("streaming model", &err));
        assert_eq!(code, Code::FailedPrecondition);
        assert_eq!(msg, "the streaming model is not available on this server");
        assert_eq!(key, MODEL_UNAVAILABLE);
        // Positive control: the error does carry the text the caller must not see.
        assert!(format!("{err:#}").contains("secret"));
        assert!(!msg.contains("secret") && !msg.contains("/models"));
    }

    /// The GPU-only paths cannot run in a test, so their routing is pinned by
    /// count: replacing one with `status(..)` or a formatted message fails here.
    #[test]
    fn the_gpu_paths_go_through_the_internal_constructors() {
        let src = include_str!("service.rs");
        assert_eq!(src.matches(concat!("errkeys::", "internal(")).count(), 8);
        assert_eq!(
            src.matches(concat!("errkeys::", "model_unavailable("))
                .count(),
            2
        );
    }
}
