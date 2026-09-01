//! Persian streaming recognition — Shenava Koochik v1.0, a cache-aware FastConformer
//! CTC model driven through `ort` directly (ADR-0047).
//!
//! This is the first direct use of `ort` in this service. The other two models go
//! through `parakeet-rs`, which has no CTC type, so the session, the tensor
//! plumbing, the cache threading and the greedy decode all live here.
//!
//! ## The graph's contract, read off the file rather than the model card
//!
//! ```text
//! in : audio_signal           f32 [1, 80, 121]      unnormalized NeMo log-mel
//!      length                 i64 [1]               true frame count, pre-padding
//!      cache_last_channel     f32 [1, 17, 70, 512]
//!      cache_last_time        f32 [1, 17, 512, 8]
//!      cache_last_channel_len i64 [1]
//! out: logprobs               f32 [1, 14, 1025]
//!      encoded_lengths        i64 [1]
//!      cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len
//! ```
//!
//! **The export emits exactly 14 steps and already drops the pre-encode overlap.**
//! That was the open question this model posed: 121 input frames at
//! `subsampling_factor: 8` do not divide by the 14 encoder frames the 112-frame
//! shift implies, and NeMo's streaming path carries a `drop_extra_pre_encoded` for
//! it. Measured on the real graph, `logprobs` is `[1, 14, 1025]` — 112/8 exactly —
//! so consuming every emitted step is correct here. Had it emitted more, taking
//! them all would have duplicated a token every 1.12 s, fluently and invisibly.
//!
//! `encoded_lengths` tracks `length` (feed 60 frames, get 7 steps), which is what
//! makes padding the final chunk safe. `cache_last_channel_len` climbs 14, 28, 42,
//! 56, 70 and then saturates on its own — thread it through, do not clamp it.

use crate::fbank::{Fbank, N_MELS};
use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Log-mel frames consumed per step, and how far the window advances. The 9-frame
/// difference is the pre-encode overlap the graph needs to see and then discards.
pub const CHUNK_FRAMES: usize = 121;
/// 112 frames at a 10 ms hop is 1.12 s per step — twice Nemotron's 560 ms, and the
/// floor on how fast Persian text can appear.
pub const SHIFT_FRAMES: usize = 112;

const CACHE_CH: [usize; 4] = [1, 17, 70, 512];
const CACHE_T: [usize; 4] = [1, 17, 512, 8];
/// `<blk>` is the last line of tokens.txt. Asserted at load, because an off-by-one
/// blank id yields fluent-looking garbage rather than an error.
const BLANK: usize = 1024;
const VOCAB: usize = 1025;

/// Which execution provider to ask for.
///
/// Deliberately a parameter and not a constant: the CPU/CUDA choice for this model
/// is unsettled. The published "83.9 ms per 1.12 s chunk" is a *tract* number of
/// unknown thread count, so it transfers to neither ORT provider, and the card
/// already holds 6880 MiB of 8192 with two models resident. `--selftest-fa`
/// measures both so the decision is made with a number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}

pub struct PersianModel {
    session: Session,
    /// Behind an `Arc` so a decode can hold the vocabulary after releasing the
    /// session lock — 1025 Strings, cloned once at load rather than per step.
    tokens: Arc<Vec<String>>,
}

impl PersianModel {
    /// Loads `model.onnx` and `tokens.txt` from `dir`.
    ///
    /// On `Device::Cuda` the provider is registered with `error_on_failure`, which
    /// is where `parakeet-rs` gets it wrong for the other two models: it attaches
    /// that to the CPU fallback instead, so CUDA declining is silent. The caller
    /// still wants the GPU-memory delta check on top — this catches "the provider
    /// did not register", the delta catches "it registered and then placed every
    /// node on CPU".
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let tokens = load_tokens(&dir.join("tokens.txt"))?;

        let mut builder = Session::builder().context("creating an ort session builder")?;
        if device == Device::Cuda {
            // `ort::Error<SessionBuilder>` carries the builder back and is not
            // Send + Sync, so anyhow's `.context` will not take it; the message is
            // attached by hand instead.
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                .map_err(|e| {
                    anyhow::anyhow!(
                        "registering the CUDA execution provider for the Persian model \
                         failed: {e}. This is the provider declining, not the model — \
                         check libonnxruntime_providers_cuda.so sits beside the binary \
                         and that the CUDA major matches the base image."
                    )
                })?;
        }
        let model = dir.join("model.onnx");
        let session = builder
            .commit_from_file(&model)
            .with_context(|| format!("loading the Persian model from {}", model.display()))?;

        // The graph declares no output shapes, so the names are the only thing that
        // can be checked at load. Bail listing what was actually found rather than
        // indexing into a map later and getting a less useful panic.
        let found: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        for want in [
            "logprobs",
            "encoded_lengths",
            "cache_last_channel_next",
            "cache_last_time_next",
            "cache_last_channel_next_len",
        ] {
            if !found.iter().any(|f| f == want) {
                bail!("Persian model is missing output {want:?}; it has {found:?}");
            }
        }

        Ok(Self {
            session,
            tokens: Arc::new(tokens),
        })
    }
}

/// One dictation. Holds the recogniser state that makes a stream a stream: the
/// encoder caches, the feature extractor's sample tail, and the last emitted token.
pub struct PersianStream {
    /// One session shared by every Persian stream. `Session::run` takes
    /// `&mut self`, so this serialises Persian inference — at 44.8 ms per 1.12 s
    /// step that is 4% duty per stream, so roughly 25 streams before the lock is
    /// the limit, against a session cap of 8.
    ///
    /// ponytail: one shared session. A second only if that ceiling is ever hit.
    model: Arc<Mutex<PersianModel>>,
    fbank: Fbank,
    frames: VecDeque<[f32; N_MELS]>,
    cache_ch: Vec<f32>,
    cache_t: Vec<f32>,
    cache_len: i64,
    /// Carried across chunk boundaries so a CTC symbol spanning one is not emitted
    /// twice. Without it every step boundary can double a character.
    prev_token: usize,
    /// `<unk>` count. Not dropped silently: a Persian decode that has gone wrong
    /// announces itself as `<unk>` spam, and quietly filtering that turns a visible
    /// failure into short, fluent, wrong output.
    unks: usize,
    finished: bool,
    /// Whether anything has been emitted yet, so the utterance's leading word
    /// separator can be trimmed exactly once.
    emitted: bool,
}

impl PersianStream {
    pub fn new(model: Arc<Mutex<PersianModel>>) -> Self {
        Self {
            model,
            fbank: Fbank::new().expect("bundled filterbank is valid; checked by a unit test"),
            frames: VecDeque::new(),
            cache_ch: vec![0.0; CACHE_CH.iter().product()],
            cache_t: vec![0.0; CACHE_T.iter().product()],
            cache_len: 0,
            prev_token: BLANK,
            unks: 0,
            finished: false,
            emitted: false,
        }
    }

    pub fn unks(&self) -> usize {
        self.unks
    }

    /// Feed audio, get back the text for whatever whole steps it completed.
    ///
    /// Returns an empty string when the buffer has not reached `CHUNK_FRAMES` yet.
    /// That is normal and frequent: clients send 560 ms chunks and a step needs
    /// 1.12 s, so the first two calls of a dictation produce nothing and roughly
    /// every other one after that does. Callers concatenate.
    pub fn push(&mut self, samples: &[f32], last: bool) -> Result<String> {
        if self.finished {
            bail!("this Persian stream was already closed with last: true");
        }
        self.frames.extend(self.fbank.push(samples, last));
        self.finished = last;

        let mut text = String::new();
        while self.frames.len() >= CHUNK_FRAMES || (last && !self.frames.is_empty()) {
            let take = self.frames.len().min(CHUNK_FRAMES);
            let mut window = vec![0.0f32; N_MELS * CHUNK_FRAMES];
            // The ONNX layout is [1, 80, 121] — mel-major. `frames` is time-major,
            // which is what streaming wants, so this is where it transposes.
            for (t, frame) in self.frames.iter().take(take).enumerate() {
                for (m, v) in frame.iter().enumerate() {
                    window[m * CHUNK_FRAMES + t] = *v;
                }
            }
            text.push_str(&self.step(window, take)?);

            for _ in 0..SHIFT_FRAMES.min(self.frames.len()) {
                self.frames.pop_front();
            }
            if last
                && self.frames.len() < CHUNK_FRAMES
                && self.frames.len() <= CHUNK_FRAMES - SHIFT_FRAMES
            {
                // The tail is entirely overlap already consumed by the last window.
                self.frames.clear();
            }
        }
        // SentencePiece `▁` is a word separator, so it becomes a space HERE rather
        // than in the caller. Doing it per chunk and not trimming is deliberate: a
        // word boundary can fall on a chunk boundary, and trimming each chunk would
        // silently glue the last word of one to the first word of the next. Only
        // the utterance's very first separator is dropped.
        let mut text = text.replace('\u{2581}', " ");
        if !self.emitted {
            let trimmed = text.trim_start().to_string();
            if !trimmed.is_empty() {
                self.emitted = true;
            }
            text = trimmed;
        }
        Ok(text)
    }

    fn step(&mut self, window: Vec<f32>, true_frames: usize) -> Result<String> {
        // The lock is held across this one ~45 ms inference and nothing else.
        // Never across a whole request: one shared session means a batch decode
        // holding it for minutes would freeze every live dictation.
        let shared = Arc::clone(&self.model);
        let mut model = shared.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Persian model mutex was poisoned by a panic; recovering");
            poisoned.into_inner()
        });

        let (vocab, usable, logprobs, cache_ch, cache_t, cache_len) = {
            let outputs = model
                .session
                .run(ort::inputs![
                "audio_signal" => Tensor::from_array(([1usize, N_MELS, CHUNK_FRAMES], window))?,
                // The padded window is always CHUNK_FRAMES wide; `length` is what
                // tells the encoder how much of it is real, and `encoded_lengths`
                // comes back scaled by the subsampling factor.
                "length" => Tensor::from_array(([1usize], vec![true_frames as i64]))?,
                "cache_last_channel" => Tensor::from_array((CACHE_CH, std::mem::take(&mut self.cache_ch)))?,
                "cache_last_time" => Tensor::from_array((CACHE_T, std::mem::take(&mut self.cache_t)))?,
                "cache_last_channel_len" => Tensor::from_array(([1usize], vec![self.cache_len]))?,
                ])
                .context("Persian encoder step")?;

            let (lp_shape, logprobs) = outputs["logprobs"].try_extract_tensor::<f32>()?;
            let steps = lp_shape[1] as usize;
            let vocab = lp_shape[2] as usize;
            if vocab != VOCAB {
                bail!("Persian model emitted a {vocab}-wide vocabulary, expected {VOCAB}");
            }
            let (_, enc_len) = outputs["encoded_lengths"].try_extract_tensor::<i64>()?;
            let usable = steps.min(enc_len[0].max(0) as usize);

            // Copied out so the session borrow ends here and the lock can be
            // released before the decode, which needs no model state.
            let (_, ch) = outputs["cache_last_channel_next"].try_extract_tensor::<f32>()?;
            let (_, t) = outputs["cache_last_time_next"].try_extract_tensor::<f32>()?;
            let (_, len) = outputs["cache_last_channel_next_len"].try_extract_tensor::<i64>()?;
            (
                vocab,
                usable,
                logprobs.to_vec(),
                ch.to_vec(),
                t.to_vec(),
                len[0],
            )
        };

        // Thread the caches forward. They are the stream's entire memory of the
        // utterance; `cache_last_channel_len` saturates at 70 by itself.
        self.cache_ch = cache_ch;
        self.cache_t = cache_t;
        self.cache_len = cache_len;

        // Cheap: an Arc bump, not 1025 String clones. Releasing the lock before
        // the decode keeps the shared session free for other streams.
        let tokens = Arc::clone(&model.tokens);
        drop(model);

        Ok(self.decode(&tokens, &logprobs, usable, vocab))
    }

    /// Greedy CTC: argmax per step, drop blanks, drop repeats, drop specials, then
    /// SentencePiece `▁` becomes a space. `prev_token` persists across calls so a
    /// symbol held across a step boundary collapses correctly.
    fn decode(
        &mut self,
        tokens: &[String],
        logprobs: &[f32],
        steps: usize,
        vocab: usize,
    ) -> String {
        let mut out = String::new();
        for t in 0..steps {
            let row = &logprobs[t * vocab..(t + 1) * vocab];
            let mut best = 0usize;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            if best != self.prev_token && best != BLANK {
                let tok = &tokens[best];
                if best == 0 {
                    self.unks += 1;
                } else if !(tok.starts_with('<') && tok.ends_with('>')) {
                    out.push_str(tok);
                }
            }
            self.prev_token = best;
        }
        out
    }
}

/// `tokens.txt` is `<token> <id>` per line, id ascending, 1025 lines.
fn load_tokens(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading the Persian vocabulary from {}", path.display()))?;
    let mut tokens = vec![String::new(); VOCAB];
    let mut seen = 0usize;
    for line in raw.lines().filter(|l| !l.is_empty()) {
        let (tok, idx) = line
            .rsplit_once(' ')
            .with_context(|| format!("malformed tokens.txt line {line:?}"))?;
        let idx: usize = idx
            .parse()
            .with_context(|| format!("malformed token id in {line:?}"))?;
        if idx >= VOCAB {
            bail!("token id {idx} is outside the {VOCAB}-token vocabulary");
        }
        tokens[idx] = tok.to_string();
        seen += 1;
    }
    if seen != VOCAB {
        bail!("tokens.txt has {seen} entries, expected {VOCAB}");
    }
    // An off-by-one blank decodes to fluent nonsense rather than to an error, so it
    // is checked rather than assumed.
    if tokens[BLANK] != "<blk>" {
        bail!(
            "token {BLANK} is {:?}, expected \"<blk>\" — the blank id has moved",
            tokens[BLANK]
        );
    }
    Ok(tokens)
}

/// Trailing tidy for whole transcripts. `PersianStream::push` already turns the
/// SentencePiece separators into spaces as it goes, because a word boundary can
/// fall on a chunk boundary and trimming per chunk would glue words together.
pub fn tidy(s: &str) -> String {
    s.trim().to_string()
}

/// Character error rate, for `--selftest-fa`.
///
/// Characters and not words: Persian word-level error is dominated by ezāfe and
/// ZWNJ orthographic variants, so WER moves for reasons that have nothing to do
/// with whether the pipeline is right. A feature-extraction bug shows up as
/// character garbage long before that argument starts.
pub fn cer(reference: &str, hypothesis: &str) -> f32 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, hc) in h.iter().enumerate() {
            let sub = prev[j] + usize::from(rc != hc);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[h.len()] as f32 / r.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ponytail: the pure helpers only, same line this repo draws everywhere else —
    // anything model-shaped needs a 459MB download and is covered by --selftest-fa.

    #[test]
    fn cer_counts_characters() {
        assert_eq!(cer("سلام", "سلام"), 0.0);
        assert_eq!(cer("", ""), 0.0);
        assert!((cer("abcd", "abxd") - 0.25).abs() < 1e-6);
        assert!((cer("abcd", "") - 1.0).abs() < 1e-6);
        // Insertions count against a short reference, so this exceeds 1.0.
        assert!(cer("ab", "abcdef") > 1.0);
    }

    #[test]
    fn tidy_only_trims() {
        assert_eq!(tidy("  از زندان  "), "از زندان");
        assert_eq!(tidy(""), "");
    }
}
