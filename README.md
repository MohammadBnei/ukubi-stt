# ukubi-stt

GPU speech-to-text for `ukubi-cluster`. Rust, gRPC, node-pinned to the one
machine with an RTX 2070 SUPER.

Design and rationale live in the infra repo, not here:
[ADR-0044](https://github.com/MohammadBnei/infra-bootstrap/blob/main/docs/adr/0044-stt-grpc-service.md)
(this service) and
[ADR-0043](https://github.com/MohammadBnei/infra-bootstrap/blob/main/docs/adr/0043-gpu-node-enablement.md)
(the GPU it runs on).

## Status: live

Gate 0 passed on 2026-08-31 and the service has been in production since, with
two consumers: **dream-analyst** (`/dreams/new`) and **agent-fleet** (the
console composer). Streaming lands text roughly 560ms behind speech.

The gate itself is kept below, because it is the check to re-run after any
change to the image, the driver, the node or the `ort`/`parakeet-rs` versions —
and because the failure it guards against is silent.

### The assumption everything rests on

`parakeet-rs`'s CUDA execution provider must actually engage. If it does not,
the model still loads, still transcribes, and still returns correct text —
roughly 30x slower, with nothing in the logs that reads as an error.

### Why that assertion is not paranoia

`parakeet-rs` registers execution providers like this (`execution.rs`, 0.3.7):

```rust
ExecutionProvider::Cuda => builder.with_execution_providers([
    ort::ep::CUDA::default().build(),
    CPUExecutionProvider::default().build().error_on_failure(),
])?
```

`error_on_failure()` is on the **CPU** provider. If CUDA fails to initialise,
ORT falls through to CPU and the model loads, transcribes, and returns correct
text — roughly 30x slower, with nothing in the logs that reads as an error.

There is a second trap in front of that one: **the `cuda` cargo feature enables
the provider, it does not select it.** `from_pretrained(path, None)` gives a CPU
session no matter what was compiled in. Both halves are required:

```toml
parakeet-rs = { version = "0.3.7", features = ["cuda"] }   # enable
```
```rust
let cfg = ExecutionConfig::new()
    .with_execution_provider(ExecutionProvider::Cuda);       // select
ParakeetTDT::from_pretrained(dir, Some(cfg))?
```

## Releasing

Same `release-it` flow as the other build repos (`editable-blog`,
`agent-fleet`, `wedding-2026`) — conventional-changelog, angular preset,
`CHANGELOG.md` written in the release commit:

```bash
bun install     # or npm i
bun run release
```

**`package.json` holds the canonical version and an `after:bump` hook rewrites
`Cargo.toml` to match.** Two files, one source of truth, because `image.yml`
reads the version out of `Cargo.toml` and compares it to the pushed tag — if
they drift the build fails with "does not match the tag", which is the right
failure but an annoying one to debug.

`tagName` is the bare `${version}`, not release-it's default `v${version}`, for
the same reason: `image.yml` compares it to `github.ref_name` verbatim.

The tag push is what builds and pushes the image. There is no deploy step in
the workflow — ArgoCD picks up `helm/values.yaml` from `main`, so bumping the
running version is a separate `image.tag` edit there.

## Running the gate

The startup assertion runs on every boot, but `--selftest` is the manual
version — ADR-0044 Decision 3 keeps it as a flag, not a build gate, because
there is no point in the build where a GPU is reachable:

```bash
kubectl exec deploy/ukubi-stt -- ukubi-stt --selftest            # synthetic
kubectl exec deploy/ukubi-stt -- ukubi-stt --selftest /tmp/a.wav # real audio
```

Pass looks like:

```
gpu.used before load : 4 MiB
model loaded in      : 6.2s
gpu.used after warmup: 1843 MiB (delta 1839 MiB)
real-time factor     : 0.041
GATE PASSED: CUDA engaged (1839 MiB resident), RTF 0.041
```

A delta under 128 MiB is a failure regardless of how good the transcript looks.

### What the first run found (2026-08-31)

The gate ran and **failed**, exactly as designed to:

```
gpu.used before load : 1 MiB
model loaded in      : 2.4s
gpu.used after warmup: 1 MiB (delta 0 MiB)
real-time factor     : 0.081
GATE FAILED: GPU memory grew by only 0 MiB (< 128 MiB)
```

An RTF of 0.081 is 12x faster than realtime and reads as a healthy GPU result.
Without the memory assertion this would have shipped as a working GPU service
decoding entirely on CPU. `nvidia-smi` answered *inside the container*, so the
RuntimeClass, device plugin and `nvidia.com/gpu` request were all correct — the
fault was above them.

Two independent bugs, both in the Dockerfile, both now fixed:

1. **`libonnxruntime_providers_cuda.so` was never in the image.** ONNX Runtime
   is linked statically (`ldd` on the binary shows no `libonnxruntime`), but its
   CUDA provider is not part of that archive — it is a separate 79MB shared
   object ORT dlopens *next to the calling module*, i.e. next to the executable.
   The build produced it and only the binary was copied out.

   Compounding it: ort-sys's `copy-dylibs` **symlinks** these into
   `target/release/` from `~/.cache/ort.pyke.io/dfbin/`, so a naive
   `COPY --from=build target/release/*.so` lands dangling links and fails
   identically. `cp -L` first.

2. **The CUDA major was wrong, and could only ever have been wrong.** ort-sys
   2.0.0-rc.13 does not build ONNX Runtime — it downloads a prebuilt one chosen
   from a hardcoded table (`build/download/dist.tsv`). For
   `x86_64-unknown-linux-gnu` that table holds four rows: no-features, `webgpu`,
   `nvrtx`, and `cuda13,tensorrt,nvrtx`. **No CUDA 12 build exists for Linux**,
   and the resolver's own fallback comment says `"guessing 13"`. Building in a
   CUDA 12.6 image therefore produced a binary wanting `libcudart.so.13`.

   Its real dependency set, read off the provider rather than assumed:
   `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcurand.so.10`,
   `libcuda.so.1` — plus `libcudnn.so.9` and `libcufft.so.12` dlopened lazily by
   name, which is why the runtime image keeps the `-cudnn-` variant even though
   nothing links cuDNN. Driver 580.173.02 is a CUDA 13.0 driver, and the
   provider's embedded arch list carries `sm_75`, so Turing is covered.

`ORT_CUDA_VERSION=13` is now set in the builder so that resolution is read
rather than guessed, and the binary logs ORT's own `debug` events — which
report which provider was declined and why — instead of dropping them.

### Still unknown

**Whether `ParakeetTDT` is `Send`.** Not needed for this single-threaded
binary, but the gRPC server's design depends on it — if it is not, the model
needs a dedicated thread and a channel rather than a mutex. Compiling this
binary does not exercise it.

## If the gate fails

Fall back, in order: Whisper via `ort` (same backend, same CUDA story, ~99
languages, offline-only, so streaming becomes VAD-chunked), then `whisper-rs`
(whisper.cpp — the most-trodden CUDA path in Rust), then `sherpa-rs` (archived
2026-03, but the one sherpa binding that ships a working `cuda` feature).

Not `sherpa-onnx`: its `build.rs` contains zero occurrences of `cuda` or `gpu`
and only ever downloads the CPU tarball, so `provider: Some("cuda")` links a
CPU-only runtime and decodes on CPU. Same class of trap, no escape hatch.

## Integrating a consumer

Both existing consumers were built this way, and the shape is not optional if
your consumer has users.

### Architecture: your backend proxies, the browser holds nothing

```
browser ──(your app's own session cookie, SAME-ORIGIN)──▶ your backend ──(STT_TOKEN_<YOU>)──▶ ukubi-stt
```

**No browser ever holds an STT credential.** Handing one to the page would give
every user of your app a credential for the GPU, recoverable from devtools, and
`stt.bnei.dev` appears in Certificate Transparency logs minutes after issuance —
so the endpoint is not obscure.

It also happens to be the only shape that works. Both consumers discovered the
same wall independently: their APIs allow no CORS and their session cookies are
`SameSite=Lax`, so a cross-origin call from the page carries no identity at all.
There is nothing to relax — the proxy is the design.

The pay-off beyond credentials: **the `session_id` stops being client-chosen.**
Derive it server-side from the authenticated user (an HMAC over identity plus a
per-dictation id) and one user cannot interleave audio into another's
recognizer. ADR-0046 accepted that hazard when every caller shared one token;
this closes it.

### Step 1 — a token

Add `STT_TOKEN_<YOURAPP>` to Infisical `ukubi-stt-bhr-m` (env `dev`). The
service scans its whole environment for `STT_TOKEN_*` at startup, so adding a
caller is adding a secret — no redeploy, no code change — and revoking one
caller does not revoke the others. The matched name is logged with every
request as `client=`, which is the only reason a leak can be attributed to one
consumer rather than all of them.

**Then copy the value into your own Infisical project.** This is deliberate and
it is *not* what a first instinct suggests. Granting your app's identity read on
`ukubi-stt-bhr-m` looks tidier and avoids a second copy to rotate — but an
`InfisicalSecret` syncs the **whole project env** into your namespace. Doing
that for dream-analyst put `REGISTRY_PASSWORD` — push rights on the registry
every node in the cluster pulls from — into its namespace. `secretsScope`
narrows what is *synced* but not what the identity may *read*. Two copies to
rotate is the smaller problem.

### Step 2 — the server side

Address it at **`ukubi-stt.ukubi-stt.svc.cluster.local:9090`**, plaintext h2c.
Not `stt.bnei.dev`. In-cluster skips TLS, the ingress and the gRPC-Web
translation, and the whole edge — CORS, rate limiting, idle timeouts — stops
being your problem. Keep the address in an env var so local development can
point at the public host.

Make the dependency **optional**. Both consumers log and disable the feature if
the token is absent or the dial fails, and answer `UNAVAILABLE` thereafter. A
service that refuses to start without STT trades a working product for a broken
deployment, on a single-replica pod whose node reboots for gaming.

Map errors before they reach your users: `RESOURCE_EXHAUSTED` (session cap) and
`FAILED_PRECONDITION` (streaming model not loaded) are both "try later", not
"you did something wrong".

If you expose this to your own frontend over gRPC, use **chunked unary, not
server-streaming.** A browser cannot stream *up* under any transport gRPC
offers, so audio arrives as discrete requests regardless — and each chunk's text
comes back in that chunk's own response. A server stream would add lifecycle,
reconnect machinery and a cursor to deliver one message per request already
made.

### Step 3 — the browser side

**Vendor `web/stt-capture.js`.** Do not import it from `https://stt.bnei.dev` —
a cross-origin import makes your microphone break whenever this node reboots,
turning a degraded feature into a broken page. Copy it, keep the header naming
the origin, and re-copy when it changes. Drift is the accepted cost.

The module is transport-agnostic: it emits 16 kHz mono s16 PCM chunks and calls
your `send(pcm, last)`. It knows nothing about gRPC, tokens or origins, which is
why two consumers with different backends use it unmodified.

```js
import { createDictation, prewarm } from './stt-capture.js';

// On hover/focus — builds the AudioContext and compiles the worklet.
// Touches no device, so it prompts for nothing.
button.addEventListener('pointerenter', () => void prewarm());

const dictation = createDictation({
  send: (pcm, last) => postToYourOwnBackend(pcm, last),
  onError: (e) => { /* abandon; do NOT retry the chunk */ },
});
await dictation.start();
// ...later
await dictation.stop();   // flushes the tail and waits for it to land
```

**Two contracts, both load-bearing:**

*Chunks are strictly ordered and never concurrent.* The encoder carries cache
forward, so a re-sent or reordered chunk corrupts everything after it. The
module serialises sends through a promise chain — that is a **contract, not an
implementation detail**. A consumer that "optimises" it into parallel sends
corrupts every transcript after the first reorder. On any error, abandon: send
nothing further, keep the text already appended, surface the failure. The
orphaned recognizer is swept after 120s idle.

*Never retry a streaming chunk.* Same reason.

## Caveats, all of them learned the hard way

Every one of these cost real time. They are grouped by who hits them.

### If you are changing this service

**CUDA fails open, not closed.** `parakeet-rs` puts `error_on_failure()` on the
*CPU* provider, so a CUDA failure falls through to CPU silently and returns
correct text ~30x slower. And the `cuda` cargo feature *enables* the provider,
it does not *select* it — `from_pretrained(path, None)` gives you a CPU session
no matter what was compiled in. Both halves are required. Run the gate.

**The provider is a separate 79MB `.so` that must be beside the executable.**
It is dlopened at runtime, not linked. Copy it with `cp -L` — the build copies
*symlinks* on Unix, and a dangling symlink in the image is a runtime CPU
fallback, i.e. the silent failure above.

**CUDA major version must match the base image.** `ort-sys` ships no cuda12
Linux build, which is why the image is on CUDA 13 rather than 12.6.

**Turn ORT's logging down.** At `info` it drowns the startup lines you actually
need in BFCArena allocation noise; only ~10 lines survived rotation when the
first real diagnosis was needed.

### If you are writing a client

**`MediaRecorder` cannot do streaming.** With a `timeslice` it emits WebM
*cluster fragments* that are **not independently decodable**, so a live path
cannot reuse a batch recording route. `AudioContext({sampleRate: 16000})` plus
an AudioWorklet is the only option — and forcing the context rate is what
removes the need to write a resampler.

**Chunk arithmetic is where the bugs live.** Three separate latency bugs shipped
during 0.5.x, none visible in review: shipping the whole buffer sent 768ms
chunks and left the server holding a remainder (~1.4s and irregular instead of
~650ms); a 2048-frame callback added up to 128ms of jitter; and the tail flush
sent zero samples when the buffer landed exactly empty — 1 callback in 35, and
*every* Stop before speaking — which the server then rejected, losing the last
words and leaking the session. This is why the module exists once and is
vendored rather than reimplemented.

**An empty final chunk is a valid close.** Do not filter it out.

**The first words go missing if you build the graph on click.** Fetching the
module, constructing a context, compiling a worklet and *then* opening the
device is several hundred milliseconds of speech nobody captured. Call
`prewarm()` on hover, and note `getUserMedia` was originally serialised behind
`addModule()` for no reason — they are independent and the device is the slow
one.

**A context built outside a user gesture starts suspended, and a suspended
context runs no worklet.** A naive prewarm therefore looks live and records pure
*silence* — worse than the bug it fixes, because nothing errors. `start()`
resumes it, which is permitted because the click is what reached it. Also: never
memoise a *failed* warm, or the button is dead until reload.

**Show an arming state.** Even fully warmed, `getUserMedia` is 100-300ms. A
button that looks ready while the graph is still being built invites exactly the
words that go missing. Prewarming shrinks the gap; showing it stops the gap
costing a sentence. Pre-opening the microphone would close it entirely and is
deliberately not done — it leaves the recording indicator lit on a page the user
has not asked to record on.

**Append each chunk verbatim. Never trim one, never add your own separator.**
The leading space *is* the word boundary. SentencePiece marks a word-INITIAL
piece, the detokeniser renders that mark as a leading space, and a chunk that
continues a word therefore arrives without one. Trimming each chunk and joining
with a space turns `" bon"` + `"jour"` into `"bon jour"` — the word splits, and
it looks like a model failure rather than a client bug. Concatenation is the
whole protocol: `transcript += r.text`, with no guard, because a chunk that is
nothing but a separator has to survive too. The reference page got this wrong
until 0.10.3; if it and this paragraph ever disagree again, this paragraph wins.

**Append with a functional state update.** In React,
`onChange(value + text)` captures `value` from the render that started the
recording, so every chunk appends to the same stale string and visibly
overwrites the last. `onChange(prev => prev + text)` reads current state and
captures nothing. A latest-callback ref fixes the stale render but *not* two
chunks landing in the same tick, which the tail flush does.

**Do not send chunks through a framework's RPC serialisation.** SvelteKit's
remote functions devalue the argument and then base64 the whole JSON string, so
the expansion compounds: `v.array(v.number())` is ~4.9x, even
`v.instance(Uint8Array)` is ~1.78x, a raw `application/octet-stream` body is
1.0x. At two chunks per second that penalty is continuous.

### If you are operating it

**One batch decode at a time, cluster-wide.** The offline path holds a
`Semaphore(1)` and a second caller gets `RESOURCE_EXHAUSTED` *immediately* —
there is no queue, deliberately, because a queue on a single GPU turns overload
into unbounded latency. Streaming is different: `STT_MAX_SESSIONS` concurrent
sessions (default 8), since each occupies the GPU for only 20-50ms per 560ms
chunk.

**Rotating any token restarts the pod.** `autoReload: true` plus
`strategy: Recreate`. Sessions are in-memory in a single replica, so every
in-flight dictation loses its encoder cache and resumes mid-sentence with no
error the user can see. **Adding a consumer's token is a maintenance action, not
a routine one.**

**This service is best-effort and will disappear.** Single replica, pinned to
the one node with a GPU, no HA and no CPU fallback — all deliberate (ADR-0044
Context), including that `.165` reboots for gaming and takes STT with it.
**Treat `UNAVAILABLE` as normal and degrade.** If a consumer genuinely needs
uptime, that reopens ADR-0044's availability decision rather than being worked
around in the client.

**Streaming accuracy is visibly below the offline model** ("UQB cluster" for
"Yukie cluster"). Fine for dictation into an editable box; not for anything
acting on the transcript unreviewed.

**Releases are cut by hand.** There is no release workflow here — `image.yml`
triggers on `push: tags` and nothing creates a tag. Run `bun run release` (see
Releasing above). Merging to `main` builds nothing.

## Reference

### Transports

| caller | address | notes |
|---|---|---|
| another pod on ukubi-cluster | `ukubi-stt.ukubi-stt.svc.cluster.local:9090` | plaintext h2c — no TLS, no Traefik, ~1ms of network. **Use this.** |
| a machine on LAN/WAN | `stt.bnei.dev:443` | TLS, native gRPC |
| a browser | `https://stt.bnei.dev` | gRPC-Web; `web/index.html` is a working reference client |

### Discovering the API

Reflection is enabled, so nothing needs a vendored `.proto`:

```bash
grpcurl -plaintext ukubi-stt:9090 list
grpcurl -plaintext ukubi-stt:9090 describe stt.v1.Stt.Recognize
```

**In-cluster only, by construction.** The IngressRoute matches
`PathPrefix(/stt.v1.Stt/)`; reflection answers on `/grpc.reflection.v1.*`, so
Traefik 404s it from outside. External callers need the proto from
`proto/stt/v1/stt.proto`.

### Credentials

```
STT_TOKEN_<NAME>   one per caller, e.g. STT_TOKEN_FLEET, STT_TOKEN_DREAMER
STT_AUTH_TOKEN     the original, still accepted, reported as client "default"
```

The comparison loop deliberately has **no early exit** — it checks every
configured token even after a match, so response time is not a function of a
token's position in the map.

### Making a call

```bash
grpcurl -plaintext -H "authorization: Bearer $TOKEN" \
  -d '{"config":{"sampleRateHertz":16000},"audio":"<base64 PCM>"}' \
  ukubi-stt:9090 stt.v1.Stt/Recognize
```

Audio is **16 kHz mono little-endian s16 PCM**, raw — no container, no encoding
field. Anything else is rejected rather than resampled, because a silently
resampled request returns a plausible transcript and a meaningless real-time
factor.

Leave `session_id` empty for a one-shot decode of a whole utterance. Set it (and
send ~560ms chunks in order, `last: true` on the final one) for realtime — see
ADR-0046.

## Models

| | Repo | Files |
|---|---|---|
| Batch (now) | [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt` |
| Streaming (Phase E) | `altunenes/parakeet-rs` → `nemotron-3.5-asr-streaming-0.6b-onnx` | `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model` |

fp32 exports, not int8: int8 is a CPU optimisation and typically runs *slower*
on the ORT CUDA provider because of quantise/dequantise round-trips.

Batch and streaming are two different models, both GPU-resident. Budget both
against 8GB before Phase E.

## CI

Two workflows, deliberately split by what they can prove:

| | runs on | proves |
|---|---|---|
| `ci.yml` | every PR/push, GitHub-hosted | fmt, clippy, tests, that it compiles |
| `image.yml` | tags only, build-runner LXC | that it builds into a CUDA image and pushes |

`ci.yml` exists because `image.yml` costs ~15 minutes on the LXC pulling a
multi-GB CUDA tree, and a syntax error should not cost that. It cannot prove
CUDA engages — no hosted runner has an RTX 2070, which is the whole reason
Gate 0 runs on real hardware.

`cargo audit` is advisory and non-blocking on purpose: this depends on an `ort`
release candidate by design, so a clean audit is not achievable on demand and a
permanently-red check trains people to ignore it.

**Rollback** is redeploying an earlier tag — and this repo's retention is
tighter than the cluster default: zot keeps the last **2** tags plus `latest`
for `ukubi-stt` (3 everywhere else), because the image is a ~5GB CUDA tree.
Anything older is GC'd from Garage and must be rebuilt.

## Local build

Not possible on a Mac, and not expected to be — the image needs CUDA. Builds run
on the `build-runner` LXC via the `image` workflow (ADR-0034: one runner
instance per repo, buildah under `sudo`, plain-HTTP push to
`registry.bnei.lan:5000`).
