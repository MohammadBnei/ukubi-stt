# ukubi-stt

GPU speech-to-text for `ukubi-cluster`. Rust, gRPC, node-pinned to the one
machine with an RTX 2070 SUPER.

Design and rationale live in the infra repo, not here:
[ADR-0044](https://github.com/MohammadBnei/infra-bootstrap/blob/main/docs/adr/0044-stt-grpc-service.md)
(this service) and
[ADR-0043](https://github.com/MohammadBnei/infra-bootstrap/blob/main/docs/adr/0043-gpu-node-enablement.md)
(the GPU it runs on).

## Status: Gate 0 — unproven

**There is no service here yet, on purpose.**

The whole design rests on one unverified assumption: that `parakeet-rs`'s CUDA
execution provider actually engages on a Turing card through this cluster's
container runtime. Until that is measured, writing the gRPC layer would be
building on sand — and specifically, an engine swap invalidates the *streaming
proto*, which is the one artefact that does not survive a change of engine. So
the proto is deliberately not written yet either.

`src/main.rs` is the whole repo: load the model, measure GPU memory across load
and warmup, decode, report a real-time factor, and **exit non-zero if the GPU
was never touched**.

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

## Running the gate

```bash
git tag 0.1.0 && git push --tags        # builds on the build-runner LXC
kubectl apply -f k8s/gate-pod.yaml
kubectl logs -f stt-gate
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

### Known unknowns the gate is there to find

1. **CUDA/cuDNN majors.** `ort 2.0.0-rc.13` links a prebuilt ONNX Runtime whose
   majors must match the base image. `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`
   is a guess. The host driver (580.173.02) covers CUDA 12.x and 13.x, so the
   driver is not the constraint — the ORT build is. Failure at *session
   creation* rather than at the memory assertion points here.
2. **Whether `ort` can find its CUDA libraries at runtime.** May need the
   `load-dynamic` feature or `ORT_DYLIB_PATH`.
3. **Whether `ParakeetTDT` is `Send`.** Not needed for this single-threaded
   binary, but the gRPC server's design depends on it — if it is not, the model
   needs a dedicated thread and a channel rather than a mutex.

## If the gate fails

Fall back, in order: Whisper via `ort` (same backend, same CUDA story, ~99
languages, offline-only, so streaming becomes VAD-chunked), then `whisper-rs`
(whisper.cpp — the most-trodden CUDA path in Rust), then `sherpa-rs` (archived
2026-03, but the one sherpa binding that ships a working `cuda` feature).

Not `sherpa-onnx`: its `build.rs` contains zero occurrences of `cuda` or `gpu`
and only ever downloads the CPU tarball, so `provider: Some("cuda")` links a
CPU-only runtime and decodes on CPU. Same class of trap, no escape hatch.

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

**Rollback** is redeploying an earlier tag — the registry keeps the last 3 plus
`latest` (ADR-0034), so anything older than that is gone and must be rebuilt.

## Local build

Not possible on a Mac, and not expected to be — the image needs CUDA. Builds run
on the `build-runner` LXC via the `image` workflow (ADR-0034: one runner
instance per repo, buildah under `sudo`, plain-HTTP push to
`registry.bnei.lan:5000`).
