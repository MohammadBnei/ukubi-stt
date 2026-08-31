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

It compiles, is clippy-clean under `-D warnings`, and its two unit tests pass
in CI. That says nothing about whether CUDA engages — no hosted runner has an
RTX 2070 — which is precisely what the gate is for.

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
git tag 0.2.0 && git push --tags        # builds on the build-runner LXC (tag == Cargo.toml version)
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
