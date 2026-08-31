# Gate 0 image (ADR-0044 Decision 1). Built on the build-runner LXC, never
# in-cluster — buildah needs CAP_SYS_ADMIN and ADR-0034 keeps that off the
# cluster.
#
# THE CUDA MAJOR IS 13, AND THAT IS NOT A CHOICE — IT IS A CONSTRAINT.
# ort-sys 2.0.0-rc.13 does not build ONNX Runtime; it downloads a prebuilt one
# from pyke's CDN, chosen from a hardcoded table (`build/download/dist.tsv`).
# For x86_64-unknown-linux-gnu that table has exactly four rows: no-features,
# `webgpu`, `nvrtx`, and `cuda13,tensorrt,nvrtx`. **There is no CUDA 12 build
# for Linux.** Its own resolver says so out loud:
#
#     _ => { log::debug!("couldn't determine CUDA version, guessing 13");
#            "cuda13" } // "fallback" to the lowest version we ship
#
# So the first version of this file — CUDA 12.6.3 — produced a binary linked
# against an ONNX Runtime whose CUDA provider wants libcudart.so.13. That is
# the first of the two bugs that made Gate 0 fail on 2026-08-31.
#
# ORT_CUDA_VERSION is set below so the resolver reads the answer instead of
# guessing it. Driver 580.173.02 on k8s-worker-01 is a CUDA 13.0 driver
# (>= 580.65.06 required), and CUDA 13.0 still supports Turing sm_75 — verified
# directly in the provider's embedded arch list, which carries sm_75.
ARG CUDA_VERSION=13.0.3
# -cudnn- is load-bearing on the RUNTIME image and not obvious from linkage:
# libonnxruntime_providers_cuda.so has no DT_NEEDED entry for cuDNN, it dlopens
# `libcudnn.so.9` by name at first use. Its real DT_NEEDED set is
# libcudart.so.13, libcublas.so.13, libcublasLt.so.13, libcurand.so.10 and
# libcuda.so.1 (the last injected by the container toolkit).
#
# ubuntu24.04, not 22.04. The ONNX Runtime binary that ort downloads is built
# against a NEWER toolchain than 22.04 ships: linking on 22.04 fails with
# `undefined symbol: __isoc23_strtoll` (glibc 2.38+) and
# `_M_replace_cold` (libstdc++ 13+), while 22.04 has glibc 2.35 / libstdc++ 12.
# 24.04 is glibc 2.39 / libstdc++ 14 and links clean.
#
# Note the usual glibc rule — build on an image no newer than the runtime — is
# necessary but not sufficient here. It constrains OUR binary against the
# runtime; it says nothing about a third-party prebuilt demanding newer than
# both. Builder and runtime are pinned to the same version below, so that
# rule holds regardless.
ARG UBUNTU_VERSION=ubuntu24.04

# Builder is the -devel variant of the SAME base as the runtime, not rust:1-*.
# Two reasons, both learned the hard way elsewhere:
#   - glibc. rust:1-bookworm is glibc 2.36; ubuntu22.04 is 2.35. A binary linked
#     against the newer one does not run on the older one, and the failure is an
#     unhelpful symbol-lookup error at exec time, not at build time.
#   - ort's `cuda` feature may want CUDA headers present at build time; -devel
#     has them and -runtime does not.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-${UBUNTU_VERSION} AS build

# Read, do not guess. ort-sys sniffs NV_CUDA_CUDART_VERSION, CUDA_HOME and
# `nvcc --version` for a CUDA 13 signature and falls back to "guessing 13" when
# none matches — which means a CUDA 12 builder silently produces a CUDA 13
# binary and says nothing. Stating it here makes the resolution explicit and
# makes a future CUDA 14 row in dist.tsv a one-line change.
ENV ORT_CUDA_VERSION=13

# libssl-dev is required and its absence is not obvious: something in the
# parakeet-rs/ort tree pulls openssl-sys, whose build script fails with
# "Package openssl was not found in the pkg-config search path". GitHub's
# ubuntu-latest runners ship libssl-dev preinstalled, so `cargo clippy` in
# ci.yml passes without it — the CUDA base image does not, which is exactly
# the kind of gap a hosted CI check cannot catch for you.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates pkg-config libssl-dev \
 && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /src
# Dependencies first, so a source-only edit does not rebuild the parakeet-rs/ort
# tree. Cargo.lock is copied when present; on the very first build it does not
# exist yet, which is why this is not `--locked`. Commit the lock file that the
# first green build produces and this becomes reproducible.
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release
COPY src ./src
# cargo fingerprints on content+mtime; the stub above already produced a binary,
# so touch to force the real main.rs to compile over it.
RUN touch src/main.rs && cargo build --release

# THE SECOND BUG THAT FAILED GATE 0: these two files were never shipped.
#
# ONNX Runtime is linked statically (`libonnxruntime.a`, hence no libonnxruntime
# in `ldd`), but its CUDA execution provider is NOT part of that archive. It is
# a separate shared object that ORT dlopens by name on first use, resolved
# against the directory of the calling module — for a static link, the directory
# of the executable itself. If it is absent, provider registration fails, ORT
# falls through to CPU, and parakeet-rs says nothing because `error_on_failure()`
# sits on the CPU provider, not on CUDA.
#
# `cp -L` is mandatory. ort-sys's `copy-dylibs` feature does not copy anything on
# Unix — it SYMLINKS these into target/release/ from ~/.cache/ort.pyke.io/dfbin/.
# A plain `COPY --from=build` of the symlink lands a dangling link in the runtime
# image, which fails exactly the same way as the file being missing.
#
# Named explicitly rather than globbed so a rename upstream is a build failure
# here, not a silent CPU fallback in production. The tensorrt/nvrtx providers
# from the same distribution are deliberately left behind — they need libnvinfer
# and nothing asks for them.
RUN mkdir -p /ortlib \
 && cp -L target/release/libonnxruntime_providers_shared.so \
          target/release/libonnxruntime_providers_cuda.so /ortlib/ \
 && ls -la /ortlib


FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-${UBUNTU_VERSION}

# curl + ca-certificates only. NOT python3/pip and the `hf` CLI, which is what
# the HuggingFace docs suggest: on 24.04 `pip3 install` into the system
# interpreter fails with PEP 668's "error: externally-managed-environment", and
# working around that (--break-system-packages, or a venv) means carrying ~150MB
# of Python in the runtime image to fetch four static files. The weights are
# plain HTTP objects; curl fetches them and the failure class disappears.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Parakeet TDT 0.6B v3 — the offline/batch model, 25 languages with auto
# detection. Streaming (Phase E) needs a SECOND model and is not fetched here.
#
# fp32 export on purpose: int8 is a CPU optimisation and typically runs slower
# on the ORT CUDA execution provider because of quantise/dequantise round-trips.
#
# --retry because encoder-model.onnx.data is ~2.3GB and a truncated download
# would produce an image that loads a corrupt model at runtime rather than
# failing here. -f so an HTTP error is a build failure, not a 0-byte file.
ARG TDT_REPO=istupakov/parakeet-tdt-0.6b-v3-onnx
ARG HF_BASE=https://huggingface.co/${TDT_REPO}/resolve/main
RUN mkdir -p /models/tdt && cd /models/tdt \
 && for f in encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx vocab.txt; do \
      echo "fetching $f" && \
      curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -o "$f" "${HF_BASE}/$f" || exit 1; \
    done \
 && ls -la /models/tdt

# The providers go NEXT TO the binary, not in a lib directory, and that is the
# whole point: ORT resolves them relative to the calling module's own path
# (dladdr -> dirname), so /usr/local/lib would not be looked at.
COPY --from=build /src/target/release/ukubi-stt /usr/local/bin/ukubi-stt
COPY --from=build /ortlib/libonnxruntime_providers_shared.so /usr/local/bin/
COPY --from=build /ortlib/libonnxruntime_providers_cuda.so /usr/local/bin/

ENV STT_MODEL_DIR=/models/tdt
ENTRYPOINT ["/usr/local/bin/ukubi-stt"]
