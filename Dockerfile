# Gate 0 image (ADR-0044 Decision 1). Built on the build-runner LXC, never
# in-cluster — buildah needs CAP_SYS_ADMIN and ADR-0034 keeps that off the
# cluster.
#
# THE CUDA/cuDNN MAJORS HERE ARE THE GATE'S FIRST UNKNOWN.
# ort 2.0.0-rc.13 links a prebuilt ONNX Runtime whose CUDA and cuDNN majors must
# match these images, and the host driver must satisfy that CUDA major.
# k8s-worker-01 runs 580.173.02, which covers CUDA 12.x and 13.x, so the driver
# is not the constraint — the ORT build is. CUDA 12.6 + cuDNN 9 is the current
# best guess. If the gate fails at ONNX session creation rather than at the
# memory assertion, these two lines are the first thing to change.
ARG CUDA_VERSION=12.6.3
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


FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-${UBUNTU_VERSION}

# ca-certificates for the model download; python3-pip only to get `hf`, the
# supported way to pull the ONNX weights. pip is removed again below — the
# model layer is ~2.5GB and this image is already large enough.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates python3 python3-pip \
 && pip3 install --no-cache-dir "huggingface_hub[cli]" \
 && apt-get purge -y python3-pip && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Parakeet TDT 0.6B v3 — the offline/batch model, 25 languages with auto
# detection. Streaming (Phase E) needs a SECOND model and is not fetched here.
#
# fp32 export on purpose: int8 is a CPU optimisation and typically runs slower
# on the ORT CUDA execution provider because of quantise/dequantise round-trips.
ARG TDT_REPO=istupakov/parakeet-tdt-0.6b-v3-onnx
RUN hf download "${TDT_REPO}" \
      encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx vocab.txt \
      --local-dir /models/tdt \
 && ls -la /models/tdt

COPY --from=build /src/target/release/ukubi-stt /usr/local/bin/ukubi-stt

ENV STT_MODEL_DIR=/models/tdt
ENTRYPOINT ["/usr/local/bin/ukubi-stt"]
