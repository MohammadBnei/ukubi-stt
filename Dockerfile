# ukubi-stt image. ADR-0044 (the service) and ADR-0045 (what is and is not in
# here). Built on the build-runner LXC, never in-cluster — buildah needs
# CAP_SYS_ADMIN and ADR-0034 keeps that off the cluster.
#
# WHAT IS IN THIS IMAGE, AND WHY EACH PART IS
#   - the binary                     obviously
#   - the ORT CUDA provider .so      ORT dlopens it next to the executable
#   - the CUDA runtime + cuDNN       the ABI pin; see ADR-0045 Decision 2
#   - fetch-model.sh                 so the init container can self-heal a PVC
# and, deliberately, NOT the ~2.4GB of model weights. They live on a node-local
# PVC (ADR-0045 Decision 1) because they are an immutable third-party artifact
# that changes when upstream publishes, while the binary changes when we do.

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
# Driver 580.173.02 on k8s-worker-01 is a CUDA 13.0 driver (>= 580.65.06
# required), and CUDA 13.0 still supports Turing sm_75 — verified in the
# provider's embedded arch list, which carries sm_75.
ARG CUDA_VERSION=13.0.3
# -cudnn- is load-bearing and NOT visible in linkage:
# libonnxruntime_providers_cuda.so has no DT_NEEDED entry for cuDNN, it dlopens
# `libcudnn.so.9` by name at first use. Gate 0's log settles it —
# `INFO ort::logging: cuDNN version: 91400`. Dropping to the plain -runtime
# variant looks like a free ~1.5GB and is not. ADR-0045 Decision 3.
#
# Its real DT_NEEDED set is libcudart.so.13, libcublas.so.13, libcublasLt.so.13,
# libcurand.so.10 and libcuda.so.1 — the last injected by the container toolkit,
# and the only library that correctly comes from outside this image.
ARG UBUNTU_VERSION=ubuntu24.04

# THE BUILDER NEEDS NO CUDA. ADR-0045 Decision 4.
# It used to be nvidia/cuda:*-cudnn-devel at 8.23GB, on the strength of a
# comment that said ort's `cuda` feature "may want CUDA headers at build time".
# It does not: ort-sys downloads a prebuilt ONNX Runtime, links
# libonnxruntime.a statically, and the CUDA provider is dlopened rather than
# linked. No nvcc, no headers, nothing to compile against.
#
# ubuntu:24.04 matches the runtime's distro, so the glibc rule that forced
# 24.04 in the first place still holds: the ONNX Runtime binary ort downloads
# is built against a newer toolchain than 22.04 ships — linking there fails
# with `undefined symbol: __isoc23_strtoll` (glibc 2.38+) and `_M_replace_cold`
# (libstdc++ 13+), while 22.04 has glibc 2.35 / libstdc++ 12.
FROM ubuntu:24.04 AS build

# Read, do not guess. ort-sys sniffs NV_CUDA_CUDART_VERSION, CUDA_HOME and
# `nvcc --version` for a CUDA 13 signature and falls back to "guessing 13" when
# none matches. On a non-CUDA builder none of them exist, so the guess is the
# only path — stating it makes the resolution explicit rather than incidental,
# and makes a future CUDA 14 row in dist.tsv a one-line change.
ENV ORT_CUDA_VERSION=13

# libssl-dev: something in the parakeet-rs/ort tree pulls openssl-sys, whose
#   build script fails with "Package openssl was not found in the pkg-config
#   search path". GitHub's ubuntu-latest runners ship it preinstalled, so
#   `cargo clippy` in ci.yml passes without it — a bare base image does not,
#   which is exactly the kind of gap a hosted CI check cannot catch for you.
# protobuf-compiler: tonic-prost-build shells out to protoc.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates pkg-config libssl-dev protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /src
# Dependencies first, so a source-only edit does not rebuild the parakeet-rs/ort
# tree. Cargo.lock is copied when present; on the very first build it does not
# exist yet, which is why this is not `--locked`.
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
 && printf 'fn main() {}\n' > build.rs \
 && cargo build --release \
 && rm -f build.rs
COPY build.rs ./
COPY proto ./proto
COPY web ./web
# include_str!'d by src/fbank.rs (the mel filterbank and its golden fixture).
# Missing this line compiles fine locally and fails only in the tag build, which
# is a 15-minute round trip to learn about a one-line omission.
COPY assets ./assets
COPY src ./src
# cargo fingerprints on content+mtime; the stub above already produced a binary,
# so touch to force the real sources to compile over it.
# include_str!("../web/index.html") makes the page a compile-time input, so a
# page-only edit still rebuilds the binary. That is the intended trade: the
# alternative is a second artefact to deploy and keep in step with the proto.
RUN touch src/main.rs build.rs && cargo build --release

# The ORT CUDA provider is NOT part of libonnxruntime.a. It is a separate 79MB
# shared object that ORT dlopens on first use, resolved against the directory of
# the calling module — for a static link, the directory of the executable. On
# 2026-08-31 it was absent from the image and ORT fell silently back to CPU.
#
# `cp -L` is mandatory. ort-sys's `copy-dylibs` does not copy on Unix — it
# SYMLINKS these into target/release/ from ~/.cache/ort.pyke.io/dfbin/. A plain
# `COPY --from=build` of the symlink lands a dangling link in the runtime image,
# which fails exactly the same way as the file being missing.
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

# curl + ca-certificates are for fetch-model.sh, which runs from THIS image as
# an init container. Reusing our own image rather than pulling curlimages/curl
# costs nothing — it is already on the node — and avoids one more upstream
# dependency for four static file downloads.
#
# NOT python3/pip and the `hf` CLI, which is what the HuggingFace docs suggest:
# on 24.04 `pip3 install` into the system interpreter fails with PEP 668's
# "externally-managed-environment", and working around that means carrying
# ~150MB of Python to fetch four plain HTTP objects.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# The providers go NEXT TO the binary, not in a lib directory, and that is the
# whole point: ORT resolves them relative to the calling module's own path
# (dladdr -> dirname), so /usr/local/lib would never be looked at.
COPY --from=build /src/target/release/ukubi-stt /usr/local/bin/ukubi-stt
COPY --from=build /ortlib/libonnxruntime_providers_shared.so /usr/local/bin/
COPY --from=build /ortlib/libonnxruntime_providers_cuda.so /usr/local/bin/
COPY fetch-model.sh /usr/local/bin/fetch-model.sh

ENV STT_MODEL_DIR=/models/tdt
EXPOSE 8080 9090
ENTRYPOINT ["/usr/local/bin/ukubi-stt"]
