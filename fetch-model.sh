#!/bin/sh
# Fetch the Parakeet TDT weights into $STT_MODEL_DIR if they are not there.
# ADR-0045 Decision 1: the weights live on a node-local PVC, and this runs as an
# init container so an empty volume self-heals instead of failing the pod.
#
# Two things this gets right on purpose:
#
#   1. The sentinel is the WEIGHT FILE, not the directory. A mkdir that outran a
#      failed download leaves a directory that exists and a model that does not,
#      and the pod then starts and fails at ONNX session creation instead of
#      re-fetching.
#   2. Downloads land on `.part` and are renamed. A truncated 2.3GB file that
#      already has its final name looks complete to check 1 forever.
set -eu

# fp32 exports on purpose for both: quantisation is a CPU optimisation and
# typically runs SLOWER on the ORT CUDA execution provider, because of
# quantise/dequantise round-trips. ADR-0044 and ADR-0046 Decision 1.
fetch() {
  dir="$1"; repo="$2"; files="$3"
  base="https://huggingface.co/${repo}/resolve/main"

  complete=1
  for f in $files; do
    [ -s "$dir/$f" ] || complete=0
  done
  if [ "$complete" = 1 ]; then
    echo "already present in $dir"
    return 0
  fi

  echo "fetching $repo into $dir"
  mkdir -p "$dir"
  for f in $files; do
    if [ -s "$dir/$f" ]; then
      echo "  $f present"
      continue
    fi
    echo "  fetching $f"
    curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -o "$dir/$f.part" "$base/$f"
    mv "$dir/$f.part" "$dir/$f"
  done
}

# Batch / offline: Parakeet TDT 0.6B v3, ~2.49GB.
fetch "${STT_MODEL_DIR:-/models/tdt}" \
      "${TDT_REPO:-istupakov/parakeet-tdt-0.6b-v3-onnx}" \
      "encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx vocab.txt"

# Streaming: Nemotron 3.5 ASR Streaming 0.6B, ~2.59GB. ADR-0046 Decision 1.
#
# THE FILENAMES ARE THE CONSTRAINT, not the model. parakeet-rs loads exactly
# encoder.onnx / decoder_joint.onnx / tokenizer.model (model_nemotron.rs:80-81,
# nemotron.rs:384). Two of the four ONNX exports published for this model ship
# decoder.onnx + joint.onnx and vocab.json instead and simply will not load —
# including the FP16 one, which is otherwise the obvious pick. This repo and
# pantinor/nemotron-3.5-asr-streaming-0.6b-onnx are the two that match; this one
# is chosen because it carries the upstream LICENSE and NOTICE.
fetch "${STT_STREAM_MODEL_DIR:-/models/nemotron}" \
      "${NEMOTRON_REPO:-tonythethompson/Nemotron-3.5-ASR-Streaming-0.6B-ONNX}" \
      "encoder.onnx encoder.onnx.data decoder_joint.onnx tokenizer.model"

# Persian (ADR-0047). One file, no .onnx.data sidecar unlike the other two.
#
# Deliberately NOT fetching LICENSE: this function treats every listed file as a
# completeness sentinel under `set -eu`, so a cosmetic file would become a hard
# dependency of pod startup and an upstream rename would abort the init container.
# The licence ships in the image instead, under assets/.
fetch "${STT_FA_MODEL_DIR:-/models/shenava}" \
      "${SHENAVA_REPO:-PersianML/Shenava-Koochik-v1.0-tract-streaming}" \
      "model.onnx tokens.txt"

ls -la "${STT_MODEL_DIR:-/models/tdt}" "${STT_STREAM_MODEL_DIR:-/models/nemotron}" \
       "${STT_FA_MODEL_DIR:-/models/shenava}"
