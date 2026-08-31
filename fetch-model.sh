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

DIR="${STT_MODEL_DIR:-/models/tdt}"
REPO="${TDT_REPO:-istupakov/parakeet-tdt-0.6b-v3-onnx}"
BASE="https://huggingface.co/${REPO}/resolve/main"

# fp32 export on purpose: int8 is a CPU optimisation and typically runs slower
# on the ORT CUDA execution provider because of quantise/dequantise round-trips.
FILES="encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx vocab.txt"

complete=1
for f in $FILES; do
  [ -s "$DIR/$f" ] || complete=0
done
if [ "$complete" = 1 ]; then
  echo "model already present in $DIR"
  ls -la "$DIR"
  exit 0
fi

echo "fetching Parakeet TDT weights into $DIR from $REPO"
mkdir -p "$DIR"
for f in $FILES; do
  if [ -s "$DIR/$f" ]; then
    echo "  $f present"
    continue
  fi
  echo "  fetching $f"
  curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -o "$DIR/$f.part" "$BASE/$f"
  mv "$DIR/$f.part" "$DIR/$f"
done
ls -la "$DIR"
