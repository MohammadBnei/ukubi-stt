# Bundled assets

Compiled into the binary with `include_str!`. These are constants of the
*algorithm*, not model weights — weights live on the PVC per ADR-0045, and putting
these there instead would create an old-PVC/new-binary skew whose symptom is
silently wrong numbers.

| file | source | why it is here |
|---|---|---|
| `mel_filters_slaney_80x257.json` | [`PersianML/Shenava-Koochik-v1.0-ONNX-fp16`](https://huggingface.co/PersianML/Shenava-Koochik-v1.0-ONNX-fp16), rev `41603e4beed9c889700e6367a26be1d670d74cc9`, sha256 `327ad485dfcf1cbd9405ea6512aa0a788990a5c98ef14ba8585896cdc9749866` | The exact Slaney mel filterbank the model was trained against. Recomputing `librosa.filters.mel(htk=False, norm='slaney')` is ~40 lines whose failure mode is a plausible-but-wrong matrix. |
| `golden_mel.json` | generated, see below | Test fixture for `src/fbank.rs`. |
| `LICENSE-shenava` | the model repo | Apache-2.0. Fetched into the repo rather than onto the PVC, because `fetch-model.sh` fails closed on every file it lists and a licence file is not worth an outage. |

Note the filterbank comes from the **fp16** repo while `fetch-model.sh` pulls the
**streaming** one. That is safe and not an accident: `export_manifest.json` names
`shenava-koochik-1.0.nemo` as the source of both exports, and that `.nemo`'s
`model_config.yaml` carries the single preprocessor definition both inherit.

Verify the filterbank has not drifted from what upstream serves:

```sh
curl -sL https://huggingface.co/PersianML/Shenava-Koochik-v1.0-ONNX-fp16/resolve/main/mel_filters_slaney_80x257.json \
  | shasum -a 256
```

A revision alone is not checkable once you have the file in hand, which is the
whole reason to write the hash down next to it.

## Regenerating `golden_mel.json`

It was **not** written from the prose in `fbank.rs` — a fixture derived from the
same description as the code only proves the two agree. It came from a numpy
reference that was validated end to end first: six Persian clips decoded through the
real ONNX model at a mean CER of 0.022, five of them character-exact. Only then were
its intermediate mel frames frozen here.

Regenerate the same way: implement the pipeline independently, prove it transcribes
real Persian correctly, and only then dump frames 0, 100 and last. If you regenerate
from `fbank.rs` itself you have deleted the test.

`dither` must be 0. The model config sets `1.0e-05`, which is additive training
noise — leave it on and the fixture differs on every run.
