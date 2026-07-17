# Earnings21

English long-form earnings calls, an entity-dense benchmark. Guards bewer's core and
key-term metrics against drift between releases.

| Examples | Audio |
|---|---|
| 44 | 39h 15m |

## Files

| File | Description |
|---|---|
| `data.jsonl` | One row per call: `{"id", "ref", "hyp"}`. The full dataset. |
| `key_terms.txt` | The Earnings21 oracle contextual-biasing list (one term/phrase per line). |

## Provenance

**References + audio:** HuggingFace [`Revai/earnings21`](https://huggingface.co/datasets/Revai/earnings21)
(license: CC-BY-SA-4.0). The reference transcript is the dataset's `text` field and
the `id` is the source `wav/<id>.wav` stem. References are cleaned of non-speech
angle-bracket tags (`<inaudible>`, `<crosstalk>`, `<unk>`, `<laugh>`, ...) with spacing
tidied; those tags are not transcription content and the angle brackets have special
meaning to `error_align`.

**Hypotheses:** transcribed with NVIDIA NeMo
[`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b)
(English FastConformer-CTC). The encoder uses local attention
(`change_attention_model("rel_pos_local_attn", [256, 256])`) so full multi-minute
calls transcribe without OOM, with CTC greedy decoding for determinism.

**Oracle key terms:** `oracle_list.txt` from the original Earnings21 repo
(<https://github.com/revdotcom/speech-datasets/blob/main/earnings21/bias_lists/oracle_list.txt>),
the 1013 words/phrases found in the references, de-duplicated and sorted.
