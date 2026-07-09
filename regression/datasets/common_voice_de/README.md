# Common Voice: German

Short read-speech clips. Regression-tests bewer's core metrics and its German
preprocessing/normalization overlay (`Dataset(language="de")`).

| Examples | Audio |
|---|---|
| 16,188 | 27h 29m |

## Files

| File | Description |
|---|---|
| `data.jsonl` | One row per clip: `{"id", "ref", "hyp"}`. All Common Voice test clips. |

## Provenance

**References + audio:** public mirror
[`fsicoli/common_voice_19_0`](https://huggingface.co/datasets/fsicoli/common_voice_19_0)
of Mozilla Common Voice 19.0 (license: CC0-1.0). Only the `test` split is fetched
(transcript tsv + test audio tar via `hf_hub_download`); the train split is never
downloaded. The reference is the clip's `sentence` and the `id` is the mp3 filename
stem. Clips are ordered by filename (deterministic).

**Hypotheses:** `openai/whisper-tiny` (multilingual) with the language forced to
German, greedy/deterministic decoding.
