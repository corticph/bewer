#!/usr/bin/env python
"""Generate the Earnings21 regression dataset (refs + Parakeet-CTC hyps + oracle key terms).

Reproducible artifact builder. Run it to (re)create:
  - data.jsonl       one {"id", "ref", "hyp"} row per Earnings21 call
  - key_terms.txt    the Earnings21 oracle contextual-biasing list

Sources (see README.md in this folder):
  - Audio + reference transcripts: HuggingFace ``Revai/earnings21`` (CC-BY-SA-4.0).
  - Hypotheses: NVIDIA NeMo ``nvidia/parakeet-ctc-1.1b`` (English FastConformer-CTC).
  - Key terms: ``earnings21/bias_lists/oracle_list.txt`` from the original
    revdotcom/speech-datasets GitHub repo.

Long-form: each call is ~30-95 min, so the FastConformer encoder is switched to
local attention (``rel_pos_local_attn``) to transcribe full files without OOM.
CTC greedy decoding is deterministic, so the hypotheses are reproducible for a
fixed model + library versions. Audio itself is never committed — only the text.

Reproducibility: the exact dependency versions are pinned by this folder's
``poetry.lock`` (that lock file *is* the provenance record). Run inside it:
    cd regression/datasets/earnings21/generate
    poetry install && poetry run python generate.py [--limit N] [--device 0]
The model weights are pinned to a specific HuggingFace revision (below); the two
things that move the hypotheses — code (lock) and weights (revision) — are fixed.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

# Outputs live in the dataset folder (this script lives in its generate/ subdir).
DATASET_DIR = Path(__file__).resolve().parent.parent

HF_DATASET = "Revai/earnings21"
PARAKEET_MODEL = "nvidia/parakeet-ctc-1.1b"
# Pin the exact weights that produced the committed data.jsonl. HF's `main` has
# since moved on, so reproduction requires this revision, not the latest.
PARAKEET_REVISION = "cfdeadd14830783de8f2b1a436f2ad059db27424"
PARAKEET_NEMO_FILE = "parakeet-ctc-1.1b.nemo"
ORACLE_LIST_URL = (
    "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings21/bias_lists/oracle_list.txt"
)

# Earnings21 references contain non-speech angle-bracket tags (e.g. <inaudible>,
# <crosstalk>, <unk>, <laugh>). These are not transcription content and the angle
# brackets have special meaning to error_align, so we strip them and tidy spacing.
_TAG_RE = re.compile(r"<[^>]*>")


def clean_ref(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)  # collapse gaps left behind
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # no space before punctuation
    return text.strip()


def fetch_oracle_key_terms(dest: Path) -> int:
    """Download the Earnings21 oracle biasing list (one term/phrase per line)."""
    with urllib.request.urlopen(ORACLE_LIST_URL) as resp:  # (trusted URL)
        raw = resp.read().decode("utf-8")
    # Normalise to sorted, de-duplicated, non-empty lines.
    terms = sorted({line.strip() for line in raw.splitlines() if line.strip()})
    dest.write_text("\n".join(terms) + "\n")
    return len(terms)


def _hyp_text(record) -> str:
    """Extract text from a NeMo transcribe() record (Hypothesis | dict | str)."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return record.get("text", "")
    return getattr(record, "text", "")


def build(limit: int | None, device: int) -> None:
    import nemo.collections.asr as nemo_asr
    import torch
    from datasets import load_dataset

    print(f"Loading {HF_DATASET} (audio + reference transcripts)...")
    ds = load_dataset(HF_DATASET, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  {len(ds)} calls")

    dev = f"cuda:{device}"
    # Download the exact pinned .nemo weights, then restore. hf_hub_download honors
    # `revision`, whereas NeMo's from_pretrained() always tracks the repo's main.
    from huggingface_hub import hf_hub_download

    print(f"Loading ASR model: {PARAKEET_MODEL}@{PARAKEET_REVISION[:8]} on {dev}")
    nemo_path = hf_hub_download(PARAKEET_MODEL, filename=PARAKEET_NEMO_FILE, revision=PARAKEET_REVISION)
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=dev)
    # Long-form: local attention lets the FastConformer handle full-length calls.
    model.change_attention_model("rel_pos_local_attn", [256, 256])
    model.change_subsampling_conv_chunking_factor(1)
    model = model.to(dev).eval()

    rows = []
    for i, ex in enumerate(ds):
        audio = ex["audio"]
        file_id = Path(audio["path"]).stem  # e.g. ".../wav/4320211.wav" -> "4320211"
        arr = audio["array"].astype("float32")  # Revai audio is 16 kHz mono
        print(f"  [{i + 1}/{len(ds)}] {file_id}  ({len(arr) / audio['sampling_rate'] / 60:.1f} min)")
        with torch.inference_mode():
            out = model.transcribe([torch.from_numpy(arr).to(dev, torch.float32)], batch_size=1, verbose=False)
        rows.append({"id": file_id, "ref": clean_ref(ex["text"]), "hyp": _hyp_text(out[0]).strip()})

    rows.sort(key=lambda r: r["id"])
    data_path = DATASET_DIR / "data.jsonl"
    with open(data_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {data_path}")

    n_terms = fetch_oracle_key_terms(DATASET_DIR / "key_terms.txt")
    print(f"Wrote {n_terms} oracle key terms -> {DATASET_DIR / 'key_terms.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N calls (for testing).")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--key-terms-only", action="store_true", help="Only (re)download the oracle key-term list.")
    args = parser.parse_args()

    if args.key_terms_only:
        n = fetch_oracle_key_terms(DATASET_DIR / "key_terms.txt")
        print(f"Wrote {n} oracle key terms -> {DATASET_DIR / 'key_terms.txt'}")
        return
    build(args.limit, args.device)


if __name__ == "__main__":
    main()
