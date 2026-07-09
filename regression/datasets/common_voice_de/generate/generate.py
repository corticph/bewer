#!/usr/bin/env python
"""Generate the Common Voice German (de) regression dataset (refs + Whisper-tiny hyps).

Reproducible artifact builder. Run it to (re)create ../data.jsonl — one
{"id", "ref", "hyp"} row per clip.

Source (public, CC0): the community mirror ``fsicoli/common_voice_19_0`` of Mozilla
Common Voice 19.0. Test-split only: instead of ``datasets.load_dataset`` (which prepares
*all* splits — the German train alone is ~11 GB across 15 shards), we fetch just the test
transcript tsv and the test audio tar(s) via ``hf_hub_download`` and transcribe every
test clip.

Hypotheses: ``openai/whisper-tiny`` (multilingual), greedy/deterministic decoding with the
language forced to German. Audio is never committed — only the resulting text.

Reproducibility: this folder is a self-contained poetry project; its poetry.lock pins the
exact stack (that lock is the provenance record). Run inside it:
    cd regression/datasets/common_voice_de/generate
    poetry install && poetry run python generate.py --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import tempfile
from pathlib import Path

# Outputs live in the parent dataset folder (this script lives in its generate/ subdir).
DATASET_DIR = Path(__file__).resolve().parent.parent
HF_DATASET = "fsicoli/common_voice_19_0"
WHISPER_MODEL = "openai/whisper-tiny"  # multilingual
LANGUAGE = "de"
# Clips are transcribed in GPU batches of this size (part of the recipe: changing it can
# perturb hypotheses at the fp16 level). A fixed value keeps regeneration reproducible.
BATCH_SIZE = 32


def _download_test_files() -> tuple[str, list[str]]:
    """Fetch only the test transcript tsv + test audio tar(s) (no train download)."""
    from huggingface_hub import HfApi, hf_hub_download

    tsv = hf_hub_download(HF_DATASET, f"transcript/{LANGUAGE}/test.tsv", repo_type="dataset")
    all_files = [s.rfilename for s in HfApi().dataset_info(HF_DATASET).siblings]
    shards = sorted(f for f in all_files if f.startswith(f"audio/{LANGUAGE}/test/") and f.endswith(".tar"))
    if not shards:
        raise ValueError(f"No test audio shards found for language '{LANGUAGE}' in {HF_DATASET}.")
    tars = [hf_hub_download(HF_DATASET, s, repo_type="dataset") for s in shards]
    return tsv, tars


def build(device: int) -> None:
    import librosa
    import torch
    from tqdm import tqdm
    from transformers import pipeline

    print(f"[{LANGUAGE}] fetching test split (tsv + audio tar, no train) ...")
    tsv, tars = _download_test_files()
    with open(tsv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # Sorted by filename for a deterministic order (fixes batch composition).
    wanted = {r["path"]: r["sentence"] for r in sorted(rows, key=lambda r: r["path"])}

    print(f"[{LANGUAGE}] loading {WHISPER_MODEL} on cuda:{device}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        chunk_length_s=30,
        batch_size=BATCH_SIZE,
        device=device,
        torch_dtype=torch.float16,
    )
    # The model card ships stale `forced_decoder_ids`; clearing it avoids a conflict warning
    # with the `task`/`language` we pass below (task already takes precedence — no output change).
    asr.model.config.forced_decoder_ids = None
    asr.model.generation_config.forced_decoder_ids = None
    gen_kwargs = {
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.0,
        "language": LANGUAGE,
        "task": "transcribe",
    }

    results = []
    with (
        tempfile.TemporaryDirectory() as tmp,
        tqdm(total=len(wanted), desc=f"[{LANGUAGE}] transcribing", unit="clip") as pbar,
    ):

        def flush(batch: list[tuple[str, str, object]]) -> None:
            if not batch:
                return
            # Pass the whole batch so the pipeline runs it as one GPU batch (not clip-by-clip).
            outs = asr([{"raw": a, "sampling_rate": 16000} for _, _, a in batch], generate_kwargs=gen_kwargs)
            for (cid, ref, _), out in zip(batch, outs):
                results.append({"id": cid, "ref": ref, "hyp": out["text"].strip()})
            pbar.update(len(batch))
            batch.clear()

        batch: list[tuple[str, str, object]] = []
        for tar_path in tars:
            with tarfile.open(tar_path) as tf:
                for member in tf.getmembers():
                    base = Path(member.name).name
                    if base not in wanted:
                        continue
                    dst = Path(tmp) / base
                    dst.write_bytes(tf.extractfile(member).read())
                    arr, _ = librosa.load(str(dst), sr=16000, mono=True)  # decode + resample to 16 kHz
                    dst.unlink()
                    batch.append((Path(base).stem, wanted[base].strip(), arr))
                    if len(batch) >= BATCH_SIZE:
                        flush(batch)
        flush(batch)

    results.sort(key=lambda r: r["id"])
    data_path = DATASET_DIR / "data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{LANGUAGE}] wrote {len(results)} rows -> {data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    args = parser.parse_args()
    build(args.device)


if __name__ == "__main__":
    main()
