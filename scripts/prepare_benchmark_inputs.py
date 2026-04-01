"""Prepare runnable benchmark jsonl inputs from local dataset directories.

This script converts benchmark sources under /mnt/sda/Datasets into the compact
json/jsonl schema consumed by scripts/run_benchmark.py.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode("utf-8"))
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt_text(ciphertext_b64: str, password: str) -> str:
    if not ciphertext_b64:
        return ciphertext_b64
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes([a ^ b for a, b in zip(encrypted, key)])
    return decrypted.decode("utf-8")


def _ext_from_image_bytes(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return ".bin"


def prepare_videodr(video_dir: Path, csv_path: Path, out_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) < 3:
                continue
            sample_id = str(line[0]).strip()
            question = str(line[1]).strip()
            answer = str(line[2]).strip()
            if not sample_id:
                continue
            video_path = video_dir / f"{sample_id}.mp4"
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": str(video_path),
                    "question": question,
                    "reference_answer": answer,
                    "metadata": {
                        "category": line[3] if len(line) > 3 else "",
                        "difficulty": line[4] if len(line) > 4 else "",
                        "dataset_root": str(video_dir.parent),
                    },
                }
            )
    _save_jsonl(rows, out_path)
    return len(rows)


def prepare_vdr_bench(parquet_path: Path, image_dir: Path, out_path: Path) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(str(parquet_path))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(table.to_pylist()):
        sample_id = str(item.get("id") or f"vdr_{index:06d}")
        question = str(item.get("question") or "")
        answer = str(item.get("answer") or "")
        image_blob = item.get("image")
        image_path = image_dir / f"{sample_id}.jpg"
        if isinstance(image_blob, bytes):
            image_path.write_bytes(image_blob)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "question": question,
                "reference_answer": answer,
                "metadata": {"dataset_root": str(parquet_path.parent)},
            }
        )
    _save_jsonl(rows, out_path)
    return len(rows)


def prepare_mmsearch_plus(
    parquet_files: list[Path],
    image_dir: Path,
    out_path: Path,
    *,
    canary: str = "MMSearch-Plus",
) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counter = 0

    for parquet_path in parquet_files:
        table = pq.read_table(str(parquet_path))
        for item in table.to_pylist():
            sample_id = f"mmsearch_{counter:06d}"
            counter += 1

            question_raw = str(item.get("question") or "")
            question = _decrypt_text(question_raw, canary) if question_raw else ""

            answers_raw = item.get("answer") or []
            answers: list[str] = []
            for candidate in answers_raw:
                candidate = str(candidate)
                answers.append(_decrypt_text(candidate, canary) if candidate else candidate)
            reference_answer = " | ".join(answers)

            media_paths: list[str] = []
            for img_idx in range(1, 6):
                key = f"img_{img_idx}"
                image_payload = item.get(key)
                if not image_payload:
                    continue
                if isinstance(image_payload, dict):
                    blob = image_payload.get("bytes")
                else:
                    blob = image_payload
                if not isinstance(blob, bytes):
                    continue
                ext = _ext_from_image_bytes(blob)
                image_path = image_dir / f"{sample_id}_{key}{ext}"
                image_path.write_bytes(blob)
                media_paths.append(str(image_path))

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "question": question,
                "reference_answer": reference_answer,
                "metadata": {
                    "num_images": item.get("num_images"),
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "subtask": item.get("subtask"),
                    "video_url": _decrypt_text(str(item.get("video_url") or ""), canary)
                    if item.get("video_url")
                    else "",
                    "arxiv_id": _decrypt_text(str(item.get("arxiv_id") or ""), canary)
                    if item.get("arxiv_id")
                    else "",
                    "dataset_root": str(parquet_path.parent.parent.parent),
                },
            }
            if len(media_paths) <= 1:
                row["image_path"] = media_paths[0] if media_paths else ""
            else:
                row["media_paths"] = media_paths
            rows.append(row)

    _save_jsonl(rows, out_path)
    return len(rows)


def prepare_browsecomp(level1_path: Path, level2_path: Path, image_root: Path, out_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    for source_tag, source_path in [("level1", level1_path), ("level2", level2_path)]:
        with source_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                relative_image_path = str(item.get("image_path") or "")
                image_name = Path(relative_image_path).name
                image_path = image_root / source_tag / image_name
                answers = item.get("answers") or []
                reference = " | ".join(str(x) for x in answers)
                rows.append(
                    {
                        "sample_id": f"browsecomp_{source_tag}_{idx:06d}",
                        "image_path": str(image_path),
                        "question": str(item.get("question") or ""),
                        "reference_answer": reference,
                        "metadata": {
                            "domain": item.get("domain", ""),
                            "source_split": source_tag,
                            "dataset_root": str(image_root.parent),
                        },
                    }
                )
    _save_jsonl(rows, out_path)
    return len(rows)


def prepare_mmdeepresearch(quiz_path: Path, quiz_vef_path: Path, image_dir: Path, out_path: Path) -> int:
    ref_by_id: dict[str, str] = {}
    with quiz_vef_path.open("r", encoding="utf-8") as f_ref:
        for line in f_ref:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ref_by_id[str(item.get("id"))] = str(item.get("body") or "")

    rows: list[dict[str, Any]] = []
    with quiz_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = str(item.get("id", idx))
            image_names = item.get("image_url") or []
            media_paths = [str(image_dir / str(name)) for name in image_names]
            rows.append(
                {
                    "sample_id": f"mmdeepresearch_{sample_id}",
                    "media_paths": media_paths,
                    "question": str(item.get("body") or ""),
                    "reference_answer": ref_by_id.get(str(sample_id), ""),
                    "metadata": {
                        "caption": item.get("caption", ""),
                        "tags": item.get("tags", []),
                        "language": item.get("language", ""),
                        "difficulty": item.get("difficulty", ""),
                        "dataset_root": str(image_dir.parent),
                    },
                }
            )
    _save_jsonl(rows, out_path)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark input jsonl files from local /mnt/sda/Datasets.")
    parser.add_argument("--datasets-root", default="/mnt/sda/Datasets")
    parser.add_argument("--output-dir", default="data/benchmarks/real")
    parser.add_argument("--mmsearch-canary", default="MMSearch-Plus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videodr_count = prepare_videodr(
        video_dir=datasets_root / "VideoDR-Bench" / "video",
        csv_path=datasets_root / "VideoDR-Bench" / "VideoDR.csv",
        out_path=output_dir / "videodr.jsonl",
    )
    print(f"Prepared VideoDR rows: {videodr_count}")

    vdr_count = prepare_vdr_bench(
        parquet_path=datasets_root / "VDR-Bench" / "vision-deepresearch_VDR-Bench-full.parquet",
        image_dir=datasets_root / "VDR-Bench" / "images",
        out_path=output_dir / "vdr_bench.jsonl",
    )
    print(f"Prepared VDR-Bench rows: {vdr_count}")

    mmsearch_count = prepare_mmsearch_plus(
        parquet_files=[
            datasets_root / "MMSearch-Plus" / "parquet" / "default" / "train" / "0000.parquet",
            datasets_root / "MMSearch-Plus" / "parquet" / "default" / "train" / "0001.parquet",
        ],
        image_dir=datasets_root / "MMSearch-Plus" / "images",
        out_path=output_dir / "mmsearch_plus.jsonl",
        canary=args.mmsearch_canary,
    )
    print(f"Prepared MMSearch-Plus rows: {mmsearch_count}")

    browsecomp_count = prepare_browsecomp(
        level1_path=datasets_root / "BrowseComp-VL" / "bc_vl_level1.jsonl",
        level2_path=datasets_root / "BrowseComp-VL" / "bc_vl_level2.jsonl",
        image_root=datasets_root / "BrowseComp-VL" / "images",
        out_path=output_dir / "browsecomp_vl.jsonl",
    )
    print(f"Prepared BrowseComp-VL rows: {browsecomp_count}")

    mmdeep_count = prepare_mmdeepresearch(
        quiz_path=datasets_root / "MMDeepResearch-Bench" / "quiz.jsonl",
        quiz_vef_path=datasets_root / "MMDeepResearch-Bench" / "quiz_vef.jsonl",
        image_dir=datasets_root / "MMDeepResearch-Bench" / "image",
        out_path=output_dir / "mmdeepresearch_bench.jsonl",
    )
    print(f"Prepared MMDeepResearch-Bench rows: {mmdeep_count}")

    print(f"All benchmark inputs written to: {output_dir}")


if __name__ == "__main__":
    main()
