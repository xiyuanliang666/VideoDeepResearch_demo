#!/usr/bin/env python3
"""Test script: run rsagent-v4 on first video's first question from vl_test_question.json."""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Test data
TEST_JSON = str(_ROOT / "vl_test_question_new.json")
VIDEO_CACHE_DIR = Path("/tmp/video_cache")

def download_video(url: str, out_dir: Path) -> str:
    """Download video with yt-dlp, return local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use video ID as filename
    vid_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
    out_path = out_dir / f"{vid_id}.mp4"
    if out_path.is_file():
        print(f"Video already cached: {out_path}")
        return str(out_path)
    print(f"Downloading {url} ...")
    subprocess.run([
        "yt-dlp", "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "--merge-output-format", "mp4", "-o", str(out_path), url
    ], check=True)
    return str(out_path)


def main():
    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)

    sample = data["test_questions_wangyu"][0]
    video_url = sample["video_url"]
    question = sample["question1"]

    print(f"Video: {video_url}")
    print(f"Question: {question}")
    print("=" * 60)

    # Download video
    video_path = download_video(video_url, VIDEO_CACHE_DIR)

    # Run agent
    out_dir = _ROOT / "runs" / "test_wangyu_q1"
    cmd = [
        sys.executable, str(_ROOT / "run_agent.py"),
        "--video", video_path,
        "--question", question,
        "--out-dir", str(out_dir),
        "--max-rounds", "12",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)
    subprocess.run(cmd, check=True)

    # Print result
    result_file = out_dir / "result.json"
    if result_file.is_file():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        print("\n" + "=" * 60)
        print("FINAL ANSWER:")
        print(result.get("final_answer", "(empty)"))
        print("\nSub-questions:")
        for i, sq in enumerate(result.get("sub_questions", []), 1):
            status = "✓" if sq.get("sufficient") else "✗"
            print(f"  {status} SQ{i}: {sq['text'][:80]}")
            if sq.get("answer"):
                print(f"       Answer: {sq['answer'][:120]}")


if __name__ == "__main__":
    main()
