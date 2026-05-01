#!/usr/bin/env python3
"""
将 FOCUS 筛选结果整理到独立子目录：
  output_dir/frames/     选中的帧（图片）
  output_dir/question/   问题文本 + 元数据 JSON
  output_dir/video/      原视频（默认符号链接，可用 --copy-video 复制）

可选 --html 额外生成 comparison.html 对比页。
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from decord import VideoReader, cpu
from PIL import Image


def load_lvb_entry(lvb_path: Path, index: int) -> dict:
    with open(lvb_path, encoding="utf-8") as f:
        data = json.load(f)
    if index < 0 or index >= len(data):
        raise IndexError(f"video_index {index} out of range [0, {len(data)})")
    return data[index]


def resolve_video_path(dataset_path: Path, entry: dict, dataset_name: str) -> Path:
    if dataset_name == "longvideobench":
        return dataset_path / "videos" / entry["video_path"]
    if dataset_name == "videomme":
        return dataset_path / "data" / (entry["videoID"] + ".mp4")
    raise ValueError("dataset_name must be longvideobench or videomme")


def load_selected_indices(selected_path: Path, video_index: int) -> list[int]:
    with open(selected_path, encoding="utf-8") as f:
        data = json.load(f)
    if video_index >= len(data):
        raise IndexError(f"video_index {video_index} out of range for selected_frames ({len(data)} videos)")
    return [int(x) for x in data[video_index]]


def save_frame_images(
    vr: VideoReader,
    indices: list[int],
    out_dir: Path,
    max_side: int,
) -> list[tuple[int, float, str]]:
    """导出帧图，返回 [(frame_idx, time_sec, 相对路径 frames/xxx), ...]"""
    fps = float(vr.get_avg_fps())
    meta = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        idx = max(0, min(idx, len(vr) - 1))
        arr = vr[idx].asnumpy()
        im = Image.fromarray(arr)
        if max_side > 0:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        fname = f"{i:03d}_frame{idx}.jpg"
        fpath = out_dir / fname
        im.save(fpath, quality=92)
        t = idx / fps if fps > 0 else 0.0
        meta.append((idx, t, f"frames/{fname}"))
    return meta


def build_html(
    title: str,
    question: str,
    extra_lines: list[str],
    video_rel: str | None,
    frames_meta: list[tuple[int, float, str]],
) -> str:
    q_esc = html.escape(question)
    lines_html = "".join(f"<p class='meta'>{html.escape(x)}</p>" for x in extra_lines)

    if video_rel:
        video_block = f"""
        <section class="panel">
          <h2>原视频</h2>
          <video controls preload="metadata" src="{html.escape(video_rel)}"></video>
          <p class="hint">若无法播放，可在本目录执行 <code>python -m http.server 8765</code> 后访问本页。</p>
        </section>
        """
    else:
        video_block = """
        <section class="panel">
          <h2>原视频</h2>
          <p class="meta">未放置视频文件。</p>
        </section>
        """

    thumbs = []
    for rank, (idx, t_sec, rel) in enumerate(frames_meta):
        thumbs.append(
            f"""
            <figure class="thumb">
              <img loading="lazy" src="{html.escape(rel)}" alt="frame {idx}" />
              <figcaption>#{rank + 1} · 帧 {idx} · {t_sec:.2f}s</figcaption>
            </figure>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #5eb8ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1.5rem;
      line-height: 1.5;
    }}
    h1 {{ font-size: 1.35rem; margin: 0 0 1rem; color: var(--accent); }}
    h2 {{ font-size: 1.05rem; margin: 0 0 0.75rem; color: var(--muted); font-weight: 600; }}
    .panel {{
      background: var(--card);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.5rem;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .question {{ font-size: 1.1rem; white-space: pre-wrap; }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin: 0.35rem 0; }}
    .hint {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.75rem; }}
    video {{ max-width: 100%; max-height: 56vh; border-radius: 8px; background: #000; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 1rem;
    }}
    figure.thumb {{
      margin: 0;
      background: #111820;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    figure.thumb img {{ width: 100%; height: auto; display: block; }}
    figure.thumb figcaption {{
      padding: 0.5rem 0.6rem;
      font-size: 0.75rem;
      color: var(--muted);
    }}
    code {{ background: rgba(0,0,0,0.35); padding: 0.15rem 0.4rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <section class="panel">
    <h2>问题</h2>
    <p class="question">{q_esc}</p>
    {lines_html}
  </section>
  {video_block}
  <section class="panel">
    <h2>FOCUS 筛选帧集（共 {len(frames_meta)} 帧）</h2>
    <div class="grid">{''.join(thumbs)}</div>
  </section>
</body>
</html>
"""


def export_folder_bundle(
    output_dir: Path,
    entry: dict,
    video_path: Path,
    indices: list[int],
    vr: VideoReader,
    *,
    thumb_size: int,
    copy_video: bool,
    symlink_video: bool,
    write_html: bool,
) -> None:
    frames_dir = output_dir / "frames"
    question_dir = output_dir / "question"
    video_dir = output_dir / "video"

    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in (frames_dir, question_dir, video_dir):
        if sub.exists():
            shutil.rmtree(sub)
    html_path = output_dir / "comparison.html"
    if html_path.is_file():
        html_path.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = save_frame_images(vr, indices, frames_dir, max_side=thumb_size)

    q = entry.get("question", "")
    (question_dir / "question.txt").write_text(q, encoding="utf-8")
    meta = {
        **entry,
        "selected_frame_indices": indices,
        "source_video_path": str(video_path.resolve()),
        "total_video_frames": len(vr),
        "fps": float(vr.get_avg_fps()),
    }
    (question_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    video_dest = video_dir / "source.mp4"
    video_rel_for_html: str | None = None
    if copy_video:
        shutil.copy2(video_path, video_dest)
        video_rel_for_html = "video/source.mp4"
    elif symlink_video:
        video_dest.symlink_to(video_path.resolve())
        video_rel_for_html = "video/source.mp4"

    if write_html:
        vid = entry.get("id", entry.get("video_id", ""))
        extra = [
            f"样本 id: {vid}",
            f"原视频路径: {video_path}",
            f"总帧数: {len(vr)}",
        ]
        title = f"FOCUS · {vid}"
        doc = build_html(title, q, extra, video_rel_for_html, frames_meta)
        (output_dir / "comparison.html").write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 frames/、question/、video/ 三个子目录（可选 HTML）"
    )
    parser.add_argument(
        "--dataset_name",
        default="longvideobench",
        choices=["longvideobench", "videomme", "custom"],
        help="custom：单视频自定义跑 FOCUS 时用，配合 --custom_video 与 --custom_question*",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="benchmark 数据根目录；dataset_name=custom 时不需要",
    )
    parser.add_argument("--custom_video", type=Path, default=None, help="custom 模式下的视频路径")
    parser.add_argument("--custom_question", type=str, default=None, help="custom 模式下的问题文本")
    parser.add_argument(
        "--custom_question_file",
        type=Path,
        default=None,
        help="custom 模式下从文件读取问题（UTF-8）",
    )
    parser.add_argument("--selected_frames", required=True, type=Path)
    parser.add_argument("--video_index", type=int, default=0)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--thumb_size",
        type=int,
        default=0,
        help="帧图最长边；0 表示原分辨率（文件较大）",
    )
    parser.add_argument("--copy_video", action="store_true", help="将原视频复制到 video/source.mp4")
    parser.add_argument(
        "--symlink_video",
        action="store_true",
        help="将原视频符号链接到 video/source.mp4（与 --copy_video 二选一，默认两者都不选则 video/ 为空）",
    )
    parser.add_argument("--html", action="store_true", help="额外生成根目录下的 comparison.html")
    args = parser.parse_args()

    if args.dataset_name == "custom":
        if not args.custom_video or not args.custom_video.is_file():
            raise FileNotFoundError("custom 模式需要有效的 --custom_video")
        if args.custom_question_file:
            q = args.custom_question_file.read_text(encoding="utf-8").strip()
        elif args.custom_question:
            q = args.custom_question.strip()
        else:
            raise ValueError("custom 模式需要 --custom_question 或 --custom_question_file")
        video_path = args.custom_video.resolve()
        entry = {"question": q, "id": "custom", "video_path": str(video_path)}
    else:
        if args.dataset_path is None:
            raise ValueError("longvideobench/videomme 需要 --dataset_path")
        if args.dataset_name == "longvideobench":
            label_path = args.dataset_path / "lvb_val.json"
        else:
            label_path = args.dataset_path / "videomme.json"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        entry = load_lvb_entry(label_path, args.video_index)
        video_path = resolve_video_path(args.dataset_path, entry, args.dataset_name)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

    indices = load_selected_indices(args.selected_frames, args.video_index)
    vr = VideoReader(str(video_path), ctx=cpu(0))

    export_folder_bundle(
        args.output_dir,
        entry,
        video_path,
        indices,
        vr,
        thumb_size=args.thumb_size,
        copy_video=args.copy_video,
        symlink_video=args.symlink_video,
        write_html=args.html,
    )

    root = args.output_dir.resolve()
    print(f"已写入目录结构:")
    print(f"  {root}/frames/     ({len(indices)} 张)")
    print(f"  {root}/question/   (question.txt, meta.json)")
    vdir = root / "video" / "source.mp4"
    if vdir.exists() or vdir.is_symlink():
        print(f"  {root}/video/      (source.mp4)")
    else:
        print(f"  {root}/video/      (空，可加 --symlink_video 或 --copy_video)")
    if args.html:
        print(f"  {root}/comparison.html")


if __name__ == "__main__":
    main()
