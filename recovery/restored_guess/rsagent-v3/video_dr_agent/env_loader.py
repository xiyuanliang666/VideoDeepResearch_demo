"""
Load `video_dr_agent/.env` from the repository root and apply name aliases.

Supported aliases (after dotenv load):
  DEEP_RESEARCH_BASE_URL -> VLLM_BASE_URL (if unset)
  DEEP_RESEARCH_MODEL     -> VLLM_MODEL (if unset)
  DEEP_RESEARCH_API_KEY   -> VLLM_API_KEY (if unset)
  SERP_API_KEY            -> SERPER_API_KEY (if unset)
  READER_URL              -> JINA_READER_BASE (if unset; used by jina_reader)

Optional:
  RESEARCH_SYSTEM_PROMPT_PATH   UTF-8 file path for research system prompt (repo-relative or absolute).
                                  Also used by ``run_video_dr_agent`` via ``research_prompt.resolve_research_system_content``.

  FOCUS_SOURCE_VIDEO / VIDEO_PATH   Local video file used when ``--video`` is omitted: passed to vLLM as ``video_url``
                                    and as ``default_video_path`` for ``focus_select_keyframes`` (omit ``video_path`` in tool).
                                    ``FOCUS_SOURCE_VIDEO`` wins if both are set.

  Uniform overview (before the research loop, OpenCV + ``test_data_frame/uniform_sample_frames.py``):
  TEST_UNIFORM_OVERVIEW_FRAMES   If > 0, run uniform sampling from the effective source video into the overview dir,
                                 and use those images as the initial frame set (see ``run_video_dr_agent``).
  TEST_UNIFORM_OVERVIEW_DIR      Output directory for overview frames. If unset but count > 0, defaults to
                                 ``<repo>/test_data_frame/overview_<video_stem>/``.

  RESEARCH_INITIAL_INCLUDE_FULL_VIDEO   If 0 / false / no, the first user message omits the full ``video_url`` payload
                                        and only sends overview frames + question (still use ``--video`` / env for tools).
                                        Default: include full video when a path is set (backward compatible).

  RESEARCH_INJECT_FOCUS_FRAMES          If 0 / false / no, do not append multimodal user messages with FOCUS-exported
                                        JPEGs after critic feedback (default: inject when paths exist).
  RESEARCH_MAX_FOCUS_IMAGES             Max number of FOCUS frames to attach per round (default 32).

  RESEARCH_REJECT_ROUND1_NO_TOOLS       If 1 / true / yes: if round 1 assistant has no ``<tool_call>``, append a nudge
                                        user message and request one more assistant turn before ending (see ``loop.run_loop``).
  RESEARCH_ROUND1_NO_TOOLS_FEEDBACK     Optional custom text for that nudge (UTF-8 one line or short paragraph).

  RESEARCH_EVIDENCE_DOMAIN              Optional **problem domain** for the current task (e.g. ``registry``, ``academic``).
                                        Biases ``deep_research_web_search`` worker ``site:`` filters and aligns the deep
                                        Critic’s URL ranking / ``registry_verified`` hints. Alias: ``TEST_QUESTION_DOMAIN``
                                        is copied into ``RESEARCH_EVIDENCE_DOMAIN`` when the latter is unset.

  RESEARCH_PLANNING_PHASE               If 0 / false / no / off, skip the mandatory planning turn before the tool loop.
                                        Default (unset): planning phase is **enabled**.

  RESEARCH_PLANNING_USER_PATH           Optional UTF-8 file path whose contents replace the default English planning user
                                        message (Host turn before tools).

  RESEARCH_IMAGE_MAX_SIDE               If >0, resize local images (longest side) before base64 in vLLM
                                        requests—reduces multimodal tokens (important for many overview frames).
                                        Requires Pillow. Example: 768 or 1024.
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def load_video_dr_agent_dotenv(repo_root: Path | str, *, override: bool = False) -> bool:
    """
    Load ``<repo_root>/video_dr_agent/.env`` if present (requires python-dotenv).

    Returns True if the file existed and load_dotenv ran.
    """
    root = Path(repo_root)
    env_path = root / "video_dr_agent" / ".env"
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv(env_path, override=override)
    _apply_aliases()
    return True


def _apply_aliases() -> None:
    """Map alternate names from .env files to names used by the CLI and tools."""

    def copy_if(src: str, dst: str) -> None:
        v = os.environ.get(src)
        if v is None or not str(v).strip():
            return
        v = _strip_quotes(str(v))
        if not os.environ.get(dst) or not str(os.environ.get(dst, "")).strip():
            os.environ[dst] = v

    copy_if("DEEP_RESEARCH_BASE_URL", "VLLM_BASE_URL")
    copy_if("DEEP_RESEARCH_MODEL", "VLLM_MODEL")
    copy_if("DEEP_RESEARCH_API_KEY", "VLLM_API_KEY")
    copy_if("SERP_API_KEY", "SERPER_API_KEY")
    copy_if("READER_URL", "JINA_READER_BASE")
    copy_if("TEST_QUESTION_DOMAIN", "RESEARCH_EVIDENCE_DOMAIN")


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})


def collect_image_paths_from_directory(directory: Path | str) -> list[str]:
    """Sorted list of image files under ``directory`` (non-recursive)."""
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        return []
    out: list[str] = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
            out.append(str(f))
    return out


def collect_frames_from_test_env() -> list[str]:
    """
    Use ``TEST_FRAME_PATH`` from the environment: directory -> sorted image paths;
    file -> single-element list.
    """
    raw = (os.environ.get("TEST_FRAME_PATH") or "").strip()
    raw = _strip_quotes(raw)
    if not raw:
        return []
    p = Path(raw).expanduser().resolve()
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return collect_image_paths_from_directory(p)
    return []


def get_uniform_overview_frame_count_from_env() -> int:
    raw = (os.environ.get("TEST_UNIFORM_OVERVIEW_FRAMES") or "").strip()
    raw = _strip_quotes(raw)
    if not raw:
        return 0
    try:
        return max(0, int(raw, 10))
    except ValueError:
        return 0


def get_uniform_overview_output_dir_from_env() -> str | None:
    raw = (os.environ.get("TEST_UNIFORM_OVERVIEW_DIR") or "").strip()
    raw = _strip_quotes(raw)
    return raw or None


def research_inject_focus_frames_from_env() -> bool:
    raw = (os.environ.get("RESEARCH_INJECT_FOCUS_FRAMES") or "").strip()
    raw = _strip_quotes(raw).lower() if raw else ""
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


def research_max_focus_images_from_env() -> int:
    raw = (os.environ.get("RESEARCH_MAX_FOCUS_IMAGES") or "").strip()
    raw = _strip_quotes(raw)
    if not raw:
        return 32
    try:
        return max(0, int(raw, 10))
    except ValueError:
        return 32


def research_reject_round1_no_tools_from_env() -> bool:
    raw = (os.environ.get("RESEARCH_REJECT_ROUND1_NO_TOOLS") or "").strip()
    raw = _strip_quotes(raw).lower() if raw else ""
    if not raw:
        return False
    return raw in ("1", "true", "yes", "on")


def get_research_round1_no_tools_feedback_from_env() -> str | None:
    raw = (os.environ.get("RESEARCH_ROUND1_NO_TOOLS_FEEDBACK") or "").strip()
    raw = _strip_quotes(raw)
    return raw or None


def research_planning_phase_enabled_from_env() -> bool:
    """Default True unless RESEARCH_PLANNING_PHASE is explicitly disabled."""
    raw = (os.environ.get("RESEARCH_PLANNING_PHASE") or "").strip()
    raw = _strip_quotes(raw).lower() if raw else ""
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


def get_research_planning_user_path_from_env() -> str | None:
    raw = (os.environ.get("RESEARCH_PLANNING_USER_PATH") or "").strip()
    raw = _strip_quotes(raw)
    return raw or None


def research_initial_include_full_video_from_env() -> bool:
    """
    ``RESEARCH_INITIAL_INCLUDE_FULL_VIDEO``: unset / 1 / true / yes → include full video in first user turn when path set.
    0 / false / no → omit (overview frames + question only).
    """
    raw = (os.environ.get("RESEARCH_INITIAL_INCLUDE_FULL_VIDEO") or "").strip().lower()
    raw = _strip_quotes(raw) if raw else ""
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


def get_test_question_from_env() -> str | None:
    q = (os.environ.get("TEST_QUESTION") or "").strip()
    q = _strip_quotes(q)
    return q or None


def get_research_evidence_domain_from_env() -> str | None:
    """
    Task / question **evidence domain** (Host profile for search + critic alignment).

    Reads ``RESEARCH_EVIDENCE_DOMAIN`` (after aliases: may be filled from ``TEST_QUESTION_DOMAIN``).
    Returns stripped string or None if unset.
    """
    raw = (os.environ.get("RESEARCH_EVIDENCE_DOMAIN") or "").strip()
    raw = _strip_quotes(raw)
    return raw or None


def get_default_source_video_path() -> str | None:
    """
    Default local video for research multimodal input and FOCUS ``default_video_path``.

    Reads ``FOCUS_SOURCE_VIDEO`` first, then ``VIDEO_PATH``. Returns resolved absolute path
    string if the file exists, else the stripped path string (caller may error later).
    """
    raw = (os.environ.get("FOCUS_SOURCE_VIDEO") or os.environ.get("VIDEO_PATH") or "").strip()
    raw = _strip_quotes(raw)
    if not raw:
        return None
    p = Path(raw).expanduser()
    try:
        p = p.resolve()
    except OSError:
        return raw
    return str(p) if p.is_file() else raw
