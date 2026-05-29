"""Compatibility import for the public VideoDeepResearch framework.

The implementation currently lives under ``src.videodeepresearch`` while the
repository is being consolidated. Import this package in external code so the
future package move does not change user-facing imports.
"""

from src.videodeepresearch import *  # noqa: F401,F403
