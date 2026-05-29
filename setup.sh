#!/usr/bin/env bash
# ============================================================
# VDR Pilot — One-time setup
# ============================================================
# Usage: ./setup.sh
#
# 1. Creates .env from template (if not exists)
# 2. Installs Python dependencies
# 3. Pre-downloads CV models (YOLO, PaddleOCR, CLIP)
# 4. Validates API config and model routing
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "VDR Pipeline Setup"
echo "baseline: $BASELINE_HOME"
echo "=============================================="

# ── .env template ─────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" << 'ENVEOF'
# ── Workspace paths ───────────────────────────────────────
VDR_BENCHMARK_HOME=/path/to/vdr_investigation
VDR_BASELINE_HOME=/path/to/VideoDeepResearch_demo

# ── API keys ──────────────────────────────────────────────
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.openai.com/v1
SERPER_API_KEY=your-serper-key
# TAVILY_API_KEY=        # alternative to Serper
# OPENAI_API_KEY=         # fallback if LLM_API_KEY is empty

# ── Model routing (model names per stage) ─────────────────
STATE_CONSTRUCTOR_MODEL=gpt-4o
QUERY_PLANNER_MODEL=gpt-4o
HYPOTHESIS_GENERATOR_MODEL=gpt-4o
EVIDENCE_BINDER_MODEL=gpt-4o
CANDIDATE_EVALUATOR_MODEL=gpt-4o
EVENT_OBSERVER_MODEL=gpt-4o
ANCHOR_EXTRACTOR_MODEL=gpt-4o
REASONING_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o-mini

# ── Optional overrides ────────────────────────────────────
# YOLO_MODEL_PATH=/custom/path/to/yolo11n.pt
ENVEOF
  echo "[setup] Created $ENV_FILE — please edit it with your paths and API keys."
else
  echo "[setup] $ENV_FILE already exists, skipping."
fi

# ── Python deps ───────────────────────────────────────────
echo ""
echo "[setup] Installing Python dependencies..."
pip install -q openai tavily-python pydantic pyyaml \
  torch ultralytics "opencv-python>=4.8" scenedetect \
  open-clip-torch paddlepaddle paddleocr \
  2>&1 | tail -5

# ── Pre-download CV models ────────────────────────────────
echo ""
echo "[setup] Pre-downloading CV models (YOLO, PaddleOCR, CLIP)..."
python3 "$SCRIPT_DIR/scripts/prepare_v1_local_models.py" 2>&1

# ── Validate ──────────────────────────────────────────────
echo ""
echo "[setup] Checking API config and model routing..."
python3 "$SCRIPT_DIR/scripts/check_hybrid_env.py" 2>&1

echo ""
echo "=============================================="
echo "Setup complete."
echo "Next: cd \$VDR_BENCHMARK_HOME && ./benchmark/baselines/videodeepresearch_demo_adapter/run_pilot_full.sh"
echo "=============================================="
