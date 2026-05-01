#!/usr/bin/env bash
set -euo pipefail
base="recovery/reconstructed"
out="recovery/restored_guess"
rm -rf "$out"
mkdir -p "$out"/{rsagent,rsagent-v2,rsagent-v3}/{video_dr_agent,scripts,prompts,legacy}

pick_one() {
  local pattern="$1"
  local ver="$2"
  local sha
  sha=$(rg -l "$pattern" "$base" | rg "\\.(py|py\\.txt)$" | while read -r f; do
    if [ "$ver" = "v3" ]; then
      rg -q "rsagent-v3|evidence_policy" "$f" && echo "$f"
    elif [ "$ver" = "v2" ]; then
      rg -q "rsagent-v2|force_focus" "$f" && ! rg -q "rsagent-v3|evidence_policy" "$f" && echo "$f"
    else
      ! rg -q "rsagent-v2|rsagent-v3|evidence_policy" "$f" && echo "$f"
    fi
  done | head -n1)
  if [ -z "${sha:-}" ]; then
    sha=$(rg -l "$pattern" "$base" | rg "\\.(py|py\\.txt)$" | head -n1 || true)
  fi
  echo "$sha"
}

copy_norm() {
  local src="$1"
  local dst="$2"
  [ -n "$src" ] || return 0
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

for v in base v2 v3; do
  case "$v" in
    base) root="$out/rsagent";;
    v2) root="$out/rsagent-v2";;
    v3) root="$out/rsagent-v3";;
  esac

  copy_norm "$(pick_one 'Video DeepResearch agent framework' "$v")" "$root/run_video_dr_agent.py"
  copy_norm "$(pick_one 'Protocols for Video DeepResearch' "$v")" "$root/video_dr_agent/protocols.py"
  copy_norm "$(pick_one 'Parse <tool_call> blocks' "$v")" "$root/video_dr_agent/parsing.py"
  copy_norm "$(pick_one 'Research-side message buffer' "$v")" "$root/video_dr_agent/research_context.py"
  copy_norm "$(pick_one 'Critic-side history' "$v")" "$root/video_dr_agent/critic_context.py"
  copy_norm "$(pick_one 'vLLM OpenAI-compatible chat for Qwen3-VL' "$v")" "$root/video_dr_agent/qwen3vl_client.py"
  copy_norm "$(pick_one 'Critic via OpenAI-compatible' "$v")" "$root/video_dr_agent/openai_critic.py"
  copy_norm "$(pick_one 'Default research \(Qwen3-VL\) system prompt resolution' "$v")" "$root/video_dr_agent/research_prompt.py"
  copy_norm "$(pick_one 'Critic system prompt and staged user-prompt templates' "$v")" "$root/video_dr_agent/critic_prompts.py"
  copy_norm "$(pick_one 'Main research loop \(Video DeepResearch plan\)' "$v")" "$root/video_dr_agent/loop.py"
  copy_norm "$(pick_one 'FOCUS 关键帧筛选工具' "$v")" "$root/video_dr_agent/focus_keyframe_tool.py"
  copy_norm "$(pick_one '多源网页搜索工具' "$v")" "$root/video_dr_agent/serper_web_search_tool.py"
  copy_norm "$(pick_one 'Placeholder tool / critic implementations' "$v")" "$root/video_dr_agent/stubs.py"
  copy_norm "$(pick_one '示例工具：`ToolDispatcher`' "$v")" "$root/video_dr_agent/tools_example.py"
  copy_norm "$(pick_one 'Load `video_dr_agent/.env`' "$v")" "$root/video_dr_agent/env_loader.py"
  copy_norm "$(pick_one 'Offline smoke: core ``run_loop``' "$v")" "$root/scripts/smoke_offline_loop.py"
  copy_norm "$(pick_one 'Lightweight checks that the Video DeepResearch stack imports' "$v")" "$root/scripts/run_architecture_checks.py"

  if [ "$v" = "v3" ]; then
    copy_norm "$(pick_one 'Evidence-tier policy \(rsagent-v3\)' "$v")" "$root/video_dr_agent/evidence_policy.py"
  fi

done

find "$out" -type f | wc -l
find "$out" -type f | sed -n '1,200p'
