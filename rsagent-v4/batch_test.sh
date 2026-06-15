#!/bin/bash
# Batch test script: loops through all videos and questions in vl_test_question.json
# Usage: bash batch_test.sh [start_index]
#   start_index: 0-based index to resume from (default: 0)

set -e

TEST_JSON="/tmp/rsagent-v4/vl_test_question_new.json"
CONTAINER="egd-train"
VLLM_IP="172.17.0.3"
ENV_OPTS="-e http_proxy=http://agent.baidu.com:8188 -e https_proxy=http://agent.baidu.com:8188 -e no_proxy=${VLLM_IP},127.0.0.1,localhost"
MAX_ROUNDS=12
START_INDEX=${1:-0}

# Extract all video/question pairs into a flat list
PAIRS=$(sudo docker exec ${CONTAINER} python3 -c "
import json
with open('${TEST_JSON}') as f:
    data = json.load(f)
idx = 0
for field in ['test_questions', 'test_questions_xiyuan', 'test_questions_wangyu']:
    if field not in data:
        continue
    for vi, sample in enumerate(data[field]):
        url = sample['video_url']
        vid_id = url.split('v=')[-1].split('&')[0] if 'v=' in url else url.split('/')[-1]
        for qk in sorted(k for k in sample if k.startswith('question')):
            q = sample[qk]
            print(f'{idx}\t{field}\t{vi}\t{qk}\t{vid_id}\t{q}')
            idx += 1
")

TOTAL=$(echo "$PAIRS" | wc -l)
echo "Total test cases: ${TOTAL}, starting from index ${START_INDEX}"
echo "================================================================"

echo "$PAIRS" | while IFS=$'\t' read -r idx field vi qk vid_id question; do
    if [ "$idx" -lt "$START_INDEX" ]; then
        continue
    fi

    OUT_NAME="${field}_v${vi}_${qk}"
    echo ""
    echo "[${idx}/${TOTAL}] ${OUT_NAME}"
    echo "  Video: ${vid_id}"
    echo "  Question: ${question:0:80}..."
    echo "----------------------------------------------------------------"

    # Download video if needed
    sudo docker exec ${ENV_OPTS} ${CONTAINER} bash -c "
        if [ ! -f /tmp/video_cache/${vid_id}.mp4 ]; then
            yt-dlp -f 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]' \
                --merge-output-format mp4 -o '/tmp/video_cache/${vid_id}.mp4' \
                'https://www.youtube.com/watch?v=${vid_id}'
        fi
    "

    # Run agent
    sudo docker exec ${ENV_OPTS} ${CONTAINER} python3 /tmp/rsagent-v4/run_agent.py \
        --video "/tmp/video_cache/${vid_id}.mp4" \
        --question "${question}" \
        --out-dir "/tmp/rsagent-v4/runs/${OUT_NAME}" \
        --max-rounds ${MAX_ROUNDS} \
    && echo "  [OK] Done" \
    || echo "  [FAIL] Error on ${OUT_NAME}"

    echo "================================================================"
done

echo ""
echo "All tests complete. Copy results with:"
echo "  sudo docker cp ${CONTAINER}:/tmp/rsagent-v4/runs /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4/runs"
