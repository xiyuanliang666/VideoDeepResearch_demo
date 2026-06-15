# RSAgent-V4 测试教程

## 1. 启动 vLLM 服务容器

```bash
sudo docker run -d \
   --name qwen3-vl-32b-vllm \
   --gpus '"device=0,1"' \
   --ipc=host \
   --shm-size=16g \
   -p 8000:8000 \
   -v /ssd1/wangyu144:/root/.cache/huggingface \
   -e HF_HUB_OFFLINE=1 \
   -e http_proxy=http://agent.baidu.com:8188 \
   -e https_proxy=http://agent.baidu.com:8188 \
   -e no_proxy=localhost,127.0.0.1,::1,0.0.0.0 \
   -e HTTP_PROXY=http://agent.baidu.com:8188 \
   -e HTTPS_PROXY=http://agent.baidu.com:8188 \
   -e NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0 \
   vllm/vllm-openai:latest-cu129 \
   --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb \
   --served-model-name Qwen/Qwen3-VL-32B-Instruct \
   --host 0.0.0.0 \
   --port 8000 \
   --tensor-parallel-size 2 \
   --gpu-memory-utilization 0.99 \
   --max-model-len 65536 \
   --trust-remote-code
```

等待启动完成，检查状态：

```bash
sudo docker logs -f qwen3-vl-32b-vllm  # 看到 "Started server process" 即就绪
```

## 2. 获取 vLLM 容器 IP

```bash
VLLM_IP=$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' qwen3-vl-32b-vllm)
echo "vLLM IP: ${VLLM_IP}"
```

验证服务可用：

```bash
sudo docker exec egd-train curl -s http://${VLLM_IP}:8000/v1/models
```

## 3. 配置 .env

确保 `rsagent-v4/.env` 中 `VLLM_BASE_URL` 指向正确 IP：

```
VLLM_BASE_URL=http://172.17.0.3:8000/v1
VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
VLLM_API_KEY=EMPTY
SERPER_API_KEY=622434d8cf81450ef5efdaa2f18ebdd405d69b22
JINA_API_KEY=jina_c41d8b4d53e84a8cb34b8996f47ab675BV81LvlC0bXRQWf4wslp0SMZks9-
RESEARCH_IMAGE_MAX_SIDE=1024
```

如果 IP 变了，用 sed 替换：

```bash
sed -i "s|VLLM_BASE_URL=http://.*:8000/v1|VLLM_BASE_URL=http://${VLLM_IP}:8000/v1|" /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4/.env
```

## 4. 拷贝代码到工作容器

```bash
# 清除旧版本并拷入新代码
sudo docker exec egd-train rm -rf /tmp/rsagent-v4
sudo docker cp /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4 egd-train:/tmp/rsagent-v4

# 拷入测试数据文件
sudo docker cp /home/wangyu144/vl_script/vl_test_question.json egd-train:/tmp/rsagent-v4/vl_test_question.json
sudo docker cp /home/wangyu144/vl_script/vl_test_question_new.json egd-train:/tmp/rsagent-v4/vl_test_question_new.json
```

## 5. 安装依赖（首次）

```bash
sudo docker exec egd-train pip install json5 httpx decord Pillow
```

## 6. 单问题测试

### 方式A：直接调用 run_agent.py

```bash
# 环境变量（每次都需要）
ENV_OPTS="-e http_proxy=http://agent.baidu.com:8188 -e https_proxy=http://agent.baidu.com:8188 -e no_proxy=172.17.0.3,127.0.0.1,localhost"

# 下载视频（已缓存则跳过）
VIDEO_ID="BkiTScinYOQ"
sudo docker exec ${ENV_OPTS} egd-train bash -c "
  [ -f /tmp/video_cache/${VIDEO_ID}.mp4 ] || \
  yt-dlp -f 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]' \
    --merge-output-format mp4 -o '/tmp/video_cache/${VIDEO_ID}.mp4' \
    'https://www.youtube.com/watch?v=${VIDEO_ID}'
"

# 运行 agent
sudo docker exec ${ENV_OPTS} egd-train python3 /tmp/rsagent-v4/run_agent.py \
  --video "/tmp/video_cache/${VIDEO_ID}.mp4" \
  --question "视频里出现的所有场比赛中，哪一场C罗个人的进球数最多？" \
  --out-dir /tmp/rsagent-v4/runs/my_test \
  --max-rounds 12
```

### 方式B：使用 test_run.py

修改 `test_run.py` 中的字段和索引后运行：

```bash
sudo docker exec ${ENV_OPTS} egd-train bash -c "cd /tmp/rsagent-v4 && python3 test_run.py"
```

## 7. 批量测试

使用 `batch_test.sh` 自动遍历 JSON 中所有视频的所有问题：

```bash
# 运行全部（从第0个开始）
bash /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4/batch_test.sh

# 从第N个问题断点续跑（跳过前N个）
bash /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4/batch_test.sh 5
```

输出目录命名：`runs/{字段名}_v{视频序号}_{question编号}/`

## 8. 拷贝结果到本地

```bash
# 拷贝全部结果
sudo docker cp egd-train:/tmp/rsagent-v4/runs /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4/runs

# 拷贝单个
sudo docker cp egd-train:/tmp/rsagent-v4/runs/my_test /home/wangyu144/VideoDeepResearch_demo_temp/rsagent-v4/runs/my_test
```

## 9. 结果文件结构

每个测试输出目录包含：

```
runs/test_name/
├── planning.json          # 规划阶段（子问题拆分）
├── evidence_rounds/       # 每轮 evidence loop 的详细 trace
│   ├── round_001.json
│   ├── round_002.json
│   └── ...
├── binding.json           # 证据绑定阶段
├── final_answer.json      # 最终答案生成
└── result.json            # 汇总结果（含 sub_questions 和 final_answer）
```

## 10. 常见问题

| 问题 | 解决 |
|------|------|
| `Connection refused` | 检查 .env 中 VLLM_BASE_URL 的 IP 是否正确 |
| `Context length exceeded` | 增大 vLLM 的 `--max-model-len` |
| vLLM OOM | 降低 `--gpu-memory-utilization` 到 0.95，或增加 GPU 数 |
| `yt-dlp` 失败 | 确保 proxy 环境变量正确传入 |
| 视频下载后测试数据丢失 | 容器 /tmp 非持久化，重启后需重新 cp 代码 |
