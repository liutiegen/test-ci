# =============================================================================
# Stage 1.2 镜像：量化模型 QAT 微调（TRL + llmcompressor）
# 目标镜像（必须与 k8s/deployment_stage_1_2.yaml 中 image 一致）：
#   test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-2-v1
# 构建（项目根目录）：
#   docker build -f pipelines/stage_1_2/Dockerfile -t <上述镜像> .
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG USE_MIRROR=0
ARG PIP_INDEX=https://pypi.org/simple
ARG APT_MIRROR=archive.ubuntu.com

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN if [ "$USE_MIRROR" = "1" ]; then \
        sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g; s|security.ubuntu.com|${APT_MIRROR}|g" \
            /etc/apt/sources.list ; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl ca-certificates bash && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --python /usr/bin/python --system \
        "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --python /usr/bin/python --system \
        "llmcompressor" "transformers" "datasets" "accelerate" "trl" \
        "pydantic" "pyyaml"

COPY src ./src
COPY configs ./configs
COPY pipelines ./pipelines

RUN uv pip install --python /usr/bin/python --system --no-deps .

RUN chmod +x pipelines/stage_1_2/run_finetune.sh
ENTRYPOINT ["bash", "pipelines/stage_1_2/run_finetune.sh"]
