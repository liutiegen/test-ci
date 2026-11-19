# =============================================================================
# Stage 1.1 镜像：并行 PTQ 量化 + lm-eval 评估 + 困惑度
# 目标镜像（必须与 k8s/deployment_stage_1_1.yaml 中 image 一致）：
#   test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-1-v1
# 构建（在项目根目录执行）：
#   docker build -f pipelines/stage_1_1/Dockerfile -t <上述镜像> .
# 启用镜像源：--build-arg USE_MIRROR=1
# =============================================================================
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG USE_MIRROR=0
ARG PIP_INDEX=https://pypi.org/simple
ARG APT_MIRROR=archive.ubuntu.com
ARG UV_GH_BASE=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# 按需切换 apt 镜像源
RUN if [ "$USE_MIRROR" = "1" ]; then \
        sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g; s|security.ubuntu.com|${APT_MIRROR}|g" \
            /etc/apt/sources.list ; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl ca-certificates bash && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && rm -rf /var/lib/apt/lists/*

# 安装 uv（启用镜像时走 GitHub 代理）
RUN if [ -n "$UV_GH_BASE" ]; then \
        curl -LsSf "${UV_GH_BASE}/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/uv.tgz ; \
        tar -xzf /tmp/uv.tgz -C /tmp && cp /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv ; \
    else \
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh ; \
    fi

WORKDIR /app

# 先装依赖（利用层缓存）
COPY pyproject.toml ./
RUN if [ "$USE_MIRROR" = "1" ]; then \
        uv pip install --python /usr/bin/python --index-url "${PIP_INDEX}" --system \
            "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 ; \
        uv pip install --python /usr/bin/python --index-url "${PIP_INDEX}" --system \
            "llmcompressor" "lm-eval==0.4.14" "transformers" "datasets" "accelerate" "pydantic" "pyyaml" ; \
    else \
        uv pip install --python /usr/bin/python --system \
            "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 ; \
        uv pip install --python /usr/bin/python --system \
            "llmcompressor" "lm-eval==0.4.14" "transformers" "datasets" "accelerate" "pydantic" "pyyaml" ; \
    fi

# 拷贝源码与流水线
COPY src ./src
COPY configs ./configs
COPY pipelines ./pipelines

RUN uv pip install --python /usr/bin/python --system --no-deps .

RUN chmod +x pipelines/stage_1_1/run_combo.sh
ENTRYPOINT ["bash", "pipelines/stage_1_1/run_combo.sh"]
