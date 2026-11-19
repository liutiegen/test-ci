# =============================================================================
# Stage 1.3 镜像：vLLM 离线服务基准 + lm-eval 准确率
# 目标镜像（必须与 k8s/deployment_stage_1_3.yaml 中 image 一致）：
#   test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-3-v1
# 基础镜像使用官方 vllm（已含匹配的 torch/CUDA），再补 lm-eval 与本包。
# 构建（项目根目录）：
#   docker build -f pipelines/stage_1_3/Dockerfile -t <上述镜像> .
# 注意：VLLM_BASE 的 tag 应与本地 compress/vllm 验证过的版本对应。
# =============================================================================
ARG VLLM_BASE=vllm/vllm-openai:v0.11.0
FROM ${VLLM_BASE}

USER root
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 补 lm-eval（vllm 后端）与配置依赖；vllm/torch 已在基础镜像中
RUN pip install --no-cache-dir "lm-eval==0.4.14" "pyyaml" "pydantic"

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY pipelines ./pipelines

RUN pip install --no-cache-dir --no-deps .

RUN chmod +x pipelines/stage_1_3/run_benchmark.sh
ENTRYPOINT ["bash", "pipelines/stage_1_3/run_benchmark.sh"]
