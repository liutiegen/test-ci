# GitHub CI 构建测试：使用 uv 管理 Python 依赖（自包含，无需额外文件）
# FROM python:3.12-slim
# FROM docker.1ms.run/library/python:3.12-slim
FROM registry.aliyuncs.com/library/python:3.11-slim

# 容器内 uv 直接写入系统 Python；copy 链接模式适配容器分层；安装即编译字节码
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# 安装 uv（仅走 PyPI，CI 与本地网络路径统一）
RUN pip install --no-cache-dir uv && uv --version

# 通过 uv 安装依赖（锁定版本，保证 CI 可复现）
RUN uv pip install --no-cache \
    requests==2.32.3 \
    httpx==0.28.1 \
    pydantic==2.9.2

# 构建期确认依赖清单
RUN uv pip list

# 运行期冒烟验证：CI 中 docker run 应打印各包版本
CMD ["python", "-c", "import requests, httpx, pydantic; print('uv deps OK:', requests.__version__, httpx.__version__, pydantic.VERSION)"]
