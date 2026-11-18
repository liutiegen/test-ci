# GitHub CI 构建测试用最小镜像（自包含，无需额外文件）
FROM python:3.12-slim

# 避免交互式提示、精简日志
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 构建期校验：基础工具链可用
RUN python --version && pip --version

# 运行期冒烟验证：CI 中执行 docker run 应打印该行
CMD ["python", "-c", "print('CI docker build OK')"]
