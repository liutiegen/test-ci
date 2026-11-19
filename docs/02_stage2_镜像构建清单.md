# Stage 2 镜像构建清单（含人工暂停点）

> 对应工作流 Stage 2.1。本阶段**不自动执行构建**，按工作流要求暂停，由用户手动
> 构建并推送，再做数据盘快照。每个镜像路径必须与对应 K8s 文件中的 `image` 一致。

## 0. 构建前确认

- [ ] 已登录目标仓库
  ```bash
  docker login test-acr-business-registry.cn-shanghai.cr.aliyuncs.com
  ```
- [ ] 构建上下文为项目根目录（`pyproject.toml` 所在目录）
- [ ] 确认 `pyproject.toml`、`src/`、`configs/`、`pipelines/` 均在根目录下

镜像仓库前缀（固定）：
```
test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test
```

## 1. [P] 暂停点：构建三个目标镜像

### 1.1 Stage 1.1 镜像（tag：`stage-1-1-v1`）
- [ ] 用户执行：
  ```bash
  # 正常网络
  docker build -f pipelines/stage_1_1/Dockerfile \
    -t test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-1-v1 .

  # 若需镜像源（use_mirror=true）
  docker build -f pipelines/stage_1_1/Dockerfile \
    --build-arg USE_MIRROR=1 \
    --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    --build-arg APT_MIRROR=mirrors.aliyun.com \
    --build-arg UV_GH_BASE=https://ghproxy.cn/https://github.com \
    -t test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-1-v1 .
  ```
- [ ] 推送：
  ```bash
  docker push test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-1-v1
  ```

### 1.2 Stage 1.2 镜像（tag：`stage-1-2-v1`）
- [ ] 用户执行：
  ```bash
  docker build -f pipelines/stage_1_2/Dockerfile \
    -t test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-2-v1 .
  docker push test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-2-v1
  ```

### 1.3 Stage 1.3 镜像（tag：`stage-1-3-v1`）
- [ ] 用户执行（确认基础镜像 `VLLM_BASE` 与验证版本一致）：
  ```bash
  docker build -f pipelines/stage_1_3/Dockerfile \
    -t test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-3-v1 .
  docker push test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test:stage-1-3-v1
  ```

## 2. 构建后校验（每个 tag 都要过）

| 检查项 | 命令 | 1.1 | 1.2 | 1.3 |
|---|---|---|---|---|
| 远端 manifest 存在 | `docker manifest inspect <repo>:<tag>` | [ ] | [ ] | [ ] |
| tag 与 K8s image 一致 | 比对各 deployment | [ ] | [ ] | [ ] |
| 入口脚本可执行位 | 镜像内 `run_*.sh` | [ ] | [ ] | [ ] |

## 3. [P] 暂停点：数据盘快照预热

- [ ] 暂停提醒用户：把构建好的镜像预先保存到数据盘快照，从而在节点扩容时加速拉取
- [ ] 用户确认快照已包含三个 tag 后，方可进入 Stage 3

## 暂停与恢复规则

- 任一步骤构建/推送失败：**暂停**，输出错误，由用户修正后从失败处继续
- 未完成全部三个镜像与快照前，不进入 Stage 3
