# Stage 3 运行校验清单（含全部校验暂停点）

> 对应工作流 Stage 3.0 ~ 3.3。每个校验项都给出命令、预期结果与暂停点；
> 任一不通过即暂停要求用户修正（Rule 1）。测试清单文件位于 `tests/cluster/`。

---

## Stage 3.0 初步验证

### 3.0.1.1 验证能正常访问 K8s 集群

- [ ] 命令：
  ```bash
  kubectl get nodes
  kubectl get events --sort-by=.metadata.creationTimestamp
  ```
- [ ] 预期：节点列表正常返回，无报错；events 可正常列出
- [ ] 结论：通过 / 不通过（不通过则暂停，检查 kubeconfig 与集群连通性）

### 3.0.1.2 验证 RRSA 已经配置完成

- [ ] 检查工作负载使用的 ServiceAccount 存在：
  ```bash
  kubectl get serviceaccount quantflow-rrsa
  ```
- [ ] 检查该 SA 已绑定 OIDC 角色（RRSA 注解）：
  ```bash
  kubectl get serviceaccount quantflow-rrsa -o yaml
  ```
- [ ] 预期：含 RAM Roles 注解，且对应角色具备 OSS 访问权限
- [ ] 结论：通过 / 不通过（不通过则暂停，由用户配置 RRSA 授权）

### 3.0.1.3 验证节点自动扩缩容

- [ ] 1. 先确认当前不存在测试节点：
  ```bash
  kubectl get nodes
  ```
- [ ] 2. 应用压测清单（`tests/cluster/stress_demo.yaml`）：
  ```bash
  kubectl apply -f tests/cluster/stress_demo.yaml
  kubectl get pods -l app=stress-demo -w
  kubectl get nodes
  ```
- [ ] 预期：自动新增 **2 个节点和 2 个 Pod**，Pod 最终 `Running`
- [ ] 3. **[P] 暂停**：要求用户删除对应的 node 和 pod
  ```bash
  kubectl delete -f tests/cluster/stress_demo.yaml
  ```
- [ ] 结论：通过 / 不通过（不通过则暂停，检查节点池与 CA 配置）

### 3.0.1.4 验证能把 OSS 挂载到 Pod 文件系统

- [ ] 1. 已选用轻量镜像（`busybox:1.36`，约几 MB）
- [ ] 2. 测试清单已就绪：
  - `tests/cluster/oss_mount_pv.yaml`
  - `tests/cluster/oss_mount_pvc.yaml`
  - `tests/cluster/oss_mount_deployment.yaml`
- [ ] 3. **[P] 暂停**：请求用户审核上述清单
- [ ] 4. 应用并验证可读：
  ```bash
  kubectl apply -f tests/cluster/oss_mount_pv.yaml \
                 -f tests/cluster/oss_mount_pvc.yaml \
                 -f tests/cluster/oss_mount_deployment.yaml
  kubectl exec -it deploy/oss-mount-test -- ls -la /data
  ```
- [ ] 预期：能列出 bucket 根目录内容，文件可读
- [ ] 5. **[P] 暂停**：请求用户删掉对应的 pod 和 node
  ```bash
  kubectl delete -f tests/cluster/oss_mount_deployment.yaml \
                 -f tests/cluster/oss_mount_pvc.yaml \
                 -f tests/cluster/oss_mount_pv.yaml
  ```
- [ ] 结论：通过 / 不通过（不通过则暂停，检查 CSI 驱动与 RRSA 权限）

### 3.0.2 验证模型、数据集

- [ ] 验证存在于远程 OSS：
  ```bash
  ossutil ls oss://test-llm-compressor/Qwen2.5-0.5B-Instruct/
  ossutil ls oss://test-llm-compressor/ultrachat_200k/
  ```
- [ ] 预期：模型权重、tokenizer 与数据集 parquet 均在列
- [ ] 验证可创建挂载模型、数据集的 Pod（复用 3.0.1.4 的挂载方式）：
  ```bash
  kubectl exec -it deploy/oss-mount-test -- ls -la /data/Qwen2.5-0.5B-Instruct
  kubectl exec -it deploy/oss-mount-test -- ls -la /data/ultrachat_200k
  ```
- [ ] 结论：通过 / 不通过（不通过则暂停，先同步 OSS 数据）

### 3.0.3 验证镜像文件

- [ ] 验证三个 tag 存在于云镜像仓库：
  ```bash
  REPO=test-acr-business-registry.cn-shanghai.cr.aliyuncs.com/test-llm-compressor/test
  docker manifest inspect ${REPO}:stage-1-1-v1
  docker manifest inspect ${REPO}:stage-1-2-v1
  docker manifest inspect ${REPO}:stage-1-3-v1
  ```
- [ ] 验证镜像已存于数据盘快照：
  - **[P] 暂停**：要求用户创建该节点
  - 在物理节点上通过 crictl 查看本地是否已有镜像：
    ```bash
    crictl images | grep test-llm-compressor
    ```
- [ ] 结论：通过 / 不通过（不通过则暂停，补推镜像或重做快照）

---

## Stage 3.1 实际量化模型

**前置条件**
- [ ] K8s 集群已配置成功：`kubectl get nodes` 返回若干节点
- [ ] 相关镜像准备完备：用户确认
- [ ] 模型文件已存在于云存储：3.0.2 已验证

**流程**
- [ ] 应用 PV/PVC：
  ```bash
  kubectl apply -f pipelines/stage_1_1/k8s/pv_stage_1_1.yaml \
                 -f pipelines/stage_1_1/k8s/pvc_stage_1_1.yaml
  ```
- [ ] 应用 9 个并行 Job：
  ```bash
  kubectl apply -f pipelines/stage_1_1/k8s/deployment_stage_1_1.yaml
  kubectl get jobs -l stage=stage-1-1
  kubectl get pods -l stage=stage-1-1 -w
  ```
- [ ] **[P] 暂停**：人工审核验证
  - [ ] 9 个 Job 全部 `Complete`，无 Failed
  - [ ] 量化模型落于 `/data/output/models/<run_id>`
  - [ ] 指标落于 `/data/output/metrics/<run_id>.json` 与 `_perplexity.json`
- [ ] 选优：
  ```bash
  python scripts/select_best.py
  cat /data/output/best_selection.json
  ```
- [ ] 校验：`best_selection.json` 含 `best_run_id`，且保留率 ≥ 0.98；不达标则暂停

**交付物**
- [ ] 各 Pod 的量化模型文件与指标
- [ ] `best_selection.json` 与 `quantization_report.md`

## Stage 3.2 微调模型

**前置条件**
- [ ] `best_selection.json` 已生成且方案达标

**流程**
- [ ] 应用并运行：
  ```bash
  kubectl apply -f pipelines/stage_1_2/k8s/pv_stage_1_2.yaml \
                 -f pipelines/stage_1_2/k8s/pvc_stage_1_2.yaml \
                 -f pipelines/stage_1_2/k8s/deployment_stage_1_2.yaml
  kubectl get job finetune-stage-1-2 -w
  ```
- [ ] **[P] 暂停**：人工审核验证
  - [ ] Job `Complete`
  - [ ] 微调模型落于 `/data/output/models/<best_run_id>-finetuned`
  - [ ] 含 `finetune_manifest.json`
- [ ] 结论：通过 / 不通过（不通过则暂停排查）

## Stage 3.3 部署评估最终模型

**流程**
- [ ] 应用并运行三模型并行对比：
  ```bash
  kubectl apply -f pipelines/stage_1_3/k8s/pv_stage_1_3.yaml \
                 -f pipelines/stage_1_3/k8s/pvc_stage_1_3.yaml \
                 -f pipelines/stage_1_3/k8s/deployment_stage_1_3.yaml
  kubectl get jobs -l stage=stage-1-3
  ```
- [ ] **[P] 暂停**：人工审核验证
  - [ ] 3 个 Job 全部 `Complete`
  - [ ] 服务基准：`/data/output/serving/{baseline,compressed,finetuned}.json`
  - [ ] 准确率：`/data/output/metrics/stage13-*.json`
- [ ] 汇总对比维度：
  - 准确率（core 套件）与综合保留率
  - 吞吐（output token TPS）、TTFT、端到端延迟分位数
  - 模型体积
- [ ] 最终结论：确认微调模型在稳住精度的同时保留压缩收益；否则暂停复盘

---

## 门禁总表

| 阶段 | 关键门禁 | 不通过处理 |
|---|---|---|
| 3.0.1 | 集群/RRSA/扩缩容/OSS 挂载 | 暂停，用户修正基础设施 |
| 3.0.2 | 模型与数据集在 OSS 且可挂载 | 暂停，先同步数据 |
| 3.0.3 | 镜像在仓库且有快照缓存 | 暂停，补推/重做快照 |
| 3.1 | 9 Job 完成 + 选优达标(≥0.98) | 暂停，重跑或调整方案 |
| 3.2 | 微调 Job 完成 | 暂停，排查训练 |
| 3.3 | 三模型对比完整 | 暂停，复盘结论 |
