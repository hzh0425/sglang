# Unified Tree 直连 Mooncake 实现计划

> 本计划在当前会话内直接执行。第一版只做 `Device ↔ Mooncake`，不创建 Host/L2 Pool；远端命中和加载以完整 Bundle 为原子单位。

## 目标与约束

- `UnifiedRadixCache` 继续负责 local tree、Component 一致性以及 TP/CP 命中收敛。
- Connector 只处理 `query / load / store / poll`，接口中只出现 `device_indices`，不暴露 L2/host 概念。
- 复用 `PoolTransfer`、`PoolHitPolicy`、`SidecarPoolSpec` 和 `MooncakeStore.batch_*_v2`。
- 第一版完整支持 DeepSeek V4 unified-kv 的逻辑 Full Anchor，以及 C4、C4 Indexer、C128 Sidecar；Full 与全部必需 Sidecar 要么一起成功，要么整体失败。
- 调度器内部暂时复用 `host_hit_length` 作为“待加载 token 数”的兼容字段，避免扩散改动；Connector 和 tree 的新接口不依赖 Host Pool。

## Task 1：先固定 Bundle 和 Mooncake 直连契约

**文件：**

- 新增：`test/registered/unit/mem_cache/test_unified_radix_cache_external.py`
- 新增：`python/sglang/srt/mem_cache/external_cache_connector.py`

**步骤：**

1. 写失败测试，覆盖：
   - Query 只接受所有必需 Pool 的共同连续前缀。
   - TP/CP 结果取最小值后裁剪命中。
   - Load 任一 Pool 失败时释放整 Bundle，且不修改 tree。
   - Store 完成事件只解锁一次。
2. 定义精简接口：
   - `ExternalCacheConnector.query(key, local_tokens, transfers)`。
   - `ExternalCacheConnector.load(key, device_indices, transfers)`。
   - `ExternalCacheConnector.store_async(key, device_indices, transfers)`。
   - `poll_completed / reset / close`；marker、allocator rollback 和节点锁由 tree 管理。
3. 定义 marker/result 数据结构，仅保存请求快照、页 key、命中 token 数和 Bundle 描述。
4. 运行：

```bash
python -m pytest -q test/registered/unit/mem_cache/test_unified_radix_cache_external.py
```

## Task 2：构建不分配 Host 内存的 Device Pool 视图

**文件：**

- 新增：`python/sglang/srt/mem_cache/external_cache_pool.py`
- 修改：`python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- 测试：`test/registered/unit/mem_cache/test_unified_radix_cache_external.py`

**步骤：**

1. 写失败测试，验证逻辑 Full Anchor 没有物理 Buffer，DSV4 Sidecar 能从 token slot 推导 page row 和逐层 GPU 指针。
2. 实现 `LogicalDevicePool` 与 `PagedDevicePool`：只暴露 `page_size`、`kv_buffer`、`get_hybrid_pool_buffer()`、`get_page_buffer_meta()`。
3. 实现 DSV4 Pool 组装：
   - KV：逻辑 Anchor。
   - C4 / C4 Indexer / C128：复用 KV device slot。
   - unified-kv 的 SWA 不远端存储，由 local reprefill 重建。
4. 将 Mooncake 多 Buffer 打包从 C4 特判改为通用规则，确保一个逻辑页可以映射多层 GPU Buffer。
5. 运行本 Task 测试并执行 `ruff`/`pyright` 覆盖新增文件。

## Task 3：实现 Mooncake Connector

**文件：**

- 新增：`python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- 新增：`python/sglang/srt/mem_cache/storage/mooncake_store/__init__.py`
- 修改：`python/sglang/srt/mem_cache/registry.py`
- 测试：`test/registered/unit/mem_cache/test_unified_radix_cache_external.py`

**步骤：**

1. 写 Fake MooncakeStore 测试，覆盖 query、全 Bundle load、异步 store、失败回滚和完成轮询。
2. 用 model/layout/page/并行配置生成稳定 namespace；`extra_key` 进入页 key，CP rank 进入 backend tag。
3. 复用 `MooncakeStore` 的 setup、`batch_exists_v2` 和 `batch_get_v2/batch_set_v2`；DSV4 的逻辑 Full Anchor 不执行 v1 I/O。
4. 直接注册 GPU allocation 的底层 storage，去重共享/切片 Buffer；外部传入的 `device_indices` 只在 Connector 内转换为 MooncakeStore 兼容视图。
5. 用有界线程池执行 Store；`poll_completed()` 返回操作 ID 与状态，tree 持有的节点锁在完成后释放。

## Task 4：把 Connector 接入 Unified Tree

**文件：**

- 修改：`python/sglang/srt/mem_cache/unified_radix_cache.py`
- 修改：`python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`
- 修改：`python/sglang/srt/mem_cache/registry.py`
- 测试：`test/registered/unit/mem_cache/test_unified_radix_cache_external.py`

**步骤：**

1. `init_external_cache()` 安装 Connector、Sidecar Specs 和外部 Pool 视图，但不创建 `HybridCacheController`。
2. `init_external_cache()` 同时开启稳定 hash/key 生命周期；`match_prefix()` 先取所有 Component 同意的 local `last_device_node`，再对缺失尾部构造完整 Bundle 并 query；tree 对命中页数做 TP/CP `MIN`。
3. `init_load_back()` 使用两阶段分布式协议：先分配并对“分配成功”做 TP/CP `MIN`，一致成功后才执行 I/O；捕获各 rank I/O 异常，再对“完整 Bundle 成功”做一次 `MIN`；最后所有 rank 统一 `_insert_helper()` commit 或统一释放全部 slots。
4. DSV4 unified-kv 的 SWA validator/reprefill 同时识别 external connector，保证远端只恢复稳定 Sidecar，SWA ring 仍由尾部 prefill 重建。
5. `cache_finished_req()` 完成 `_insert_helper()` 后、释放请求原锁前，从实际 tree-owned slots 构造 immutable Store Bundle，并取得独立节点锁后异步写 Mooncake；future 成功、失败或异常都通过同一完成路径只解锁一次。
6. `release_aborted_request/reset/shutdown` 清理 marker 和异步操作。`reset()` 必须先 drain store future 再释放锁和 device slots，且绝不调用 `MooncakeStore.clear()`，保证 `/flush_cache` 只清 local tree。

## Task 5：接入参数、调度器生命周期和兼容校验

**文件：**

- 修改：`python/sglang/srt/managers/scheduler.py`
- 修改：`python/sglang/srt/server_args.py`
- 修改：`python/sglang/srt/mem_cache/base_prefix_cache.py`（仅在需要通用 capability 时）
- 测试：新增/现有 ServerArgs 与 registry 测试

**步骤：**

1. `--radix-cache-backend mooncake` 懒加载内置 factory；factory 仍创建 `UnifiedRadixCache`，随后安装 Mooncake Connector。
2. 禁止与 HiCache/L2、LMCache、FlexKV、禁用 radix cache 等冲突组合；保留 tree 侧 TP/CP 收敛能力。
3. Scheduler 对外部 backend 轮询完成事件，并在 waiting/chunked/abort 路径释放 query marker。
4. 保持现有 `host_hit_length` admission 兼容，不修改 Req 数据布局。

## Task 6：本地回归与代码审查

**步骤：**

1. 运行新增测试和 MooncakeStore 现有单测。
2. 运行 Unified Tree 单测：

```bash
python -m pytest -q test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py
```

3. 运行技能要求的压力测试：

```bash
python test/registered/unit/mem_cache/bench_unified_radix_cache.py \
  --num-seqs 5000 --verify --components mamba
```

4. 对所有修改文件运行格式、lint、类型/编译检查；确认 `git diff --check`。
5. 只提交本次新增/修改文件，不纳入用户已有的测试和文档改动。

## Task 7：上传 test-4-sglang 并做 DSV4 + Direct Mooncake 验收

**步骤：**

1. 将当前提交制作成 git bundle，通过 DBaaS/Kubernetes 上传到 `test-4-sglang`，在远端干净分支切到该提交。
2. 启动 `mooncake_master` 和 Mooncake store 服务，RDMA 明确绑定可用的 `mlx5_*` 设备；SGLang client 使用 `global_segment_size=0`。
3. 用模型：

```text
/home/t4/models/deepseek-v4-flash-fp8/sgl-project/DeepSeek-V4-Flash-FP8/
```

启动 `--radix-cache-backend mooncake`，确认日志包含 Unified Tree + Mooncake Connector 初始化，且没有 Host Pool 分配。
4. 对同一长 prompt 执行：首次请求写远端 → 明确等待 store future 完成且节点锁归零，并确认远端对象存在 → `/flush_cache` 仅清 local tree → 第二次请求从 Mooncake query/load。
5. 收集：
   - 第二次请求的 remote hit/load 日志与命中 token 数。
   - 输出一致性。
   - `Sanity check FAILED`、scheduler exception、Traceback、AssertionError、RDMA/transfer failure 均为 0。
6. 如远端暴露实现问题，按最小修改修复并重复本地与远端验证，直到链路通过。
