# UnifiedTree L2-Only Mode 第一阶段实现计划

## 目标描述

第一阶段只做 `UnifiedRadixCache` 本体能力建设，不接 `PD` 链路。

目标是在普通运行路径下，为 `UnifiedTree` 增加一种新的 `L2-only` 运行模式：

- `L1`：device KV 仍然存在，但完全按 `request` 生命周期管理，不进入 device radix tree
- `L2`：host KV 继续由 `UnifiedRadixCache` 维护 prefix tree，并负责后续前缀匹配与 `load_back`

完成后，单机/普通调度路径应具备下面的闭环：

1. 请求第一次执行时，KV 正常落到 device。
2. 请求结束后，device KV 可以异步发布到 host `UnifiedTree`。
3. 后续相同或共享前缀的请求，只依赖 host tree 命中前缀，再在准入前 `load_back` 到 device。
4. 整个过程不引入第二个公开 cache owner；外部仍然只看到 `UnifiedRadixCache`。

这一阶段的成功标准不是“PD 可用”，而是“`UnifiedTree` 自己已经能稳定承载 `request-owned L1 + host-prefix L2` 语义，并有单测和 e2e 证明它可复用、可回收、不会提早释放”。

## 验收标准

- AC-1: 模式开关和构造路径明确，默认行为不回退。
  - Positive Tests:
    - 新开关关闭时，`UnifiedRadixCache` 仍按现有 `indexed_device` 语义工作。
    - 新开关打开且 `--enable-hierarchical-cache` 打开时，`CacheInitParams` 和 `registry` 能把 `UnifiedRadixCache` 构造成 `request_owned_device` 模式。
    - 已有 unified tree 和 hicache 测试在不开新开关时行为不变。
  - Negative Tests:
    - 未开启 hierarchical cache 时不能悄悄进入 `L2-only` 模式。
    - 不能通过这个开关绕过现有 `UnifiedRadixCache` / HiCache 的初始化前置条件。

- AC-2: `match_prefix()` / `init_load_back()` 在 `request-owned L1` 下语义闭合。
  - Positive Tests:
    - `L2-only` 模式下，host 命中时 `match_prefix()` 返回 `device_indices=empty`、`last_device_node=None`、`best_match_node` 指向最深 host anchor、`host_hit_length>0`。
    - `init_load_back()` 能在 `req.last_node is None` 的情况下，从 `best_match_node` 正确恢复 device prefix。
    - `req.prefix_indices` 只表达 device 上真实可用的 prefix；恢复后长度与 `host_hit_length` 对齐。
  - Negative Tests:
    - 不允许为了兼容旧路径伪造一个假的 device tree node。
    - host miss 时不应触发多余的 `load_back`。
    - `req.last_node=None` 的情况下不应在锁路径或 restore 路径崩溃。

- AC-3: 调度和 request 生命周期兼容 `req.last_node=None`。
  - Positive Tests:
    - `schedule_policy.match_prefix_for_req()`、`_lock_node()`、`add_one_req()` 在 `L2-only` 模式下可以正常跑通。
    - host 命中的请求在准入前完成 `load_back`，随后能正常 prefill / decode。
    - `num_matched_prefix_tokens` 仍按 `len(device_prefix) + host_hit_length` 计算，不破坏预算估算。
  - Negative Tests:
    - 不能继续把 `req.last_node` 当作 L2 anchor 使用。
    - 不能在 `None` anchor 上调用需要真实 tree ownership 的逻辑并 silently succeed。

- AC-4: 请求结束后的 `L1 -> L2` 发布采用 ack 驱动的 ownership 转移，不能提早释放 device KV。
  - Positive Tests:
    - finish-path 可以把 page-aligned device prefix 异步写入 host，并在 ack 后发布到 host tree。
    - publish 成功后，后续请求能从 host tree 命中相同前缀。
    - overlap 页面在 publish 时被正确去重和释放。
    - ack 完成后，request 的 device KV 和 req_to_token 槽位才真正释放。
  - Negative Tests:
    - publish inflight 时不能先 free request-owned device pages。
    - publish 失败或中止时不能把未归属 tree 的 host pages 泄漏掉。
    - 不能把这一步简单实现成 `write_backup(node)` 对已有树节点的包装。

- AC-5: 第一阶段 e2e 能证明 “只靠 L2 host tree 复用前缀”。
  - Positive Tests:
    - 在 `L2-only` 模式下，对同一 server 连续执行两次 `GSM8K`，第二次执行不 flush cache，且能够稳定命中 host prefix 并完成 `load_back`。
    - 在 `L2-only` 模式下，对同一 server 连续执行两次 `MMLU`，第二次执行不 flush cache，且能够稳定命中 host prefix 并完成 `load_back`。
    - 第二次执行的准确率阈值不低于现有 unified tree 同模型基线，且与第一次执行相比无异常回退。
    - 对采样回放或多轮 KL 用例，第一轮请求结束后，第二轮相同或共享前缀请求产生稳定的 cached tokens 命中。
    - e2e 过程中 worker 不崩溃，server 存活。
  - Negative Tests:
    - 第一轮请求不应虚报 cache hit。
    - 第二次执行不能依赖 `PD/disaggregation` 链路才能复用前缀。
    - 第二次执行若 cached tokens 仍为 0，则说明 host publish 或 load_back 未真正生效，应判失败。

- AC-6: 第一阶段测试矩阵必须覆盖 `FULL attention` 与 `SWA/DSV4` 两条模型线。
  - Positive Tests:
    - `Qwen3-32B` 或同等 full-attention unified tree 用例，覆盖 `GSM8K`、`MMLU`、多轮 KL / cache-hit。
    - `DeepSeek-V4-Flash-FP8` unified tree 用例，覆盖 `SWA/压缩注意力` 路径下的两次 `GSM8K`、两次 `MMLU`、多轮 cache-hit。
    - `DSV4` 用例沿用当前 unified hicache 测试骨架中的关键启动参数，保证真正覆盖 `SWA full<->host` 行为，而不是退化成普通 full-only 路径。
  - Negative Tests:
    - 不能只测 `Qwen32B full attention` 就宣布第一阶段完成。
    - 不能用一个关闭 `SWA` 关键路径的 `DSV4` 启动配置来替代真实 `DSV4 SWA` 覆盖。

- AC-7: 第一阶段严格止步于 `UnifiedTree` 本体，不提前引入第二阶段范围。
  - Positive Tests:
    - 主实现文件集中在 cache、scheduler、common release path、server args 和 unified tree e2e。
    - 单测和 e2e 都不依赖 `decode.py` 的 PD prealloc / transfer 状态机。
  - Negative Tests:
    - 不新增 `PD` 专用状态机分支。
    - 不把 `L3 storage`、`SWA`、`Mamba`、`HiSparse` 一起拉进第一阶段。

## 路径边界

### 上界（最大范围）

第一阶段允许做到下面这个完整闭环：

- 通过一个显式开关把 `UnifiedRadixCache` 切到 `request_owned_device` 模式。
- host prefix 命中只通过 `UnifiedTree` 暴露给调度器。
- 请求结束时，通过新的 request-owned publish 生命周期把 device prefix 发布到 L2 host tree。
- `check_hicache_events()` 或等价轮询点负责推进 publish ack、tree commit 和 deferred release。
- 提供 unit test、server args test、以及至少一个 unified tree e2e。

### 下界（最小范围）

第一阶段最少需要交付：

- `FULL` 组件可用。
- `UnifiedRadixCache` 在 `L2-only` 模式下能正确匹配 host prefix、执行 `load_back`、并在请求结束后把 prefix 发布回 host tree。
- 普通 scheduler 路径可跑通，不需要 `PD`。
- 至少一组单测覆盖：
  - `match_prefix()/init_load_back()`
  - finish-path deferred release
  - publish 后可复用
- 至少一组 e2e 覆盖：
  - 第一轮 warmup
  - 第二轮 host hit
  - server 存活

### 允许的选择

- 可以用 `Enum` 或等价常量表示 `L1ResidencyMode`，但该模式必须进入 `CacheInitParams` 或 `UnifiedRadixCache` 的正式构造参数，不要只做临时布尔分支。
- 可以让 `UnifiedRadixCache.inc_lock_ref(None)` / `dec_lock_ref(None)` 成为显式 no-op，也可以在调用侧先判断 `None`；但整个调用链必须统一，不要一半在 cache 内兜底、一半在上层假设非空。
- 可以把 publish ack 推进挂在 `check_hicache_events()`，也可以挂在已有的统一 poll 点；但不能新建一条只对第一阶段可见的隐藏轮询路径。
- 可以先只支持 `write_through` / host publish 的最小写入策略，再扩展 `write_back`；但计划中要明确第一阶段实际支持哪一种。

### 不允许的选择

- 不新增第二个公开 cache 类型去和 `UnifiedRadixCache` 组合。
- 不复用 `decode.py` / `decode_hicache_mixin.py` 作为第一阶段主路径。
- 不把 `req.last_node` 重新解释成 host anchor。
- 不把 `L1 -> L2` 发布包装成“对树节点做一次普通 `write_backup()`”。
- 不把 `L3 storage` 依赖变成第一阶段前置条件。

## 可行性提示

- 现有 `ChunkCache` 已经证明 `req.last_node=None` 是一个可接受的运行状态，关键在于把 “device anchor 缺失” 和 “prefix 匹配失败” 区分开。
- `UnifiedRadixCache.init_load_back()` 已经具备从 `best_match_node` 恢复 host prefix 的基本能力；第一阶段主要是把“旧 device anchor 为空”这件事纳入正式语义。
- `decode_kvcache_offload_manager.py` 里的 `ongoing_*[ack_id]`、ack 驱动完成、以及 deferred release 模式可以直接借鉴，但第一阶段不应依赖它的对象边界或 `PD` 入口。
- `python/sglang/test/kits/unified_radix_cache_kit.py` 已经提供了多轮 KL / cached-token 断言框架，可以直接作为第一阶段 e2e 的骨架。
- 在 `L2-only` 模式下，只要第一次请求结束后 device tree 不保留前缀，那么第二次请求仍能出现 `cached_tokens > 0`，就足以证明命中来源来自 host tree + load_back；这正是第一阶段最重要的服务级信号。

## 测试体系

第一阶段的测试体系必须分层，不接受只补几条单测就收口。

### 1. 单元测试层

目标：锁住 `UnifiedRadixCache` 新语义本身。

必须覆盖：

- `L2-only` 模式下 `match_prefix()` 返回：
  - `device_indices` 为空
  - `last_device_node is None`
  - `best_match_node` 指向 host anchor
  - `host_hit_length` 正确
- `init_load_back()` 在 `req.last_node is None` 时正确恢复 device prefix。
- request-owned publish 的 overlap 去重、ack 完成、deferred release。
- finish-path 中 publish inflight 时，device KV 不会提前 free。

建议落点：

- `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`
- `test/registered/unit/server_args/test_server_args.py`

### 2. 服务级功能测试层

目标：证明 `UnifiedTree` 在真实 server 中可以只依赖 L2 host tree 完成第二次执行复用。

必须覆盖：

- `GSM8K` 连续执行两次：
  - 第一次执行前允许 flush 或冷启动
  - 第一次执行结束后不 flush cache
  - 第二次执行必须复用 host tree，并能正常 load back
- `MMLU` 连续执行两次：
  - 第一次执行前允许 flush 或冷启动
  - 第一次执行结束后不 flush cache
  - 第二次执行必须复用 host tree，并能正常 load back
- 多轮 KL / cache-hit 回放：
  - 第一轮构建 host tree
  - 第二轮起 `cached_tokens > 0`

判定信号：

- 在 `L2-only` 模式下，第二次执行出现 `cached_tokens > 0`，即可作为 host-hit/load-back 成功的强信号。
- 若需要更强可观测性，允许为第一阶段补充 `cache report` 或等价内部状态，暴露 host-backed cached token / load-back 次数；但这不是先决条件。

### 3. 精度回归层

目标：证明第二次 host-hit/load-back 后，输出质量不回退。

必须覆盖：

- `GSM8K`：
  - 第一次执行通过阈值
  - 第二次执行通过阈值
  - 若第二次低于第一次，下降幅度需限制在计划阈值内
- `MMLU`：
  - 第一次执行通过阈值
  - 第二次执行通过阈值
  - 若第二次低于第一次，下降幅度需限制在计划阈值内

实现建议：

- 参考 `AccuracyTwoPassMixin` 的结构，但第一阶段不要在两次执行之间 `flush_cache`。
- 更合适的抽象是新增一个 `HostHitTwoPassMixin` 或等价 helper，专门验证“第二次执行命中 host tree 并 load back”。

### 4. 模型覆盖矩阵

第一阶段至少覆盖以下两条模型线：

- `FULL attention`
  - 参考现有 `Qwen3-32B` unified tree 用例
  - 覆盖 `GSM8K`、`MMLU`、多轮 KL / cache-hit
- `SWA / DSV4`
  - 参考现有 `DeepSeek-V4-Flash-FP8` unified hicache 用例
  - 必须覆盖 `SWA` / compressed attention 相关启动参数
  - 覆盖两次 `GSM8K`、两次 `MMLU`、多轮 cache-hit

不满足上述两条模型线时，第一阶段不算测试闭环完成。

### 5. 手工验证启动模板

交付文档中应保留至少一个 `DSV4 SWA` 的手工启动模板，供开发阶段复现。

用户给定的手工启动方式如下，可直接作为 manual validation profile：

```bash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_DSV4_FP4_EXPERTS=0

nohup sglang serve \
  --trust-remote-code \
  --model-path /home/t4/models/deepseek-v4-flash-fp8/sgl-project/DeepSeek-V4-Flash-FP8 \
  --tp 4 \
  --mem-fraction-static 0.9 \
  --max-running-requests 64 \
  --host 0.0.0.0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-write-policy write_through \
  --port 30000 \
  > sglang.out &
```

自动化测试实现时，若要稳定覆盖 `DSV4 SWA` 路径，允许在 registered 用例中补充当前仓库已有的 `DSV4` 关键参数，例如：

- `--attention-backend compressed`
- `--page-size 256`
- `--chunked-prefill-size 8192`
- `--swa-full-tokens-ratio 0.25`
- `--disable-shared-experts-fusion`

## 依赖与顺序

### 里程碑 1：模式与语义建模

1. 在 `CacheInitParams` 和 `UnifiedRadixCache` 中正式引入 `L1ResidencyMode`。
2. 明确第一阶段用户开关和构造路径。
3. 明确字段语义：
   - `req.last_node` 只表示 L1/device anchor
   - `req.best_match_node` 只表示 L2 prefix anchor
   - `req.prefix_indices` 只表示 device prefix
4. 先补最小单测，锁住上述契约。

### 里程碑 2：scheduler 兼容 `None` device anchor

1. 调整 `match_prefix_for_req()` 周边调用约束，允许 `req.last_node=None` 且 `best_match_node!=None`。
2. 调整 `schedule_policy._lock_node()` 和 `add_one_req()`，保证 host 命中请求能在 admission 前安全 `load_back`。
3. 确认 `num_matched_prefix_tokens`、prefill 预算计算、`needs_host_load_back()` 等逻辑在新模式下不失真。
4. 补单测覆盖“host hit but no device anchor”的调度路径。

### 里程碑 3：request-owned publish 与 deferred release

1. 在 `UnifiedRadixCache` 内新增 request-owned publish 生命周期：
   - queue publish
   - 追踪 inflight ack
   - ack 后插树 / 去重 / 最终释放
2. 把 finish-path 从“立即 `release_kv_cache()`”改成“必要时 defer release”。
3. 把后台推进接到统一轮询点。
4. 补单测覆盖：
   - inflight 不提前 free
   - ack 后才真正释放
   - publish 成功后新请求可命中 host tree

### 里程碑 4：测试骨架建设

1. 基于 unified radix tree 现有 e2e 骨架新增 `l2-only` 专用 mixin 或 helper。
2. 新 helper 需要支持：
   - `GSM8K` 连续两次执行且第二次不 flush
   - `MMLU` 连续两次执行且第二次不 flush
   - 第二次执行的 host-hit/load-back 断言
3. 如现有 `cached_tokens` 断言不够表达需求，允许补充更细粒度的 host cache report。

### 里程碑 5：模型矩阵与最终验证

1. 新增或改造 `Qwen3-32B full attention` 的 `l2-only` 用例。
2. 新增或改造 `DeepSeek-V4-Flash-FP8` 的 `l2-only` 用例，确保覆盖 `SWA` 路径。
3. 断言：
   - 第一轮 cold / warmup
   - 第二轮 host hit + load back
   - server 存活
   - `GSM8K` / `MMLU` 精度不异常回退
   - KL / cached tokens 断言通过

## 建议修改文件

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/mem_cache/cache_init_params.py`
- `python/sglang/srt/mem_cache/registry.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/test/kits/unified_radix_cache_kit.py`
- `test/registered/unit/server_args/test_server_args.py`
- `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`
- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full_l2_only.py`
- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4_l2_only.py`

`scheduler.py`、`batch_result_processor.py`、`streaming_session.py` 是否需要跟进，以里程碑 3 实现时的真实挂点为准；如果只是为第一阶段引入 publish ack 轮询或 finish deferred release，也允许最小化触达这些文件。

## 实现说明

- 代码中不要出现 “第一阶段 / 第二阶段 / iter1 / phase1” 这类计划术语，保留面向长期维护的命名。
- 新模式应当被表达为稳定语义，而不是四处分散的 `if l2_only_mode` 特判。
- 如果某个旧路径只在 `indexed_device` 语义下成立，应尽早显式断言，而不是在 `request_owned_device` 下默默走错逻辑。
- 第一阶段完成后，应能自然支撑第二阶段 `PD` 接入：第二阶段只新增 “Prefill 直写 host landing pages 并 publish 到相同 L2 tree” 这条链路，而不是重写 `UnifiedTree` 的本体语义。
