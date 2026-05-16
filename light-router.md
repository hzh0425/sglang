# 轻量级 Router Sticky + Load-Based MVP 实施计划

## Goal Description

实现一套新的 lightweight router，用来支撑蓝图里的 Router + PD 改进方案。这个 router 不走 GMS，不做完整 gateway，只做本阶段需要的事情：

- 管理 short/long 两组 PD engine。
- 根据请求长度选择 prefill group 和 decode group。
- 在每个 stage/group 内执行 `sticky -> load_based(power_of_two)`。
- 转发请求到选中的 decode engine，并注入 selected prefill engine 的 bootstrap 信息。
- 暴露足够少但可自动验收的 debug headers 和 stats。
- 通过 router 端口跑 GSM8K 和 MMLU，确认 router 转发不会破坏模型精度。

本计划的核心要求是轻量。实现者不要写“大而全”的 router 框架，不要引入和本蓝图无关的模块。MVP 只围绕下面这张逻辑图展开：

```text
                    Router
        ┌──────────────┴──────────────┐
        │                             │
  based on prefill length      based on estimated sequence length
        │                             │
  ┌─────▼─────┐                 ┌─────▼─────┐
  │ long  P   │ ───────────────▶ │ long  D   │
  └───────────┘       ╲   ╱      └───────────┘
  ┌───────────┐       ╱   ╲      ┌───────────┐
  │ short P   │ ───────────────▶ │ short D   │
  └───────────┘                 └───────────┘
```

### Blueprint Mapping

| 蓝图能力 | MVP 实现 |
| --- | --- |
| Routing policy chain | 只实现 `sticky -> load_based(power_of_two)` |
| Balance constraints | sticky 候选必须通过 score gate，否则 fallback |
| PD length dispatch | prefill 按 `uncached_prefill_tokens`，decode 按 `estimated_sequence_length` |
| Different parallelism | router 只按配置管理 short/long pool，不推导 GPU 拓扑 |
| Multiple P/D groups | 支持 `short` 和 `long` 两个 group |
| Minimal package | 一个轻量 router binary/library，不接入完整 GMS |

### 实现落点

推荐新增一个很小的 Rust crate：

```text
rust/sglang-light-router/
```

MVP 文件数量要保持克制，建议只包含：

```text
rust/sglang-light-router/Cargo.toml
rust/sglang-light-router/src/main.rs
rust/sglang-light-router/src/router.rs
rust/sglang-light-router/src/policy.rs
rust/sglang-light-router/src/proxy.rs
rust/sglang-light-router/src/stats.rs
rust/sglang-light-router/tests/light_router_tests.rs
benchmark/router/run_two_pool_pd_router_validation.sh
```

不要为了“看起来完整”拆出 `manager/helper/utils/service/provider` 这类空泛模块。只有当一个文件已经明显承担两类职责时，才允许继续拆分。

### 当前阶段不做

本阶段不实现：

- cache-aware routing
- prefix hash routing
- radix tree
- L3 cache integration
- 真实 engine KV prefix query
- 完整 OpenAI gateway 能力
- conversation persistence
- MCP / Responses API
- external provider routing
- Prometheus exporter
- YAML/TOML 配置文件
- PyO3 / maturin / Python extension

蓝图里出现的 `simple prefix hashing` 先不写代码。可以在 policy chain parser 里保留 future extension 的错误提示，但不要实现一个没有验收目标的 prefix hash policy。

### 真实验收部署

真实验收使用 `Qwen/Qwen3-32B`，不经过 GMS：

```text
sglang-light-router
  -> short PD: 1 个 short prefill + 1 个 short decode，共 2 GPU
  -> long PD : 1 个 long prefill  + 1 个 long decode， 共 2 GPU
```

默认每个 prefill/decode role 使用 `TP=1`，整体验收是 1 个 router 进程 + 2 套 PD pool，共 4 张 GPU。如果目标机器上 Qwen32B 必须单 role `TP=2` 才能放下，可以用脚本环境变量覆盖 TP，但那不是 4 GPU MVP 验收形态，需要在结果里说明。

## Acceptance Criteria

- AC-1: Router 以轻量 binary 方式启动，并能配置 short/long PD pool。
  - Positive Tests (expected to PASS):
    - `cargo run --manifest-path rust/sglang-light-router/Cargo.toml --bin sglang-light-router -- --help` 能展示 router 参数。
    - 同时配置 short/long prefill/decode 后，router `/health` 返回成功。
    - 每个 prefill endpoint 可以配置 bootstrap port。
  - Negative Tests (expected to FAIL):
    - 缺少任意一个 group 的 prefill 或 decode 时，启动校验失败。
    - 出现非 `short`/`long` group 时，启动校验失败。
    - 验证脚本不能通过 `python3 -m sglang_router.launch_lightweight_router` 启动 router。

- AC-2: Prefill 和 decode 独立按长度选 group。
  - Positive Tests (expected to PASS):
    - `uncached_prefill_tokens < prefill_long_threshold` 选择 short prefill。
    - `uncached_prefill_tokens >= prefill_long_threshold` 选择 long prefill。
    - `estimated_sequence_length < decode_long_threshold` 选择 short decode。
    - `estimated_sequence_length >= decode_long_threshold` 选择 long decode。
    - MVP 中 `uncached_prefill_tokens = prompt_tokens`，`estimated_sequence_length = prompt_tokens + max_new_tokens`。
  - Negative Tests (expected to FAIL):
    - prefill 超阈值且 long prefill 健康时，不能进入 short prefill。
    - decode 超阈值且 long decode 健康时，不能进入 short decode。
    - prefill 的 group 选择不能强制绑定 decode 的 group 选择。

- AC-3: Sticky session 只在 stage/group 内生效。
  - Positive Tests (expected to PASS):
    - 同一 routing key、同一 role、同一 group 重复请求选择同一 worker。
    - 同一 routing key 在 prefill 和 decode 上可以有不同 sticky 记录。
    - 同一 routing key 在 short 和 long group 上可以有不同 sticky 记录。
    - 没有 routing key 时跳过 sticky，直接走 load-based selection。
  - Negative Tests (expected to FAIL):
    - short group 的 sticky 记录不能把 long 请求拉回 short group。
    - sticky 候选只是 propose 时不能修改 sticky table。

- AC-4: Sticky 可以被 load balance gate 打破。
  - Positive Tests (expected to PASS):
    - sticky worker score 在阈值内时继续 sticky。
    - sticky worker 同时超过绝对阈值和相对阈值时，fallback 到 load-based reference worker。
    - unhealthy sticky worker 被拒绝。
  - Negative Tests (expected to FAIL):
    - 只超过绝对阈值或只超过相对阈值时，不能误判为必须打破 sticky。
    - 被 balance gate 拒绝的 sticky 候选不能被 commit。

- AC-5: Load-based fallback 使用 power-of-two，并使用统一 score。
  - Positive Tests (expected to PASS):
    - 健康 worker 数量大于等于 2 时，随机 sample 2 个并选择 score 更低者。
    - 健康 worker 只有 1 个时选择唯一 worker。
    - score 使用 `work + lambda * u / (1 - u)`。
    - `u` 必须 clamp 到 `[0, 1 - eps]`，避免除零。
  - Negative Tests (expected to FAIL):
    - 不能混用一个 worker 的 rich token load 和另一个 worker 的 local request count 做比较。
    - 没有健康 candidate 时必须返回明确错误。

- AC-6: Router 只实现必要 forwarding 和 PD bootstrap 注入。
  - Positive Tests (expected to PASS):
    - 请求被转发到 selected decode engine。
    - 转发前注入 selected prefill engine 的 bootstrap host/port/room。
    - non-streaming 请求完成后 local in-flight load 回到 0。
    - streaming 或 client disconnect 后 local in-flight load 不泄漏。
  - Negative Tests (expected to FAIL):
    - Router 不能自己执行模型推理。
    - Router 不能把请求直接发给 prefill engine 当作最终响应。
    - upstream error 不能留下永久 in-flight counter。

- AC-7: Debug headers 和 stats 足够验证，但保持最小。
  - Positive Tests (expected to PASS):
    - 开启 `--enable-routing-debug-headers` 后，响应包含 selected group、worker 和 policy branch。
    - `/debug/routing_stats` 返回 route counts、policy counts 和 per-engine load snapshot。
    - stats 能看到最近一次 `/v1/loads?include=core` 是否成功。
  - Negative Tests (expected to FAIL):
    - 未开启 debug header 开关时，不能暴露 worker URL。
    - `/debug/routing_stats` 缺少 route counts 或 engine load snapshot 时，真实验收失败。

- AC-8: 真实 Qwen32B 双池 PD 验证通过。
  - Positive Tests (expected to PASS):
    - `benchmark/router/run_two_pool_pd_router_validation.sh` 启动 short prefill、short decode、long prefill、long decode 和 Rust lightweight router。
    - smoke test 验证 short/long 请求分别命中预期 prefill/decode group。
    - `bench_serving` 分别压 short workload 和 long workload。
    - 压测后 `/debug/routing_stats` 的 route counts、policy counts、engine dispatch counts 和 load snapshot 符合预期。
    - `benchmark/gsm8k/bench_sglang.py` 通过 router port 跑完，并达到配置的 `GSM8K_MIN_ACCURACY`。
    - `python3 -m sglang.test.run_eval --eval-name mmlu` 通过 router port 跑完，并达到配置的 `MMLU_MIN_SCORE`。
  - Negative Tests (expected to FAIL):
    - 任意 GPU id 重叠时，脚本必须在 server launch 前失败。
    - 任意 engine 未 ready 时，脚本必须失败并指出角色。
    - short/long workload 主要落入错误 group 时，脚本必须失败。
    - GSM8K/MMLU 直接绕过 router port 测 individual engine 时，本 AC 不算通过。
    - GSM8K/MMLU 分数低于配置阈值时，脚本必须失败。

- AC-9: 代码量和范围保持 lightweight。
  - Positive Tests (expected to PASS):
    - MVP 新增源文件只服务于 config、routing policy、proxy forwarding、stats 和测试。
    - `cargo test --manifest-path rust/sglang-light-router/Cargo.toml` 通过。
    - `bash -n benchmark/router/run_two_pool_pd_router_validation.sh` 通过。
  - Negative Tests (expected to FAIL):
    - 出现未被本计划 AC 覆盖的 provider、gateway、conversation、cache、prefix tree、metrics exporter 模块时，需要删除或移到后续计划。
    - 为了“未来可能用到”新增无调用路径代码时，本 AC 失败。

## Path Boundaries

### Lower Bound (Minimum Scope)

最低可接受实现只包含：

- 一个 `sglang-light-router` binary。
- short/long prefill/decode endpoint 配置。
- `/health`。
- `/v1/chat/completions` 或当前验证所需的最小 OpenAI-compatible endpoint。
- request feature extraction。
- prefill/decode group selection。
- sticky table。
- load-based power-of-two fallback。
- local in-flight load guard。
- `/v1/loads?include=core` polling。
- debug response headers。
- `/debug/routing_stats`。
- 单元测试、mock integration test 和真实双池 PD 验证脚本。
- 通过 router port 执行 GSM8K 和 MMLU 的精度验证。

### Upper Bound (Maximum Scope)

本阶段最多允许做到：

- 支持 non-streaming 和 streaming response 的 load guard。
- 支持多个 worker per group，但只要求 short/long 两个 group。
- 支持 policy chain parser，但只接受 `sticky` 和 `power_of_two`。
- 支持 test-only load override endpoint，用于 mock test 覆盖 sticky break；该 endpoint 必须默认关闭。

除此之外都算越界。

### Allowed Choices

- 可以参考 `sgl-model-gateway` 中的 worker、manual policy、power-of-two policy、load monitor、PD router 代码思路。
- 可以使用 `axum`、`tokio`、`reqwest`、`serde`、`clap`、`tracing`。
- 可以把简单类型放在同一个文件里，不需要为了“分层”拆文件。
- 可以用 local token count approximation，不要求接入真实 tokenizer。

### Not Allowed

- 不允许整体 fork `sgl-model-gateway`。
- 不允许把 MVP 写成一个通用 gateway。
- 不允许实现 cache-aware、prefix hash、radix tree、L3。
- 不允许新增 YAML/TOML config、Prometheus exporter、external provider、conversation storage。
- 不允许为了规避 Rust 生命周期而使用全局可变状态。
- 不允许在 `.await` 期间持有锁 guard。

## Feasibility Hints and Suggestions

### 可借鉴的现有代码

只借鉴小块行为，不做大规模复用：

```text
sgl-model-gateway/src/core/worker.rs
sgl-model-gateway/src/core/worker_registry.rs
sgl-model-gateway/src/policies/manual.rs
sgl-model-gateway/src/policies/power_of_two.rs
sgl-model-gateway/src/core/worker_manager.rs
sgl-model-gateway/src/routers/http/pd_router.rs
```

借鉴点：

- `WorkerLoadGuard` 的 RAII 思路。
- manual policy 的 sticky key 管理。
- power-of-two 中 rich load 缺失时降级到同一种 local metric 的规则。
- worker manager 轮询 `/v1/loads?include=core` 的方式。
- PD router 注入 bootstrap 信息的方式。

不要复用点：

- 不要复用完整 HTTP retry/gateway pipeline。
- 不要复用完整 provider/model registry。
- 不要把 SMG 的配置体系搬过来。

### Minimal Data Model

只需要这些核心类型：

```rust
enum Role { Prefill, Decode }
enum Group { Short, Long }

struct Engine {
    id: String,
    role: Role,
    group: Group,
    url: String,
    bootstrap_port: Option<u16>,
}

struct RoutingFeatures {
    routing_key: Option<String>,
    prompt_tokens: usize,
    uncached_prefill_tokens: usize,
    max_new_tokens: usize,
    estimated_sequence_length: usize,
}

struct LoadSnapshot {
    in_flight_requests: usize,
    work: usize,
    token_usage: Option<f64>,
    last_load_poll_ok: bool,
}
```

不要提前设计复杂的 `Manager`、`Provider`、`Backend` 抽象。

### Routing Algorithm

MVP 路由流程：

```python
def route(request):
    features = extract_features(request)

    prefill_group = select_prefill_group(features)
    decode_group = select_decode_group(features)

    prefill = select_engine(role="prefill", group=prefill_group, features=features)
    decode = select_engine(role="decode", group=decode_group, features=features)

    inject_bootstrap(request, prefill)
    return proxy_to_decode(request, decode)
```

Engine selection：

```python
def select_engine(role, group, features):
    healthy = healthy_engines(role, group)
    reference = power_of_two(healthy)

    sticky = sticky_table.propose(features.routing_key, role, group)
    if sticky and acceptable(sticky, reference):
        selected = sticky
        branch = "sticky_hit"
    else:
        selected = reference
        branch = "load_based_fallback"

    sticky_table.commit(features.routing_key, role, group, selected)
    stats.record(role, group, selected, branch)
    return selected
```

Sticky scope 必须是：

```text
(routing_key, role, group) -> engine_id
```

### Load Score

统一 score：

```text
score(engine) = work + lambda * u / (1 - u)
```

语义：

- `u` 是 token usage。
- prefill 的 `work` 是 local in-flight prefill tokens。
- decode 的 `work` 优先是 engine reported decode batch size；缺失时用 local in-flight decode requests。
- token usage 小时主要按 request/work 均衡。
- token usage 高时快速避开接近容量上限的 engine。

Balance gate：

```text
abs_bad = preferred_score - reference_score > abs_threshold
rel_bad = preferred_score > reference_score * relative_upper_bound_limit
acceptable = !(abs_bad && rel_bad)
```

### Stats Contract

`/debug/routing_stats` 只需要返回自动验收需要的字段：

```json
{
  "route_counts": {
    "prefill=short,decode=short": 128,
    "prefill=long,decode=long": 128
  },
  "policy_counts": {
    "prefill": {
      "sticky_hit": 1,
      "load_based_fallback": 127
    },
    "decode": {
      "sticky_hit": 1,
      "load_based_fallback": 127
    }
  },
  "engines": [
    {
      "id": "short-prefill-0",
      "role": "prefill",
      "group": "short",
      "url": "http://127.0.0.1:30000",
      "local_dispatch_total": 128,
      "in_flight_requests": 0,
      "reported_total_tokens": 1024,
      "reported_token_usage": 0.01,
      "last_load_poll_ok": true
    }
  ]
}
```

不要为 stats 先做 Prometheus、dashboard、历史窗口或复杂聚合。

### Qwen32B 验证命令

真实验证由脚本负责：

```bash
SHORT_PREFILL_CUDA_VISIBLE_DEVICES=<GPU0> \
SHORT_DECODE_CUDA_VISIBLE_DEVICES=<GPU1> \
LONG_PREFILL_CUDA_VISIBLE_DEVICES=<GPU2> \
LONG_DECODE_CUDA_VISIBLE_DEVICES=<GPU3> \
bash benchmark/router/run_two_pool_pd_router_validation.sh
```

脚本默认：

```text
MODEL_PATH=Qwen/Qwen3-32B

short prefill: 30000
short decode : 31000
long prefill : 30001
long decode  : 31001
router       : 20000
```

验证脚本里的 threshold 可以用 `512`，方便 short/long benchmark 稳定命中不同 group。生产默认可以单独设成 `32768`。

### Accuracy Eval Contract

GSM8K 和 MMLU 必须通过 router port 发请求，不允许直接打到 prefill 或 decode engine。按照 `sglang-deploy-test` 的规则，accuracy eval 是前台串行命令，每个 eval 写独立日志。

GSM8K：

```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --port <ROUTER_PORT> \
  --num-questions 500 \
  --num-shots 24 \
  --parallel 50 \
  > gsm8k.out 2>&1
```

MMLU：

```bash
python3 -m sglang.test.run_eval \
  --eval-name mmlu \
  --port <ROUTER_PORT> \
  --num-examples 200 \
  --max-tokens 4096 \
  --repeat 4 \
  > mmlu.out 2>&1
```

验证脚本需要解析：

- GSM8K 日志中的 `Accuracy: <value>`。
- MMLU 日志中的 `Score: <value>` 或 `mean: <value>`。

默认阈值保持保守，避免把 router correctness 和模型 leaderboard 绑定得太死：

```text
GSM8K_MIN_ACCURACY=0.50
MMLU_MIN_SCORE=0.50
```

如果目标环境已有 Qwen32B baseline，可以用环境变量提高阈值。这里的目标不是证明模型 SOTA，而是证明通过 lightweight router 转发后，输出语义和精度没有明显损坏。

## Dependencies and Sequence

### Milestones

1. Milestone 1: Skeleton and CLI
   - 新增 `rust/sglang-light-router/`。
   - 实现 `--help`、endpoint 参数、threshold 参数、balance 参数。
   - 实现配置校验和 `/health`。
   - 对应 AC-1。

2. Milestone 2: Feature Extraction and Group Dispatch
   - 提取 routing key。
   - 估算 prompt tokens 和 max output tokens。
   - 实现 prefill/decode group selection。
   - 对应 AC-2。

3. Milestone 3: Sticky and Load-Based Policy
   - 实现 sticky table 的 `propose -> acceptable -> commit`。
   - 实现 score、balance gate、power-of-two。
   - 实现 local load guard。
   - 对应 AC-3、AC-4、AC-5。

4. Milestone 4: PD Forwarding
   - 选择 prefill engine 和 decode engine。
   - 注入 bootstrap host/port/room。
   - 转发到 decode engine。
   - 覆盖普通响应和 streaming/drop load 生命周期。
   - 对应 AC-6。

5. Milestone 5: Observability for Validation
   - 添加 debug headers。
   - 添加 `/debug/routing_stats`。
   - 添加 `/v1/loads?include=core` polling。
   - 对应 AC-7。

6. Milestone 6: Tests and Real Validation Script
   - 添加单元测试和 mock integration test。
   - 更新 `benchmark/router/run_two_pool_pd_router_validation.sh` 启动 Rust router。
   - 运行 Qwen32B 双池 PD smoke test、bench_serving、GSM8K 和 MMLU。
   - 对应 AC-8、AC-9。

## Task Breakdown

- T-1 [analyze]: 收窄实现边界。
  - 确认只做 short/long PD、sticky、load-based、stats 和验证脚本。
  - 检查现有 `sgl-model-gateway` 代码，只记录可借鉴的小块逻辑。
  - 不做 cache-aware、prefix hash、full gateway。

- T-2 [coding]: 新增 minimal Rust crate。
  - 添加 `Cargo.toml`、`main.rs`、`router.rs`、`policy.rs`、`proxy.rs`、`stats.rs`。
  - 实现 CLI parser 和 `/health`。
  - 保持代码文件数量克制，避免空泛模块。

- T-3 [coding]: 实现配置和 feature extraction。
  - 解析 `--prefill short=http://host:port,bootstrap=8998`。
  - 解析 `--decode short=http://host:port`。
  - 从请求 body/header 估算 routing features。
  - 添加相关 unit tests。

- T-4 [coding]: 实现 group dispatch。
  - prefill 使用 `uncached_prefill_tokens`。
  - decode 使用 `estimated_sequence_length`。
  - 写 route counter key：`prefill=<group>,decode=<group>`。

- T-5 [coding]: 实现 sticky + load-based selection。
  - sticky scope 使用 `(routing_key, role, group)`。
  - sticky 只 propose，不提前 commit。
  - power-of-two 选 reference。
  - balance gate 决定是否打破 sticky。
  - 添加 score 和 selector tests。

- T-6 [coding]: 实现 PD forwarding。
  - 注入 selected prefill bootstrap 信息。
  - 请求转发到 selected decode engine。
  - `LoadGuard` 覆盖普通响应、stream drop、client disconnect 和 upstream error。

- T-7 [coding]: 实现最小 observability。
  - debug headers 受 `--enable-routing-debug-headers` 控制。
  - `/debug/routing_stats` 返回 route counts、policy counts、engine stats。
  - load polling 只解析当前需要的字段，先支持 `aggregate.total_tokens`。

- T-8 [coding]: 完成验证脚本。
  - 脚本启动 short/long 两套 PD 和 Rust router。
  - 脚本执行 smoke test、short bench、long bench、stats assert。
  - 脚本通过 router port 串行执行 GSM8K 和 MMLU。
  - 脚本解析 accuracy/score，并低于阈值时失败。
  - 脚本先检查 GPU id 重叠和 endpoint readiness。
  - `bash -n` 必须通过。

- T-9 [analyze]: 收尾检查。
  - 运行 `cargo fmt`。
  - 运行 `cargo test --manifest-path rust/sglang-light-router/Cargo.toml`。
  - 运行 `bash -n benchmark/router/run_two_pool_pd_router_validation.sh`。
  - 检查新增代码是否都被 AC 覆盖；没有覆盖就删掉或移出本计划。

## Implementation Notes

- 代码里不要写 `AC-1` 这类计划术语。
- 不要把 sticky/load policy 和 HTTP forwarding 写死耦合；但也不要为此设计复杂 trait 层级。
- 不要新增 `Manager`、`Handler`、`Helper`、`Utils` 这类泛名模块。
- Debug headers 必须显式开启。
- Routing key header 优先级：`X-SMG-Routing-Key` 优先，其次 `X-SGLang-Routing-Key`。
- 如果两个 worker 只有一个有 rich load，两个都降级到 local counter 比较。
- 所有 local in-flight counters 必须通过 RAII guard 管理。
- Rust handler 不要在 `.await` 期间持有锁。
- 真实验证前先跑 mock integration test；真实验证不能用 mock 替代。
- GSM8K/MMLU 必须打 router port，不能直接打 individual engine。
- Accuracy eval 必须前台串行执行，日志分别写到 `gsm8k.out` 和 `mmlu.out`。
- 本计划完成的标志不是“代码很多”，而是 Qwen32B 双池 PD 验证脚本能证明路由行为正确。

