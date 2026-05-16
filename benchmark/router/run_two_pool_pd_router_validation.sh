

真实验收部署使用 `Qwen/Qwen3-32B`，不经过 GMS，只启动最新的 lightweight router：

```text
lightweight router
  -> short PD：1 个 short prefill + 1 个 short decode，共 2 GPU
  -> long PD：1 个 long prefill + 1 个 long decode，共 2 GPU
```

也就是整体验收形态是 1 个 router 进程 + 2 套 PD pool，共 4 张 GPU。这里默认每个 prefill/decode role 使用 `TP=1`。如果某台机器上 `Qwen/Qwen3-32B` 需要单 role `TP=2` 才能放下，脚本允许覆盖 TP，但那会变成每套 PD 4 GPU、两套共 8 GPU，不再是本阶段的 4 GPU 验收形态。

Router 按请求特征先选池，再在池内选 engine：

另外需要增加一个调试统计接口，供 `bench_serving` 后做自动验收：

```text
GET /debug/routing_stats
```

建议返回结构：

```json
{
  "route_counts": {
    "prefill=short,decode=short": 128,
    "prefill=long,decode=long": 128
  },
  "policy_counts": {
    "prefill": {
      "sticky_hit": 1,
      "load_based_fallback": 255
    },
    "decode": {
      "sticky_hit": 1,
      "load_based_fallback": 255
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

`bench_serving` 默认不会保留每个响应的 routing debug header，所以批量压测后的正确性不要靠人工读日志，而是靠这个接口断言：

- short benchmark 的增量主要落在 `prefill=short,decode=short`。
- long benchmark 的增量主要落在 `prefill=long,decode=long`。
- `sticky_hit` 和 `load_based_fallback` 都至少出现过。
- 每个 engine 都有 dispatch 计数和最近一次 workload snapshot。

## Qwen32B 双池 PD 启动方式

本节是写完 router 代码后的自验证部署方式，不包括 GMS。启动顺序必须是：

```text
short prefill -> short decode -> long prefill -> long decode -> lightweight router -> smoke test -> bench_serving
```

推荐直接使用脚本：

```bash
SHORT_PREFILL_CUDA_VISIBLE_DEVICES=0 \
SHORT_DECODE_CUDA_VISIBLE_DEVICES=1 \
LONG_PREFILL_CUDA_VISIBLE_DEVICES=2 \
LONG_DECODE_CUDA_VISIBLE_DEVICES=3 \
bash benchmark/router/run_two_pool_pd_router_validation.sh
```

上面的 GPU id 只是示例。实际运行前必须先用 `nvidia-smi` 确认这 4 张卡空闲。

脚本默认模型：

```text
MODEL_PATH=Qwen/Qwen3-32B
MODEL_NAME=Qwen/Qwen3-32B
```

脚本默认端口：

```text
short prefill: 30000
short decode : 31000
long prefill : 30001
long decode  : 31001
router       : 20000
```

单套 PD 的启动命令形态如下。short 和 long 的区别只有端口、bootstrap port、dist init addr 和 GPU 绑定不同。

Prefill：

```bash
CUDA_VISIBLE_DEVICES=<PREFILL_GPU> python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --trust-remote-code \
  --page-size 64 \
  --port <PREFILL_PORT> \
  --tp 1 \
  --mem-fraction-static 0.85 \
  --prefill-round-robin-balance \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port <BOOTSTRAP_PORT> \
  --disaggregation-ib-device <RDMA_DEVICES> \
  --dist-init-addr <DIST_INIT_ADDR> \
  --nnodes 1 \
  --node-rank 0
```

Decode：

```bash
CUDA_VISIBLE_DEVICES=<DECODE_GPU> python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --trust-remote-code \
  --page-size 64 \
  --port <DECODE_PORT> \
  --tp 1 \
  --mem-fraction-static 0.85 \
  --disaggregation-mode decode \
  --load-balance-method round_robin \
  --prefill-round-robin-balance \
  --disaggregation-bootstrap-port <BOOTSTRAP_PORT> \
  --disaggregation-ib-device <RDMA_DEVICES> \
  --dist-init-addr <DIST_INIT_ADDR> \
  --nnodes 1 \
  --node-rank 0
```

Router：

```bash
python3 -m sglang_router.launch_lightweight_router \
  --pd-disaggregation \
  --prefill short=http://127.0.0.1:30000:8998 \
  --decode short=http://127.0.0.1:31000 \
  --prefill long=http://127.0.0.1:30001:8999 \
  --decode long=http://127.0.0.1:31001 \
  --prefill-routing-policy-chain sticky,power_of_two \
  --decode-routing-policy-chain sticky,power_of_two \
  --prefill-long-threshold 512 \
  --decode-long-threshold 512 \
  --balance-abs-threshold 32 \
  --balance-relative-upper-bound-limit 3.0 \
  --load-score-token-usage-weight 1.0 \
  --enable-routing-debug-headers \
  --port 20000
```

这里的 threshold 故意设成 `512`，是为了让真实验证里用较短的 random benchmark 也能稳定打到 long pool。生产默认值可以再调回 `32768` 或由部署配置决定。

写完 router 代码后，开发者必须自行完成以下验证：

```bash
SHORT_PREFILL_CUDA_VISIBLE_DEVICES=<GPU0> \
SHORT_DECODE_CUDA_VISIBLE_DEVICES=<GPU1> \
LONG_PREFILL_CUDA_VISIBLE_DEVICES=<GPU2> \
LONG_DECODE_CUDA_VISIBLE_DEVICES=<GPU3> \
bash benchmark/router/run_two_pool_pd_router_validation.sh
```

验证脚本会做三类检查：

- smoke test：通过 `/v1/chat/completions` 检查 short/long group dispatch 和 sticky repeat。
- benchmark test：用 `python3 -m sglang.bench_serving` 分别发短负载和长负载。
- stats test：读取 `/debug/routing_stats`，检查 policy chain、group dispatch 和 engine workload 采集是否正确。

## 验收标准

- AC-9：真实 Qwen32B 双池部署可以启动。
  - 正向测试：`Qwen/Qwen3-32B` 下启动 lightweight router + short PD + long PD。
  - 正向测试：short PD 占 2 GPU，long PD 占 2 GPU，总共 4 GPU。
  - 反向测试：任意 GPU id 重叠时，验证脚本必须在 server launch 前失败。

- AC-10：`bench_serving` 可以验证长短负载分发。
  - 正向测试：短负载 benchmark 后，`/debug/routing_stats` 中 `prefill=short,decode=short` 计数增加。
  - 正向测试：长负载 benchmark 后，`/debug/routing_stats` 中 `prefill=long,decode=long` 计数增加。
  - 反向测试：长负载在 long pool 健康时不能主要落到 short pool。

- AC-11：policy chain 和 workload collection 可观测。
  - 正向测试：同一 sticky key 的第二次请求出现 `sticky_hit`。
  - 正向测试：无 sticky key 或 sticky 被拒绝时出现 `load_based_fallback`。
  - 正向测试：每个 engine 都能看到 dispatch 计数、本地 in-flight 计数和最近一次 `/v1/loads?include=core` 轮询结果。
  - 反向测试：缺少 `/debug/routing_stats` 时，真实验证脚本必须失败。

## 路径边界
- 通过 `/v1/chat/completions` 做基础正确性 smoke test。
- 通过 `bench_serving` 分别压 short workload 和 long workload。
- 通过 `/debug/routing_stats` 验证 route counts、policy counts 和 engine workload snapshots。

- 检查 debug routing headers
- 运行 short `bench_serving`
- 运行 long `bench_serving`
- 检查 `/debug/routing_stats`


- `MODEL_PATH`
- 各角色 GPU ids

这些需要在真实机器上替换或通过环境变量导入。
`MODEL_PATH` 默认是 `Qwen/Qwen3-32B`。GPU ids 需要在真实机器上替换或通过环境变量导入。

- 优先兼容现有 `X-SMG-Routing-Key`。

