PD Decode RadixCache + HiCache Prefetch Incremental Transfer Plan
Goal Description
在 PD decode side 同时开启 --disaggregation-decode-enable-radix-cache 与 HiCache 时，让 D 端在 prealloc/handshake 阶段提前判断并启动 L2/L3 prefix 恢复，向 P 端发送更大的 decode_prefix_len，使 P 端只传输剩余 delta KV。

目标行为是：D 端可本地兑现的 prefix 由 L1 device hit + L2 host hit + L3 storage hit 组成；P 端仍完整 prefill 请求以保证 attention context 正确，但 send_kv_chunk() 从 decode_prefix_len 开始发送。D 端本地恢复路径先把 L3 storage prefetch 到 L2 host；L3 到 L2 后，再把原有 L2 hit 和新 prefetch 到 L2 的 L3 统一 load_back 到 L1 device。TransferQueue 等待这条本地恢复路径和 PD RDMA delta transfer 都完成后，再把 req_to_token 拼接为完整 committed KV prefix 并进入 waiting queue。

这个计划优先采用 HiRadixCache 路线，而不是旧的 DecodeKVCacheOffloadManager.load_to_indices 路线。理由是 load_back() 会自然更新 radix tree 的 device 节点，后续请求可以直接 L1 命中；同时 _pre_alloc() 只需要负责 delta slots，职责更清楚。

改动原则：尽可能小、尽可能精简。只改 decode-side prefix 计算、承诺长度、load_back 等待与 commit 拼接这条必要路径；不要重构 PD 协议、不要泛化多 backend、不要顺手修无关问题。PD 传输只关注 Mooncake backend。Mooncake 现有 decode_prefix_len / start_send_idx / delta pages 语义如果能直接满足需求，就不改 Mooncake 传输层。

Acceptance Criteria
AC-1: D 端可计算并传递 page-aligned 的三级 decode_prefix_len

Positive Tests (expected to PASS):
构造 L1-only 命中，请求传给 P 端的 decode_prefix_len 等于现有 radix cache prefix，并保持当前 delta 传输行为。
构造 L1+L2 命中，请求传给 P 端的 decode_prefix_len 等于 L1_len + host_hit_length，P 端 start_send_idx 使用该值。
构造 L1+L2+L3 命中且 prefetch 成功注册，请求传给 P 端的 decode_prefix_len 等于三级 page-aligned 总和。
Negative Tests (expected to FAIL):
非 page-aligned decode_prefix_len 进入 send_metadata() 前应被断言或对齐修正。
L3 storage 命中但 prefetch_from_storage() 未注册到 ongoing_prefetch 时，不能把 L3 tokens 计入 decode_prefix_len。
AC-2: TransferQueue 正确等待 PD RDMA 与 HiCache load_back 两路完成

Positive Tests (expected to PASS):
PD RDMA 先完成、L3->L2->L1 本地恢复后完成时，请求直到本地恢复 done 后才进入 waiting queue。
L3->L2->L1 本地恢复先完成、PD RDMA 后完成时，请求直到 RDMA success 后才 commit。
L2-only 场景跳过 L3->L2，直接把 L2 host hit load_back 到 L1，并等待 L2->L1 与 PD RDMA 都完成。
L3 命中场景必须先完成 L3->L2，再把原有 L2 hit 和新进入 L2 的 L3 hit 统一 L2->L1；不能拆成两次独立 load_back。
L1-only 场景保持现有只等待 PD RDMA 的行为。
Negative Tests (expected to FAIL):
任一路未完成时调用 _commit_transfer_to_req()。
metadata_buffer 未 ready 时释放 metadata index 或移动请求。
AC-3: req_to_token 拼接和 committed 语义正确

Positive Tests (expected to PASS):
commit 后 req_to_token[0:L1] 为 match prefix device indices，[L1:L1+L2+L3] 为统一 load_back indices，[L1+L2+L3:fill_len] 为 RDMA delta indices。
req.cache_protected_len 覆盖已经由 radix/HiCache 保护的 L1 + unified load_back 部分。
req.kv_committed_len 在 transfer/loadback 完成前不超过已真实可用 KV；commit 后为 fill_len。
Negative Tests (expected to FAIL):
cache_unfinished_req() 在 transfer 未完成时可插入未到位 delta KV。
load_back indices 被写入错误 offset，导致第一步 decode attention 读错 KV。
AC-4: P 端增量发送保持兼容

Positive Tests (expected to PASS):
PrefillBootstrapQueue.pop_bootstrapped() 通过 pop_decode_prefix_len() 设置 start_send_idx，sender.init(num_pages) 按 delta pages 初始化。
send_kv_chunk() 对 chunked prefill 保持 page 对齐，最后一段发送 metadata/state。
Mooncake backend 复用现有 decode_prefix_len 语义完成 delta 发送；除非验证失败，不新增或重构 Mooncake 协议字段。
0-page full hit 仍走 Mooncake 既有 no-KV 完成路径。
Negative Tests (expected to FAIL):
P 端 sender.init() 使用 total pages，但实际只发送 delta pages，导致 transfer 永远不完成。
D 端发送的 dst pages 数量与 P 端 src delta pages 数量不一致。
为 NIXL、Fake、Mori、Ascend 等非目标 backend 增加适配或测试矩阵。
AC-5: 失败路径不泄漏资源

Positive Tests (expected to PASS):
prefetch skipped/threshold/rate-limit 时降级为不计入 L3，delta KV 增大但请求正确完成。
transfer failure、metadata corruption、load_back failure/abort 都释放 metadata buffer、req pool slot、delta KV slots、tree lock/ref、pending prefetch host memory。
admission break 路径无条件释放 _match_prefix_and_lock() 获取的 radix lock，包括 root hit。
Negative Tests (expected to FAIL):
root lock/ref 在 admission break 后累积。
abort 后 token_to_kv_pool_allocator available size 无法恢复到基线。
AC-6: 运行范围被显式约束

Positive Tests (expected to PASS):
Mooncake + full KV + HiRadixCache 路径启用。
未支持组合启用时 fail-fast，错误信息包含实际开启的特性与支持范围。
Negative Tests (expected to FAIL):
Hisparse、SWA tail prealloc、Mamba、spec-dec、staging 在未实现适配时静默进入三级 prefetch 路径。
非 Mooncake PD transfer backend 进入本功能路径。
AC-7: 真实 PD + Router 部署验证 prefix 增量传输与 warm-cache 性能收益

Positive Tests (expected to PASS):
使用 benchmark/hicache/run_pd_decode_hicache_prefetch_validation.sh 启动 Qwen3-32B 的 PD prefill、PD decode 和 mini-lb router。
脚本通过 python3 -m sglang.bench_serving --dataset-name generated-shared-prefix 对 router 端口连续跑两轮相同 workload。
第一轮用于填充 decode-side radix/HiCache；第二轮日志中出现非零 decode_prefix_len、start_send_idx、aux_nokv、PD Delta 或 HiCache L3 load_back/prefetch 证据。
第二轮相对第一轮达到明显性能提升：默认 median_ttft_ms 降低至少 20%，或 request_throughput 提升至少 10%。
Negative Tests (expected to FAIL):
第二轮没有任何 prefix 增量传输日志证据。
第二轮 TTFT/吞吐未达到阈值改善。
prefill/decode GPU 组重叠、GPU 数量不等于 TP、L3 backend 未设置、router 或任一 server 未 ready。
Path Boundaries
Upper Bound (Maximum Scope)
完整实现 Mooncake PD 下的 decode side HiRadixCache L1/L2/L3 incremental transfer，包括 storage hit query、prefetch registration verification、L3->L2 后 post-prefetch re-match、统一 L2->L1 load_back、two-path TransferQueue gating、resource cleanup、少量必要日志、单元测试、Mooncake PD + Router 集成验证，以及两轮 bench_serving 性能对比。

Lower Bound (Minimum Scope)
只支持 Mooncake + full KV + no spec/no staging/no Hisparse/no SWA/no Mamba 的 decode path。实现 L2-only 与 L3-prefetch 成功路径，prefetch 被跳过时降级为 Mooncake PD delta 传输；无法兑现已承诺 prefix 的场景允许 abort，但必须无资源泄漏。

Allowed Choices
Can use: HiRadixCache.match_prefix()、prefetch_from_storage()、check_prefetch_progress()、init_load_back()、ready_to_load_host_cache()、现有 decode_prefix_len 协议、现有 send_kv_chunk() 的 start_send_idx 机制、TP/CP all-reduce 同步工具。
Can use: 小而明确的 DecodeRequest 字段扩展，用于记录 L1/L2/L3 长度、prefetch/loadback 状态、loadback indices、lock release state。
Can use: Mooncake 已有 decode_prefix_len 传递与 sender delta 初始化逻辑；优先验证复用，不预设需要改 Mooncake backend。
Cannot use: 让 P 端独立 query storage 来推断 D 端 prefix；这会造成 P/D 查询时刻不一致。
Cannot use: 在未更新 radix tree device 节点的情况下把 storage/H2D 数据直接塞进预分配 slots 并标记为 cache hit。
Cannot use: 发送 decode_prefix_len 后再静默回退承诺长度。
Cannot use: 为了泛化而修改非 Mooncake backend、公共 PD 抽象、router 策略或无关调度逻辑。
Dependencies and Sequence
Milestones
Milestone 1: 明确当前实现状态与运行范围

Phase A: 在 decode.py 中确认 _match_prefix_and_lock()、pop_preallocated()、_pre_alloc()、pop_transferred() 的现有职责。
Phase B: 在 scheduler.py / cache init 处确认 decode side 开启 HiCache 时实际 tree cache 类型为 HiRadixCache。
Phase C: 添加 feature gate：只有 full KV + Mooncake + HiRadixCache + radix cache enabled 进入三级 prefetch；其他组合保持现状或 fail-fast。
Phase D: 先验证 Mooncake 现有 decode_prefix_len 路径是否已经能让 P 端按 delta pages 初始化和发送；若能满足，不改 PD backend 层。
Milestone 2: 扩展 DecodeRequest 状态与 prefix matching

Phase A: 给 DecodeRequest 增加 l1_prefix_len、l2_host_hit_length、l3_storage_hit_length、decode_prefix_len、best_match_node、last_host_node、prefetch_done、loadback_done、loadback_indices。
Phase B: 把 _match_prefix_and_lock() 改成返回结构化结果，而不是二元组；先保留 L1-only 行为作为 fallback。
Phase C: 对 HiRadixCache 路径使用 MatchResult.host_hit_length 和 best_match_node 计算 L2，并记录统一 L2->L1 load_back 所需 anchor。
Milestone 3: 增加 L3 storage hit query 与 prefetch registration

Phase A: 在 HiRadixCache 增加同步 query helper，复用 HiCacheController._storage_hit_query() 的 hash-chain 语义，返回 page-aligned storage hit tokens。
Phase B: 在 _match_prefix_and_lock() 中对 L2 之后的 token suffix 做 storage query，并用 CP/TP all-reduce MIN 对齐所有 rank 的 hit length。
Phase C: 调用 prefetch_from_storage() 后检查 req.rid in ongoing_prefetch；未注册则把 L3 降级为 0。
Phase D: 只把已注册且 page-aligned 的 L3 tokens 加入 decode_prefix_len。
Milestone 4: 调整 allocation 与 metadata

Phase A: _pre_alloc() 接收 decode_prefix_len，只分配 delta slots；L1 device indices 先写入 req_to_token。
Phase B: 修改 kv_committed_len 时机：prealloc 后最多 committed 到可用 prefix，transfer/loadback commit 后再设为 fill_len。
Phase C: send_metadata(..., decode_prefix_len=decode_prefix_len) 继续使用 Mooncake 现有协议；P 端保持 start_send_idx=decode_prefix_len。
Phase D: 修复 admission break 路径，锁一律用匹配时保存的 dec params 释放，而不是仅在 prefix_len > 0 时释放。
Milestone 5: TransferQueue 双路推进

Phase A: 在 pop_transferred() 中拆出 _process_hicache_local_restore(decode_req)。
Phase B: 若有 pending L3 prefetch，先 check_prefetch_progress(req.rid)，等待 L3 storage 数据进入 L2 host。
Phase C: L3->L2 完成后必须 post-prefetch re-match，以找到包含原有 L2 hit 和新 L3 host hit 的统一 host prefix anchor。
Phase D: 用 re-match 后的 best_match_node/host_hit_length 调 init_load_back()，一次性把 L2+L3 统一 load_back 到 L1，再 ready_to_load_host_cache() 提交 L2->L1。
Phase E: 增加 L2->L1 completion check；只有 loadback_done and pd_done 才 commit。
Phase F: commit 时写入 unified loadback indices，并更新 cache_protected_len、prefix_indices、kv_committed_len。
Milestone 6: 失败和清理路径

Phase A: 封装 _abort_decode_req_with_hicache_cleanup()，统一释放 pending L3 prefetch、unified loadback lock、radix lock、metadata、req pool、delta KV。
Phase B: 对 prefetch skipped 使用降级；对 prefetch 已承诺后失败、load_back 返回不足、metadata corruption、PD failure 使用 abort。
Phase C: 增加日志字段：rid、L1/L2/L3、decode_prefix_len、delta_len、prefetch registered/skipped、post-rematch host hit、loadback tokens。
Milestone 7: 测试与验证

Phase A: 添加单元测试覆盖 prefix length calculation、prefetch skipped downgrade、req_to_token 拼接、lock release。
Phase B: 添加 Mooncake PD 集成测试，验证 P 端 start_send_idx 与 sender pages 数一致。
Phase C: 手动或脚本化验证：pre-flush、post-flush L3 storage hit、full hit 0-page、small L3 threshold downgrade。
Phase D: 记录准确率与稳定性：无 abort cascade、无 allocator leak、无 metadata corruption。
Phase E: 运行 benchmark/hicache/run_pd_decode_hicache_prefetch_validation.sh，通过真实 PD + Router + 两轮 bench_serving 验证 prefix 增量传输与第二轮 warm-cache 性能收益。
Implementation Notes
Code should NOT contain plan terminology such as AC-1 or Milestone.
Prefer a small dataclass or NamedTuple for decode-side match state; avoid passing loosely related tuples.
Error messages should include rid, decode_prefix_len, expected unified loadback tokens, actual unified loadback tokens, and transfer backend.
Keep P 端和 Mooncake backend 改动 minimal。现有 decode_prefix_len 协议和 start_send_idx 已经表达了所需语义，重点在 D 端不要承诺不可兑现的 L3。
Do not change files outside the necessary decode-side flow unless a failing AC proves the change is required. In particular, avoid touching non-Mooncake transfer backends, router policy, unrelated scheduler paths, and broad cache abstractions.
The first implementation should bias toward correctness and fail-fast behavior; performance tuning such as async storage pre-query can be deferred.
The end-to-end validation script requires explicit runtime choices: set PREFILL_CUDA_VISIBLE_DEVICES, DECODE_CUDA_VISIBLE_DEVICES, and HICACHE_STORAGE_BACKEND before running. For local L3 validation, HICACHE_STORAGE_BACKEND=file is the simplest backend.
