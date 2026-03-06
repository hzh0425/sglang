# Hybrid RadixTree 完整方案（先接口，后实现）

## 0. 目标与约束

### 目标
- 用一套统一主流程支持多 attention component（如 `full + mamba`、`full + swa`）。
- 主流程不出现大量 `if mamba / if swa` 分支；差异下沉到 component hook。
- 先定义稳定接口与流程，再进入代码实现。

### 树约束（必须满足）
- **叶子节点**：必须包含所有 attention type 的 value。
- **非叶子节点**：要么包含所有 attention type value；要么仅包含 primary（通常是 full）value。
- 对于 `full + mamba + swa`，非叶出现 `mamba+swa`（没有 full）没有意义，不允许。

### 优先级规则（Evict）
- 每个 component 有节点优先级函数 `priority(is_leaf)`。
- 一旦某 component 在某 node 被 evict，则该 node 上所有 `priority <= trigger_priority` 的 component 必须联动 evict。
- 示例（与你给的一致）：
  - 叶子：`full = mamba = swa`
  - 非叶：`full > mamba = swa`

---

## 1. 数据模型

## 1.1 TreeNode
- `key`
- `parent / children`
- `component_data: Dict[component_name, ComponentData]`
- `last_access_time`

## 1.2 ComponentData
- `value: Optional[Tensor]`（`None` 即 tombstone）
- `lock_ref: int`

> 结论：统一用 `value is None` 表示“该 component 在此 node 已释放”，不再引入额外 tombstone flag。

## 1.3 CacheComponent（核心抽象）
- 每个 component 自带：
  - `name`
  - `is_primary`
  - `lru_list`
  - `evictable_size / protected_size`
- 每个 component 只关心自己的 value、锁、LRU、约束与释放策略。

---

## 2. Hook 接口定义（精简版）

> 原则：接口尽量少，主流程尽量薄。

## 2.1 Match Hooks

```python
init_match_state(params, root) -> dict
on_match_visit_node(node, matched_seg_len, state) -> bool
on_match_touch_lru(best_node, root) -> None
on_match_finalize(params, best_node, state) -> None  # optional
```

语义：
- `on_match_visit_node`：进入节点后更新状态并返回该 component 是否满足命中。
- 主流程用 `all(components_ok)` 判断是否更新 `best_node`。

## 2.2 Insert Hooks（命名规范版）

```python
on_insert_visit_node(node, matched_seg_len, total_prefix_len, params, state) -> None
on_insert_split_node(new_node, old_child, split_len, params, state) -> None
on_insert_handle_overlap(node, prefix_len, total_prefix_len, update_after_len, value_slice, params, state) -> bool
on_insert_finalize_leaf(leaf_node, remain_key, remain_primary_value, params, state) -> ComponentInsertResult
```

说明：
- `on_insert_handle_overlap` 返回 `True` 表示“该 overlap 已由组件处理”，主流程不再走默认 overlap free。
- 这是承载 SWA 复杂 overlap 逻辑的关键接口。

## 2.3 Evict Hooks

```python
select_evict_candidate() -> Optional[TreeNode]
get_evict_priority(is_leaf: bool) -> int
evict_node_value(node) -> int
on_evict_remove_from_lru(node) -> None
can_skip_free_if_primary_evicted() -> bool
```

说明：
- `select_evict_candidate`：primary 选 leaf，secondary 可选 any node（按各自 LRU）。
- `can_skip_free_if_primary_evicted`：如 SWA 在 primary 已 free 时可只清 value 不重复 free。

## 2.4 Request/Lifecycle Hooks（合并版）

你提的点是对的：`finished / unfinished` 可合并。

```python
prepare_request_insert(req, phase, token_ids, kv_indices, params) -> PreparedInsert
on_request_insert_done(req, phase, insert_result) -> None
```

- `phase in {"finished", "unfinished"}`
- 用单一 hook 统一预处理，避免两套接口。

## 2.5 Lock Hook

```python
get_lock_range(node, root) -> List[TreeNode]
```

- full: `node -> root`
- mamba: `[node] if has_value else []`
- swa: `node` 向上累计直到 `window_size`

---

## 3. 主流程设计

## 3.1 match_prefix 主流程

1. 初始化 `node=root, best_node=root, best_len=0`。
2. 初始化所有 component `match_state`。
3. 沿 key 下探，必要时 split。
4. 每访问一个节点，调用所有 component 的 `on_match_visit_node`。
5. 若全部为 `True`，更新 `best_node/best_len`。
6. 结束后调用 `on_match_touch_lru(best_node)`。
7. 调 `on_match_finalize`（例如 mamba COW）。
8. 返回 `(primary_values[:best_len], best_node)`。

可覆盖性：
- Mamba：`on_match_visit_node = value exists`
- SWA：在 state 内维护 `distance_since_released`，并做 window 判定。

## 3.2 insert 主流程

1. 准备 `key/value`（含 page 对齐）。
2. 沿树匹配 child，必要时 split（先通用 split，再 component `on_insert_split_node`）。
3. overlap 阶段：
   - 先让组件尝试 `on_insert_handle_overlap`
   - 若都未处理，走默认 `primary overlap free`
4. 每步调用 `on_insert_visit_node`（更新各 component LRU/状态）。
5. 循环结束：
   - 若有剩余 key，创建新叶并写入 primary value
   - 调用各 component `on_insert_finalize_leaf`
6. 汇总 `InsertResult`（含例如 `mamba_exist`）。

可覆盖性：
- SWA 当前复杂分支（你贴的 950-993）应全部迁入 `on_insert_handle_overlap`。
- Mamba 的“不可分裂 state + 叶子插入/复用”由 `on_insert_split_node + on_insert_finalize_leaf` 覆盖。

## 3.3 evict 主流程

1. 按触发 component 的 LRU 选 candidate：
   - primary: leaf only
   - secondary: any node
2. 计算 `trigger_priority = trigger.get_evict_priority(is_leaf)`。
3. 筛选 victims：当前 node 中 `priority <= trigger_priority` 的 component。
4. 依次 evict victims 的 value 并从对应 LRU 移除。
5. 若叶子被清空：删除叶子并向上 cascade prune。
6. 循环直到满足 `need_to_evict` 或无候选。

cascade prune 条件：
- 节点无子节点
- 节点无锁
- 非 primary component value 全为 `None`

## 3.4 cache_finished_req / cache_unfinished_req（统一框架）

两者共享同一骨架，只是 phase 不同：

1. 构造 `(token_ids, kv_indices)`。
2. 调 `prepare_request_insert(req, phase, ...)` 做组件预处理（裁剪、buffer、对齐）。
3. `insert(...)`（finished 在 `is_insert=True` 场景下；unfinished 总是插入中间结果）。
4. `match_prefix(...)` 回写 prefix（unfinished 需要）。
5. 锁迁移：`dec_lock_ref(old, handle)` -> `inc_lock_ref(new)`。
6. `on_request_insert_done(req, phase, insert_result)` 做组件收尾。

---

## 4. 组件行为映射（保证覆盖 SWA/Mamba）

## 4.1 Full(primary)
- `get_lock_range`: node->root
- `on_match_visit_node`: 检查 value 存在
- `on_insert_handle_overlap`: 默认 `False`（走主流程默认 overlap）
- `evict_node_value`: allocator.free(full value)

## 4.2 Mamba
- `on_match_visit_node`: `mamba_value is not None`
- `on_match_finalize`: 处理 COW（若需要）
- `on_insert_split_node`: new split parent 的 mamba value 置空
- `on_insert_finalize_leaf`: 插入/复用 mamba state，维护 mamba LRU
- `get_lock_range`: 当前 node（若有 value）

## 4.3 SWA
- `on_match_visit_node`: 维护 `distance_since_released` + window 判定
- `on_insert_handle_overlap`: 承载复杂 tombstone overlap 逻辑（Branch1/2/3）
- `on_insert_finalize_leaf`: 新叶写入 swa value，维护 swa LRU
- `evict_node_value`: 仅 free_swa / 或在 primary 已释放时跳过重复 free
- `get_lock_range`: 向上累计窗口

---

## 5. 兼容策略

- 对外 API 维持现有：
  - `MatchPrefixParams / InsertParams / EvictParams`
  - `MatchResult / InsertResult / EvictResult`
- `EvictParams` 可兼容两种调用：
  - 旧：`num_tokens/swa_num_tokens/mamba_num`
  - 新：`component_requests={...}`
- req 上 lock 句柄统一：
  - `req.cache_lock_handle`（dict）
  - 不再并行维护多套字段（如 `swa_uuid_for_lock` + hybrid 自定义句柄）

---

## 6. 实施顺序（建议）

1. 先实现 `match` hooks + 主流程（最稳定）。
2. 再实现 `insert` 的 `on_insert_handle_overlap`（先打通 SWA 最复杂分支）。
3. 再实现 `evict` priority-cascade + prune。
4. 最后接 lifecycle（finished/unfinished）统一 hook。
5. 每阶段做对齐测试：
   - `Hybrid(full+mamba)` 对齐 `MambaRadixCache`
   - `Hybrid(full+swa)` 对齐 `SWARadixCache`

---

## 7. 结论

这套接口在“最少 hook 数量”下，能覆盖现有 SWA/Mamba 核心行为：
- match 的 window/存在性约束
- insert 的 split + overlap（含 SWA 复杂分支）
- evict 的 priority 级联 + 树形清理
- finished/unfinished 的统一请求生命周期

主流程保持稳定、简洁，组件差异全部下沉到 hook，满足 Hybrid Tree 组合扩展需求。

