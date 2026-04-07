# Qwen3.5 TBO (Two Batch Overlap) Support Plan

## Status: COMPLETE (test passed, gsm8k score 0.97)

## 1. Architecture Overview

Qwen3.5 397B-A17B (model_type: `qwen3_5_moe_text`):
- 60 layers, pattern: [linear, linear, linear, full] x 15
- All layers are MoE (`mlp_only_layers: []`)
- 512 experts, top-10, `shared_expert_intermediate_size: 1024`
- Two decoder layer types:
  - `Qwen3_5LinearDecoderLayer`: GatedDeltaNet (linear attention) + `Qwen2MoeSparseMoeBlock`
  - `Qwen3_5AttentionDecoderLayer`: Full Attention (with output gate) + `Qwen2MoeSparseMoeBlock`
- MoE block: `Qwen2MoeSparseMoeBlock` (from `qwen2_moe.py`, NOT `Qwen3MoeSparseMoeBlock`)

## 2. Key Differences from Qwen3-Next TBO

| Aspect | Qwen3-Next | Qwen3.5 |
|--------|-----------|---------|
| GDN fused kernel | `fused_qkvzba_split_reshape_cat` | `fused_qkvzba_split_reshape_cat_contiguous` |
| GDN DP-Attn padding | No | Yes (`core_attn_out.shape != z.shape`) |
| MoE block class | `Qwen2MoeSparseMoeBlock` | `Qwen2MoeSparseMoeBlock` (same) |
| Shared experts | Yes | Yes |
| `is_layer_sparse` | `self.is_layer_sparse = True` | Local variable (need fix) |
| LayerNorm | `RMSNorm` | `GemmaRMSNorm` |
| PP support | No (single rank) | Yes (`start_layer`/`end_layer`) |
| Deepstack embeds | No | Yes (layers 0-2) |
| Allreduce fusion | Yes (skip in TBO) | Yes (skip in TBO) |

## 3. Reuse Strategy

### 3.1 From `alibaba/qwen3-next-tbo-dev` branch (direct reuse):
- `Qwen2MoeSparseMoeBlock` op_* methods (+91 lines in `qwen2_moe.py`)
- `Qwen3HybridLinearDecoderLayer` op_* pattern -> adapt for `Qwen3_5LinearDecoderLayer`
- `Qwen3HybridAttentionDecoderLayer` op_* pattern -> adapt for `Qwen3_5AttentionDecoderLayer`
- GDN `op_prepare`/`op_core` -> adapt for `Qwen3_5GatedDeltaNet` (add DP-Attn padding + contiguous kernel)
- Operations strategy pattern (with `op_shared_experts`)

### 3.2 From current HEAD `qwen3_moe.py` (reference):
- `Qwen3MoeSparseMoeBlock` op_* methods: same pattern as Qwen2Moe but without shared experts
- `Qwen3MoeDecoderLayer` op_* methods: reference for layer-level TBO interface

## 4. Files to Modify

### 4.1 `python/sglang/srt/models/qwen2_moe.py`

Add op_* methods to `Qwen2MoeSparseMoeBlock` (directly from alibaba branch diff):
- `op_gate(self, state)` - run gate projection
- `op_shared_experts(self, state)` - run shared expert forward
- `op_select_experts(self, state)` - run topk routing
- `op_dispatch_a(self, state)` - EP dispatch phase a
- `op_dispatch_b(self, state)` - EP dispatch phase b
- `op_experts(self, state)` - run MoE experts core
- `op_combine_a(self, state)` - EP combine phase a
- `op_combine_b(self, state)` - EP combine phase b
- `op_output(self, state)` - merge shared expert output with MoE output

Add import: `is_non_idle_and_non_empty` from `sglang.srt.utils`

### 4.2 `python/sglang/srt/models/qwen3_5.py`

#### 4.2.1 `Qwen3_5GatedDeltaNet` - Add `op_prepare` and `op_core`

Adapt from Qwen3-Next's `Qwen3GatedDeltaNet.op_prepare`/`op_core` with:
- Use `fused_qkvzba_split_reshape_cat_contiguous` instead of `fused_qkvzba_split_reshape_cat`
- Use condition `self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu` (same as existing forward)
- Add DP-Attn padding in `op_core`: `if core_attn_out.shape != z.shape`

`op_prepare(self, state)`:
```
- Pop hidden_states_after_comm_pre_attn
- If idle: state.attn_intermediate_state = hidden_states, None; return
- Run _forward_input_proj
- Run fused_qkvzba_split_reshape_cat_contiguous (or fallback)
- Package kwargs dict (mixed_qkv, z, b, a, conv_weights, etc.)
- state.attn_intermediate_state = None, kwargs
```

`op_core(self, state)`:
```
- Pop attn_intermediate_state -> (hidden_state, kwargs)
- If idle: state.hidden_states_after_attn = hidden_state; return
- Call forward_batch.attn_backend.forward(...)
- Reshape + DP-Attn padding (if core_attn_out.shape != z.shape)
- Apply norm(core_attn_out, z)
- Reshape + out_proj
- state.hidden_states_after_attn = output
```

#### 4.2.2 `Qwen3_5LinearDecoderLayer` - Add TBO op_* methods

Fix: Change `is_layer_sparse` from local variable to `self.is_layer_sparse`.

Add methods (adapted from Qwen3-Next's `Qwen3HybridLinearDecoderLayer`):
- `op_comm_prepare_attn(self, state, positions, hidden_states, forward_batch, residual, tbo_subbatch_index=None)`
- `op_attn_prepare(self, state)` -> delegates to `self.linear_attn.op_prepare(state)`
- `op_attn_core(self, state)` -> delegates to `self.linear_attn.op_core(state)`
- `op_comm_prepare_mlp(self, state)`
- `op_comm_postprocess_layer(self, state)` -> returns output dict, clears state

Note: No `op_mlp` needed at layer level - MoE op_* methods handle MLP directly via `self.mlp.op_*` in the operations strategy.

#### 4.2.3 `Qwen3_5AttentionDecoderLayer` - Add TBO op_* methods

Fix: Change `is_layer_sparse` from local variable to `self.is_layer_sparse`.

Add methods (adapted from Qwen3-Next's `Qwen3HybridAttentionDecoderLayer`):
- `op_comm_prepare_attn(self, state, positions, hidden_states, forward_batch, residual, tbo_subbatch_index=None)`
- `op_attn_prepare(self, state)` - QKV projection, QK norm, RoPE, gate split
- `op_attn_core(self, state)` - RadixAttention forward, gate application, o_proj
- `op_comm_prepare_mlp(self, state)`
- `op_comm_postprocess_layer(self, state)` -> returns output dict, clears state

#### 4.2.4 `Qwen3_5ForCausalLM.forward` - Add TBO path

Add import: `model_forward_maybe_tbo` from `sglang.srt.batch_overlap.two_batch_overlap`
Add import: `ScatterMode` from `sglang.srt.layers.communicator`

Modify `forward()`:
```python
if forward_batch.can_run_tbo:
    hidden_states, residual = model_forward_maybe_tbo(
        layers=self.layers[self.start_layer:self.end_layer],
        enable_tbo=True,
        positions=positions,
        forward_batch=forward_batch,
        hidden_states=hidden_states,
        residual=residual,
        input_data_scatter_mode=ScatterMode.model_input_output(),
    )
else:
    # existing for loop
    for layer_idx in range(self.start_layer, self.end_layer):
        ...
```

Note: Deepstack embeds processing (layers 0-2) happens AFTER the TBO forward, in the non-TBO path only. For TBO, deepstack embeds are processed within each layer's op_comm_postprocess_layer, OR we skip them in the TBO path (since TBO is typically for decode, and deepstack embeds are only relevant during prefill with initial tokens). Need to verify: if `input_deepstack_embeds` is always None during decode (when TBO runs), then no special handling is needed.

### 4.3 `python/sglang/srt/batch_overlap/operations_strategy.py`

Register both Qwen3.5 layer types. Since both layer types share the same operations pattern (same as Qwen3-Next), add a single strategy function that handles both.

Add to `init_new_tbo`:
```python
elif layer_name in ["Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"]:
    return OperationsStrategy.concat([
        _compute_moe_qwen3_5_layer_operations_strategy_tbo(layer, forward_mode)
        for layer in layers
    ])
```

Add strategy functions (adapted from Qwen3-Next strategy):
- `_compute_moe_qwen3_5_layer_operations_strategy_tbo(layer, forward_mode)`
- `_compute_moe_qwen3_5_prefill(layer)` - uses `op_attn_prepare`/`op_attn_core` (not `self_attn.op_prepare`)
- `_compute_moe_qwen3_5_decode(layer)`

Operations order (prefill):
```
layer.op_comm_prepare_attn
layer.op_attn_prepare
layer.op_attn_core
layer.op_comm_prepare_mlp
layer.mlp.op_gate
layer.mlp.op_select_experts
layer.mlp.op_dispatch_a
YieldOperation()
layer.mlp.op_dispatch_b
layer.mlp.op_experts
layer.mlp.op_combine_a
YieldOperation()
layer.mlp.op_shared_experts
layer.mlp.op_combine_b
layer.mlp.op_output
layer.op_comm_postprocess_layer
```

Operations order (decode):
```
layer.op_comm_prepare_attn
layer.op_attn_prepare
layer.op_attn_core
YieldOperation()
layer.op_comm_prepare_mlp
layer.mlp.op_gate
layer.mlp.op_select_experts
YieldOperation()
layer.mlp.op_dispatch_a
layer.mlp.op_shared_experts
YieldOperation()
layer.mlp.op_dispatch_b
layer.mlp.op_experts
layer.mlp.op_combine_a
YieldOperation()
layer.mlp.op_combine_b
YieldOperation()
layer.mlp.op_output
layer.op_comm_postprocess_layer
```

## 5. Design Decisions

1. **Allreduce fusion**: Disabled in TBO path (same as DeepSeek/Qwen3-Next). TBO op_* methods call `self.mlp(hidden_states, forward_batch, use_reduce_scatter)` without `should_allreduce_fusion`, or handle MoE ops individually.

2. **`is_layer_sparse`**: Must be `self.is_layer_sparse` instance attribute for `OperationsStrategy` assertion. For Qwen3.5 MoE, all layers are sparse.

3. **PP support**: `model_forward_maybe_tbo` is called with `self.layers[self.start_layer:self.end_layer]`, naturally supporting PP.

4. **Deepstack embeds**: Only applied for layers 0-2 and typically during prefill. In TBO path, deepstack processing needs to be embedded in `op_comm_postprocess_layer` or handled post-TBO. Since TBO primarily runs during decode (`can_run_tbo`), and deepstack embeds are only used with initial embeddings, this should be safe to handle outside TBO. If needed, add deepstack logic inside `op_comm_postprocess_layer` by checking layer_id.

5. **Mixed layer types**: Qwen3.5 has both Linear and Attention layers. `OperationsStrategy.init_new_tbo` receives `layers` which may contain mixed types. The `layer_name` check uses `layers[0].__class__.__name__`, but mixed layers mean different names. The strategy needs to handle this - use `elif layer_name in [...]` to accept both names, and generate per-layer strategy individually (already handled by the concat pattern).

6. **`op_attn_prepare`/`op_attn_core` vs `self_attn.op_prepare`/`self_attn.op_core`**: Qwen3-Next uses `layer.op_attn_prepare` (layer-level delegation) instead of `layer.self_attn.op_prepare` (direct access). This is because the attn module name differs between layer types (`linear_attn` vs `self_attention`). We follow the same pattern.

## 6. Implementation Order

1. Add `op_*` methods to `Qwen2MoeSparseMoeBlock` in `qwen2_moe.py`
2. Add `op_prepare`/`op_core` to `Qwen3_5GatedDeltaNet` in `qwen3_5.py`
3. Fix `is_layer_sparse` + add TBO op_* to `Qwen3_5LinearDecoderLayer`
4. Fix `is_layer_sparse` + add TBO op_* to `Qwen3_5AttentionDecoderLayer`
5. Add TBO path to `Qwen3_5ForCausalLM.forward`
6. Register layer types in `operations_strategy.py`

### 4.4 `python/sglang/srt/batch_overlap/two_batch_overlap.py` (bug fixes)

Bug-fix changes applied to the TBO framework itself:

1. **`filter_batch` per-seq split list**: Added `mamba_track_indices`, `mamba_track_mask`, `mamba_track_seqlens` to the per-seq split list (line ~660-662) so they are properly sliced by `start_seq_index:end_seq_index`.

2. **`mrope_positions` slicing**: Added `mrope_positions[:, start:end]` slicing (line ~721-727) in `filter_batch` to properly handle multi-rope position tensors.

3. **`mamba_track_mask` two-chunk fix**: In `derive_fields_related_to_seq_len_for_two_chunk` (line ~619-621), disabled mamba tracking for child_a's boundary sequence to prevent h-tensor index out of bounds. See Bug #4 below for details.

## 7. Debugging Log

### Test command
```bash
nohup python3 test/registered/4-gpu-models/test_qwen35_two_batch_overlap.py > test.out 2>&1 &
```
Server args: `--tp 4 --dp 1 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --enable-two-batch-overlap --mamba-scheduler-strategy extra_buffer --mamba-track-interval 128`

Model: `/home/t4/models/Qwen/Qwen35-35B-FP8`

### Bug #1: `filter_batch` missing mamba fields (FIXED)

**Symptom**: `KeyError` / shape mismatch during TBO sub-batch creation.

**Root cause**: `filter_batch` in `two_batch_overlap.py` did not include `mamba_track_indices`, `mamba_track_mask`, `mamba_track_seqlens` in its per-seq split list. These fields are required by the GDN backend's `_init_track_conv_indices` and `_init_track_ssm_indices`.

**Fix**: Added the three fields to the per-seq split list at line ~660-662.

### Bug #2: `layer=None` in GDN `forward_decode` (FIXED)

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'layer_id'` in `gdn_backend.forward_decode`.

**Root cause**: `op_core` in `Qwen3_5GatedDeltaNet` was calling `forward_batch.attn_backend.forward(layer=None, ...)` instead of `forward_batch.attn_backend.forward(layer=self.attn, ...)`.

**Fix**: Changed `layer=None` to `layer=self.attn` in `op_core`.

### Bug #3: `mrope_positions` negative dimension (FIXED)

**Symptom**: `RuntimeError: Trying to create tensor with negative dimension` for `mrope_positions`.

**Root cause**: `mrope_positions` (shape `[3, num_tokens]`) was not being sliced in `filter_batch` when splitting tokens for TBO sub-batches.

**Fix**: Added `mrope_positions[:, start_token_index:end_token_index]` in `filter_batch` (line ~721-727).

### Bug #4: h-tensor index out of bounds in `_init_track_ssm_indices` (FIXED)

**Symptom**: `CUDA_ERROR_ASSERT` / `index out of bounds` in `IndexKernel.cu` during the first `forward_extend` after server startup. The error cascades to `deep_gemm` calls downstream.

**Root cause**: In the two-chunk split case, `derive_fields_related_to_seq_len_for_two_chunk` updates `extend_seq_lens` for both children (e.g., from 80 to 40 each) but does NOT update `mamba_track_seqlens` (stays at 80). This causes a mismatch in `_init_track_ssm_indices`:

- The FLA kernel's `h` tensor is sized based on `extend_seq_lens` (40 tokens -> 1 chunk -> 1 h-state)
- But `track_ssm_h_src` is computed from `mamba_track_seqlens` (80 -> `80 // 64 = 1` -> index 1)
- Index 1 is out of bounds for h with only 1 entry

Debug output confirming the issue:
```
[DEBUG GDN extend] layer=0 seq_len=40 bs=1
  query_start_loc=[0, 40]       # correct for sub-batch
  extend_seq_lens=[40]           # correct (updated by two-chunk)
  mamba_track_seqlens=[80]       # NOT updated (full sequence)
  mamba_track_mask=[True]        # should be False for child_a
  tbo_parent_token_range=(0, 40) # child_a
```

**Fix**: In `derive_fields_related_to_seq_len_for_two_chunk`, disable mamba tracking for child_a's boundary sequence:
```python
if child_a.mamba_track_mask is not None:
    child_a.mamba_track_mask = child_a.mamba_track_mask.clone()
    child_a.mamba_track_mask[-1] = False
```
This is correct because:
- child_a only processes the first portion of the boundary sequence
- The tracking position (aligned to `mamba_cache_chunk_size`) typically falls beyond child_a's token range
- child_b handles tracking correctly since its `extend_prefix_lens` is updated to account for child_a's tokens, and `lens_to_track = mamba_track_seqlens - prefix_lens` yields correct indices

### Bug #5: 1-token extend triggers empty child_a (FIXED)

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'tensor_split'` in `_scatter_hidden_states_and_residual` when `hidden_states_after_attn` is None.

**Root cause**: A 1-token extend batch (`extend_seq_lens_cpu=[1]`) incorrectly triggers two-chunk split. `_is_two_chunk_split_enabled([1])` returns True because `left_sum=0 < 1*0.48`. Then `compute_split_token_index` returns `sum([1])//2 = 0`, producing child_a with `tbo_parent_token_range=(0, 0)` — 0 tokens but batch_size=1. This causes empty tensors to flow through op_prepare/op_core, producing None outputs.

Even without two-chunk, vanilla split also fails for `[1]`: `_split_array_by_balanced_sum([1])` returns 0, giving `sum([1][:0])=0` — the same empty child_a. With only 1 total token, you fundamentally cannot split into two non-empty halves.

**Fix**: In `compute_split_seq_index`, return None (disabling TBO for the batch) when `sum(extend_lens) < 2`:
```python
if forward_mode == ForwardMode.EXTEND:
    assert extend_lens is not None
    if sum(extend_lens) < 2:
        return None
    return _split_extend_seqs(extend_lens)
```
When `compute_split_seq_index` returns None, `local_can_run_tbo` becomes False in `TboDPAttentionPreparer`, and the batch uses the regular (non-TBO) forward path.

### Bug #6: TboAttnBackend missing `forward` method (FIXED)

**Symptom**: `TypeError: AttentionBackend.forward() missing 3 required positional arguments: 'q', 'k', and 'v'` in `radix_linear_attention.py:95`.

**Root cause**: When TBO is disabled for a specific batch (e.g., due to Bug #5 fix), the batch uses the regular forward path. However, `forward_batch.attn_backend` is still `TboAttnBackend` (since TBO is enabled globally). `TboAttnBackend` only overrode `forward_extend` and `forward_decode` (delegating to `self.primary`), but not `forward`. The `RadixLinearAttention.forward` calls `attn_backend.forward(layer=..., mixed_qkv=..., a=..., b=...)`, which falls back to the base `AttentionBackend.forward(self, q, k, v, ...)` — a signature mismatch.

**Fix**: Added `forward` method to `TboAttnBackend` in `tbo_backend.py`:
```python
def forward(self, *args, **kwargs):
    return self.primary.forward(*args, **kwargs)
```

## 8. Test Result

```
Ran 1 test in 439.569s
OK
Score: 0.970 (threshold: 0.93)
```
