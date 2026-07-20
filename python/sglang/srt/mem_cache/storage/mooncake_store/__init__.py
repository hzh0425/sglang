from __future__ import annotations

import logging

from sglang.srt.mem_cache.external_cache_pool import (
    build_deepseek_v4_external_pool_stack,
)
from sglang.srt.mem_cache.registry import (
    _create_unified_radix_cache,
    register_radix_cache_backend,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
    MooncakeConnector,
)

logger = logging.getLogger(__name__)


def _group_rank_and_size(group, fallback_rank: int, fallback_size: int):
    if group is None:
        return fallback_rank, fallback_size
    import torch.distributed as dist

    return dist.get_rank(group=group), dist.get_world_size(group=group)


def _mooncake_factory(ctx):
    cache = _create_unified_radix_cache(ctx, ctx.server_args, ctx.params)
    kvcache = ctx.params.token_to_kv_pool_allocator.get_kvcache()
    pool_stack = build_deepseek_v4_external_pool_stack(kvcache, cache.page_size)
    attn_cp_rank, attn_cp_size = _group_rank_and_size(
        ctx.params.attn_cp_cache_group,
        ctx.params.attn_cp_rank,
        ctx.params.attn_cp_size,
    )
    connector = MooncakeConnector(
        pool_stack=pool_stack,
        model_config=ctx.model_config,
        server_args=ctx.server_args,
        tp_rank=ctx.tp_rank,
        tp_size=ctx.tp_size,
        pp_rank=ctx.params.pp_rank,
        pp_size=ctx.params.pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
    )
    cache.install_external_cache(
        connector,
        sidecars=pool_stack.sidecars,
        component_pools=pool_stack.component_pools,
    )
    return cache


try:
    register_radix_cache_backend("mooncake", _mooncake_factory)
except ValueError as exc:
    logger.debug("Mooncake radix backend already registered: %s", exc)
