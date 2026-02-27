from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
    page_align_keys,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


MAX_PRIORITY = 1 << 30


def _now() -> float64:
    ret = TreeNode.clock
    TreeNode.clock += 1.0
    return ret


@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0


class TreeNode:
    counter = 0
    clock = float64(1.0)

    def __init__(self, component_names: List[str]):
        self.id = TreeNode.counter
        TreeNode.counter += 1
        self.parent: Optional["TreeNode"] = None
        self.children: Dict[object, "TreeNode"] = {}
        self.key: Optional[RadixKey] = None
        self.data = {name: ComponentData() for name in component_names}
        self.last_access_time = _now()


class LRUList:
    def __init__(self, name: str):
        self.name = name
        self.prv = f"{name}_prev"
        self.nxt = f"{name}_next"
        self.cache: Dict[int, TreeNode] = {}
        self.head = object()
        self.tail = object()
        setattr(self.head, self.nxt, self.tail)
        setattr(self.tail, self.prv, self.head)

    def _add_after(self, old_node, new_node):
        setattr(new_node, self.prv, old_node)
        setattr(new_node, self.nxt, getattr(old_node, self.nxt))
        setattr(getattr(old_node, self.nxt), self.prv, new_node)
        setattr(old_node, self.nxt, new_node)

    def _remove(self, node):
        setattr(getattr(node, self.prv), self.nxt, getattr(node, self.nxt))
        setattr(getattr(node, self.nxt), self.prv, getattr(node, self.prv))

    def in_list(self, node: Optional[TreeNode]) -> bool:
        return node is not None and node.id in self.cache

    def insert_mru(self, node: TreeNode):
        if node.id in self.cache:
            self._remove(node)
        else:
            self.cache[node.id] = node
        self._add_after(self.head, node)

    def remove_node(self, node: TreeNode):
        if node.id not in self.cache:
            return
        del self.cache[node.id]
        self._remove(node)

    def reset_node_mru(self, node: TreeNode):
        if node.id not in self.cache:
            return
        self._remove(node)
        self._add_after(self.head, node)

    def _prev_no_lock(self, node, leaf_only: bool) -> Optional[TreeNode]:
        x = getattr(node, self.prv)
        while x != self.head:
            if leaf_only and len(x.children) > 0:
                x = getattr(x, self.prv)
                continue
            if x.data[self.name].lock_ref > 0:
                x = getattr(x, self.prv)
                continue
            return x
        return None

    def get_lru_no_lock(self):
        return self._prev_no_lock(self.tail, leaf_only=False)

    def get_leaf_lru_no_lock(self):
        return self._prev_no_lock(self.tail, leaf_only=True)

    def get_prev_no_lock(self, node: TreeNode, check_id: bool = True):
        if check_id and node.id not in self.cache:
            return None
        return self._prev_no_lock(node, leaf_only=False)

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        if check_id and node.id not in self.cache:
            return None
        return self._prev_no_lock(node, leaf_only=True)


class Component(ABC):
    def __init__(self, name: str, is_primary: bool, legacy_name: Optional[str] = None):
        self.name = name
        self.is_primary = is_primary
        self.legacy_name = legacy_name
        self.lru = LRUList(name)
        self.evictable = 0
        self.protected = 0
        self.skip_free_when_primary = False
        self.tree: Optional["HybridRadixCache"] = None

    def bind(self, tree: "HybridRadixCache"):
        self.tree = tree

    def insert_value(self, params: InsertParams, primary_value: torch.Tensor):
        return None

    def init_match_ctx(self):
        return {}

    @abstractmethod
    def priority(self, is_leaf: bool) -> int: ...

    @abstractmethod
    def lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]: ...

    @abstractmethod
    def on_match(self, node: TreeNode, ctx: dict) -> bool: ...

    def on_split(self, new_node: TreeNode, child: TreeNode, split_len: int): ...

    def on_traverse(self, node: TreeNode, hook: dict): ...

    def on_leaf(self, node: TreeNode, value, hook: dict) -> bool:
        return False

    @abstractmethod
    def free_value(self, node: TreeNode) -> int: ...


class FullComponent(Component):
    def __init__(self, name: str):
        super().__init__(name, True, "full")

    def priority(self, is_leaf: bool) -> int:
        return MAX_PRIORITY

    def lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        ret = []
        while node != root:
            ret.append(node)
            node = node.parent
        return ret

    def on_match(self, node: TreeNode, ctx: dict) -> bool:
        return node.data[self.name].value is not None

    def on_split(self, new_node: TreeNode, child: TreeNode, split_len: int):
        new_node.data[self.name].value = child.data[self.name].value[:split_len].clone()
        child.data[self.name].value = child.data[self.name].value[split_len:].clone()
        new_node.data[self.name].lock_ref = child.data[self.name].lock_ref
        self.lru.remove_node(child)
        self.lru.insert_mru(new_node)
        self.lru.insert_mru(child)

    def free_value(self, node: TreeNode) -> int:
        value = node.data[self.name].value
        if value is None:
            return 0
        num = len(value)
        if num > 0:
            self.tree.token_to_kv_pool_allocator.free(value)
        node.data[self.name].value = None
        return num


class MambaComponent(Component):
    def __init__(self, name: str):
        super().__init__(name, False, "mamba")

    def insert_value(self, params: InsertParams, primary_value: torch.Tensor):
        return params.mamba_value

    def priority(self, is_leaf: bool) -> int:
        return MAX_PRIORITY if is_leaf else 1

    def lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        return [node] if node.data[self.name].value is not None else []

    def on_match(self, node: TreeNode, ctx: dict) -> bool:
        return node.data[self.name].value is not None

    def on_split(self, new_node: TreeNode, child: TreeNode, split_len: int):
        new_node.data[self.name].value = None
        new_node.data[self.name].lock_ref = 0

    def on_leaf(self, node: TreeNode, value, hook: dict) -> bool:
        if value is None:
            return False
        existed = node.data[self.name].value is not None
        node.data[self.name].value = value if not existed else node.data[self.name].value
        if existed:
            self.lru.reset_node_mru(node)
        else:
            self.lru.insert_mru(node)
            self.evictable += len(value)
        return existed

    def free_value(self, node: TreeNode) -> int:
        value = node.data[self.name].value
        if value is None:
            return 0
        num = len(value)
        if num > 0 and hasattr(self.tree.req_to_token_pool, "mamba_pool"):
            self.tree.req_to_token_pool.mamba_pool.free(value)
        node.data[self.name].value = None
        return num


class SWAComponent(Component):
    def __init__(self, name: str, window_size: int):
        super().__init__(name, False, "swa")
        self.window_size = window_size
        self.skip_free_when_primary = True

    def insert_value(self, params: InsertParams, primary_value: torch.Tensor):
        return primary_value

    def priority(self, is_leaf: bool) -> int:
        return MAX_PRIORITY if is_leaf else 1

    def lock_range(self, node: TreeNode, root: TreeNode) -> List[TreeNode]:
        ret, total = [], 0
        while node != root and total < self.window_size:
            value = node.data[self.name].value
            if value is not None:
                ret.append(node)
                total += len(value)
            node = node.parent
        return ret

    def init_match_ctx(self):
        return {"dist": float("inf")}

    def on_match(self, node: TreeNode, ctx: dict) -> bool:
        value = node.data[self.name].value
        if value is None:
            ctx["dist"] = 0
            return False
        ctx["dist"] += len(value)
        return ctx["dist"] >= self.window_size

    def on_traverse(self, node: TreeNode, hook: dict):
        if node.data[self.name].value is not None:
            return
        if hook.get("swa_evicted_seqlen", 0) > hook["total_prefix_length"]:
            return
        primary = node.data[hook["primary"]].value
        if primary is None:
            return
        node.data[self.name].value = primary
        self.lru.insert_mru(node)
        self.evictable += len(primary)

    def on_split(self, new_node: TreeNode, child: TreeNode, split_len: int):
        if child.data[self.name].value is None:
            new_node.data[self.name].value = None
            new_node.data[self.name].lock_ref = child.data[self.name].lock_ref
            return
        p = self.tree.primary.name
        new_node.data[self.name].value = new_node.data[p].value
        child.data[self.name].value = child.data[p].value
        new_node.data[self.name].lock_ref = child.data[self.name].lock_ref
        self.lru.remove_node(child)
        self.lru.insert_mru(new_node)
        self.lru.insert_mru(child)

    def on_leaf(self, node: TreeNode, value, hook: dict) -> bool:
        primary = node.data[hook["primary"]].value
        if primary is None:
            return False
        if node.data[self.name].value is None:
            node.data[self.name].value = primary
            self.lru.insert_mru(node)
            self.evictable += len(primary)
        else:
            self.lru.reset_node_mru(node)
        return False

    def free_value(self, node: TreeNode) -> int:
        value = node.data[self.name].value
        if value is None:
            return 0
        num = len(value)
        if num > 0 and hasattr(self.tree.token_to_kv_pool_allocator, "free_swa"):
            self.tree.token_to_kv_pool_allocator.free_swa(node.data[self.tree.primary.name].value)
        node.data[self.name].value = None
        return num


class HybridRadixCache(BasePrefixCache):
    def __init__(self, params: "CacheInitParams"):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.device = self.token_to_kv_pool_allocator.device if self.token_to_kv_pool_allocator else torch.device("cpu")
        if params.enable_metrics:
            self.init_metrics_collector()
        self.key_match_fn = _key_match_page_size1 if self.page_size == 1 else partial(_key_match_paged, page_size=self.page_size)
        self.get_child_key_fn = get_child_key if self.page_size == 1 else partial(get_child_key, page_size=self.page_size)

        names = params.hybrid_components or ["full"]
        primary_name = params.hybrid_primary_component
        if primary_name not in names:
            names = [primary_name] + names
        self.components: Dict[str, Component] = {}
        self.primary: Optional[Component] = None
        for name in names:
            if name == primary_name:
                comp: Component = FullComponent(name)
                self.primary = comp
            elif name == "mamba":
                comp = MambaComponent(name)
            elif name == "swa":
                comp = SWAComponent(name, params.hybrid_swa_window_size or params.sliding_window_size or 1)
            else:
                raise ValueError(f"unsupported component: {name}")
            comp.bind(self)
            self.components[name] = comp
        self.reset()

    def _legacy(self, name: str) -> Optional[Component]:
        for comp in self.components.values():
            if comp.legacy_name == name:
                return comp
        return None

    def supports_swa(self) -> bool:
        return self._legacy("swa") is not None

    def supports_mamba(self) -> bool:
        return self._legacy("mamba") is not None

    def reset(self):
        self.root_node = TreeNode(list(self.components.keys()))
        self.root_node.key = RadixKey([], None)
        for comp in self.components.values():
            comp.evictable = 0
            comp.protected = 0
            self.root_node.data[comp.name].value = [] if comp.is_primary else None
            self.root_node.data[comp.name].lock_ref = 1

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        if self.disable or len(key) == 0:
            return MatchResult(torch.empty((0,), dtype=torch.int64, device=self.device), self.root_node, self.root_node)
        if self.page_size != 1:
            key = key[: len(page_align_keys(key.token_ids, self.page_size))]
            if len(key) == 0:
                return MatchResult(torch.empty((0,), dtype=torch.int64, device=self.device), self.root_node, self.root_node)

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        values, best_len, best_node = [], 0, node
        ctx = {c.name: c.init_match_ctx() for c in self.components.values()}
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split(child.key, child, prefix_len)
                for comp in self.components.values():
                    comp.on_split(new_node, child, prefix_len)
                node = new_node
                values.append(node.data[self.primary.name].value)
                if all(comp.on_match(node, ctx[comp.name]) for comp in self.components.values()):
                    best_len, best_node = len(values), node
                break
            node = child
            values.append(node.data[self.primary.name].value)
            key = key[prefix_len:]
            if all(comp.on_match(node, ctx[comp.name]) for comp in self.components.values()):
                best_len, best_node = len(values), node
            if len(key):
                child_key = self.get_child_key_fn(key)
        matched = torch.cat(values[:best_len]) if best_len else torch.empty((0,), dtype=torch.int64, device=self.device)
        for comp in self.components.values():
            comp.lru.reset_node_mru(best_node) if best_node != self.root_node else None
        return MatchResult(matched, best_node, best_node)

    def _split(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        new_node = TreeNode(list(self.components.keys()))
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)
        key, primary_value = params.key, params.value
        if primary_value is None:
            primary_value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        if self.page_size != 1:
            page_len = len(primary_value) // self.page_size * self.page_size
            key, primary_value = key[:page_len], primary_value[:page_len]
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=False)

        values = {self.primary.name: primary_value}
        for comp in self.components.values():
            if not comp.is_primary:
                values[comp.name] = comp.insert_value(params, primary_value)

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        prefix_len, cursor = 0, primary_value
        existed = {}
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            match = self.key_match_fn(node.key, key)
            if params.prev_prefix_len < prefix_len + match:
                start = max(0, params.prev_prefix_len - prefix_len)
                self.token_to_kv_pool_allocator.free(cursor[start:match])
            if match < len(node.key):
                split = self._split(node.key, node, match)
                for comp in self.components.values():
                    comp.on_split(split, node, match)
                node = split
            prefix_len += match
            key = key[match:]
            cursor = cursor[match:]
            for comp in self.components.values():
                comp.on_traverse(node, {"total_prefix_length": prefix_len, "swa_evicted_seqlen": params.swa_evicted_seqlen, "primary": self.primary.name})
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            leaf = TreeNode(list(self.components.keys()))
            leaf.parent = node
            leaf.key = key
            leaf.data[self.primary.name].value = cursor.clone()
            node.children[child_key] = leaf
            self.primary.lru.insert_mru(leaf)
            self.primary.evictable += len(cursor)
            node = leaf

        for comp in self.components.values():
            if not comp.is_primary:
                existed[comp.name] = comp.on_leaf(node, values.get(comp.name), {"primary": self.primary.name})
        mamba = self._legacy("mamba")
        return InsertResult(prefix_len=prefix_len, mamba_exist=existed.get(mamba.name, False) if mamba else False)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult(component_num_evicted={})
        req = dict(params.component_requests or {})
        if not req:
            req[self.primary.name] = params.num_tokens
            swa, mamba = self._legacy("swa"), self._legacy("mamba")
            if swa:
                req[swa.name] = params.swa_num_tokens
            if mamba:
                req[mamba.name] = params.mamba_num

        start = time.perf_counter()
        out = {}
        for name, amount in req.items():
            comp = self.components.get(name)
            if comp is None or amount <= 0:
                continue
            node = comp.lru.get_leaf_lru_no_lock() if comp.is_primary else comp.lru.get_lru_no_lock()
            evicted = 0
            while evicted < amount and comp.lru.in_list(node):
                is_leaf = len(node.children) == 0
                trig = comp.priority(is_leaf)
                affected = [c for c in self.components.values() if node.data[c.name].value is not None and c.priority(is_leaf) <= trig]
                next_node = comp.lru.get_prev_leaf_no_lock(node) if comp.is_primary else comp.lru.get_prev_no_lock(node)
                if any(node.data[c.name].lock_ref > 0 for c in affected):
                    node = next_node
                    continue
                has_primary = any(c.is_primary for c in affected)
                for c in affected:
                    release = not (has_primary and c.skip_free_when_primary)
                    size = c.free_value(node) if release else len(node.data[c.name].value)
                    node.data[c.name].value = None
                    c.lru.remove_node(node)
                    c.evictable -= size
                    if c.name == comp.name:
                        evicted += size
                node = next_node
            out[name] = evicted
        self.update_eviction_metrics(sum(out.values()), start)
        ret = EvictResult(component_num_evicted=out)
        ret.num_tokens_evicted = out.get(self.primary.name, 0)
        swa, mamba = self._legacy("swa"), self._legacy("mamba")
        ret.swa_num_tokens_evicted = out.get(swa.name, 0) if swa else 0
        ret.mamba_num_evicted = out.get(mamba.name, 0) if mamba else 0
        return ret

    def inc_lock_ref(self, node: TreeNode):
        handle = {}
        for comp in self.components.values():
            nodes = comp.lock_range(node, self.root_node)
            handle[comp.name] = nodes
            for n in nodes:
                data = n.data[comp.name]
                if data.lock_ref == 0 and data.value is not None:
                    size = len(data.value)
                    comp.evictable -= size
                    comp.protected += size
                data.lock_ref += 1
        return handle

    def dec_lock_ref(self, node: TreeNode, swa_uuid_for_lock: Optional[dict] = None):
        handle = swa_uuid_for_lock or {}
        for comp in self.components.values():
            for n in handle.get(comp.name, []):
                data = n.data[comp.name]
                data.lock_ref -= 1
                if data.lock_ref == 0 and data.value is not None:
                    size = len(data.value)
                    comp.evictable += size
                    comp.protected -= size

    def full_evictable_size(self) -> int:
        return self.primary.evictable

    def full_protected_size(self) -> int:
        return self.primary.protected

    def swa_evictable_size(self) -> int:
        comp = self._legacy("swa")
        return comp.evictable if comp else 0

    def swa_protected_size(self) -> int:
        comp = self._legacy("swa")
        return comp.protected if comp else 0

    def mamba_evictable_size(self) -> int:
        comp = self._legacy("mamba")
        return comp.evictable if comp else 0

    def mamba_protected_size(self) -> int:
        comp = self._legacy("mamba")
        return comp.protected if comp else 0

