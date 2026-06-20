use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

use crate::component_type::ComponentType;
use crate::components::{Component, FullComponent, IncLockRefResult, MambaComponent, SwaComponent};
use crate::deferred_action::DeferredAction;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::tree_node_lru::{
    EvictRequest, EvictResult, FullLRUSlot, HostFullLRUSlot, HostMambaLRUSlot, HostSwaLRUSlot,
    LRUSlot, MambaLRUSlot, SwaLRUSlot, evict_host_full, evict_host_non_full,
};
use crate::tree_node_pool::{
    ChildKeyType, MatchChildResult, NodeIdx, PageSize, TreeNode, TreeNodePool,
};

/// Result of prefix match.
pub struct MatchResult {
    pub device_indices: Tensor,
    pub last_device_node_idx: NodeIdx,
    /// Cache-repair hint: a chunk-aligned position past the cached
    /// prefix where the request's prefill could capture and save
    /// SSM state to repair a tombstoned chunk boundary in the
    /// radix tree.
    ///
    /// The scheduler may use this field to snapshot mid-prefill SSM
    /// state at the repair boundary.
    pub mamba_branching_seqlen: Option<usize>,
    /// Cached Mamba state at `best_match_node_idx`. `None` indicates
    /// cache miss, where prefill starts from scratch.
    pub mamba_value: Option<Tensor>,

    // --- HiCache ---
    /// Deepest host-backed node on the matched path (the load-back start).
    pub last_host_node_idx: NodeIdx,
    /// Deepest node accepted by all configured components when host values are
    /// allowed. This is the load-back target for Full/SWA/Mamba.
    pub best_match_node_idx: NodeIdx,
    /// FULL KV length of the host-only suffix — host-backed, device-evicted nodes
    /// past `last_device_node_idx` (the tokens load-back restores H->D). When
    /// non-zero, `last_device_node_idx` is `host_only_length` above
    /// `last_host_node_idx`.
    pub host_only_length: usize,
    /// SWA host-token length that needs host->device restoration.
    pub swa_host_hit_length: usize,
    /// Mamba host-state count that needs host->device restoration.
    pub mamba_host_hit_length: usize,
}

/// Result of `prepare_load_back`: the host-only chain to restore H->D.
pub struct PrepareLoadBackResult {
    /// Host-only chain, root->leaf order — the nodes to restore.
    pub chain: Vec<NodeIdx>,
    /// Concatenated host KV indices for `chain` (the controller's load input).
    pub host_indices: Tensor,
    /// Device-present anchor the chain hangs under; device value is locked here
    /// so the whole path to root won't be evicted.
    pub ancestor_node_idx: NodeIdx,
}

/// Result of `prepare_aux_load_back`: aux host values to restore H->D.
pub struct PrepareAuxLoadBackResult {
    pub swa_chain: Vec<NodeIdx>,
    pub swa_host_indices: Tensor,
    pub mamba_chain: Vec<NodeIdx>,
    pub mamba_host_indices: Tensor,
}

pub struct InsertResult {
    /// The number of tokens in the insert key that matched existing nodes.
    pub prefix_len: usize,
    /// True if leaf creation was skipped (e.g. SWA vetoed because the
    /// suffix is outside the window).
    pub leaf_creation_skipped: bool,
    /// Indicates whether the cache has taken ownership of the
    /// `mamba_value` (`false`), or the caller should free it (`true`).
    pub mamba_value_exists: bool,
    /// KV cache allocator pool actions.
    pub deferred_actions: Vec<DeferredAction>,
}

/// Borrowed query key proven to be page-aligned and non-empty.
/// Constructed once at the public API boundary; passed to walk helpers
/// so they can skip alignment / empty checks and treat
/// `K::make_child_key` as infallible. Atom-generic so the same shape
/// works for single-token (`Atom = i64`) and bigram (`Atom = (i64, i64)`)
/// instantiations.
struct PageAlignedQueryKey<'a, A> {
    key: &'a [A],
}

impl<'a, A> PageAlignedQueryKey<'a, A> {
    /// Returns `None` if the page-aligned length is 0 (empty input or
    /// `key.len() < page_size`). Caller should early-return on `None`.
    fn new(key: &'a [A], page_size: PageSize) -> Option<Self> {
        let ps = page_size.get();
        let aligned_len = key.len() / ps * ps;
        if aligned_len == 0 {
            None
        } else {
            Some(Self {
                key: &key[..aligned_len],
            })
        }
    }

    fn as_slice(&self) -> &[A] {
        self.key
    }
}

fn cat_or_empty(empty_tensor: &Tensor, values: Vec<Tensor>) -> Tensor {
    if values.is_empty() {
        empty_tensor.shallow_clone()
    } else {
        Tensor::cat(&values, 0)
    }
}

/// Build the tree's component list from the cache config. Currently
/// supported combinations: Full Attention; Full Attention + Sliding Window
/// Attention; Full Attention + Mamba (Linear Attention).
fn build_components<K: ChildKeyType>(
    sliding_window_size: Option<usize>,
    mamba_cache_chunk_size: Option<usize>,
    page_size: usize,
    enable_hicache: bool,
    hicache_write_back: bool,
) -> Result<Vec<Box<dyn Component<K>>>, RadixCacheInitError> {
    if sliding_window_size.is_some() && mamba_cache_chunk_size.is_some() {
        return Err(RadixCacheInitError::SwaMambaComboNotSupported);
    }
    let mut components: Vec<Box<dyn Component<K>>> = Vec::new();
    components.push(Box::new(FullComponent::new(
        enable_hicache,
        hicache_write_back,
    )));
    if let Some(window) = sliding_window_size {
        components.push(Box::new(SwaComponent::new(window)?));
    }
    if let Some(chunk_size) = mamba_cache_chunk_size {
        components.push(Box::new(MambaComponent::new(chunk_size, page_size)?));
    }
    Ok(components)
}

/// Radix tree-based KV cache, generic over the child key type:
///   - `RadixCache<i64>`     for `page_size = 1` (one token per edge segment)
///   - `RadixCache<Vec<i64>>` for `page_size > 1` (token page per edge segment)
///
/// TODO(Jialin): Decide how to handle EAGLE bigram. Either pre-expand tokens
/// outside (cleanest), or fold it inside `RadixCache` by manipulating the
/// tree_node_pool key layout. Defer until EAGLE is in scope.
pub struct RadixCache<K: ChildKeyType> {
    // -- Core data structures --
    /// Arena that owns all tree nodes; recycles slots via a freelist.
    tree_node_pool: TreeNodePool<K>,

    /// Root for queries with `extra_key = None`. Always present — re-allocated
    /// on `reset()`.
    default_root: NodeIdx,

    /// Root per `extra_key`. Allocated lazily by `insert` on first use of a
    /// namespace; owns the key string (PyO3 has already copied from Python by
    /// the time it reaches the wrapper, so insert just moves the `String` in).
    ///
    /// TODO(Jialin): Drop entries when their subtree empties via eviction —
    /// otherwise transient extra_keys leak nodes over time. Hook into
    /// `_update_leaf_status` once it lands.
    named_roots: HashMap<String, NodeIdx>,

    // -- Constructor arguments retained for later use --
    /// Validated once at construction; reused on every `reset()` rebuild and
    /// exposed via `page_size()`. `PageSize` is `Copy`, so storing it
    /// here is essentially free.
    page_size: PageSize,

    /// Preserved so `reset()` can rebuild the pool with the same capacity.
    init_node_capacity: usize,

    /// Empty Int64 tensor on the configured device, returned (via shallow_clone)
    /// when `match_prefix` finds no match. Avoids per-call allocation.
    empty_tensor: Tensor,

    /// Per-component shells used for orchestrator-level operations
    /// (match validators today; future SWA acquire/release / cascade
    /// evict / insert hooks). Built once at construction from the
    /// `sliding_window_size` arg — `FullComponent` is always present;
    /// `SwaComponent` joins iff `Some(W)`. Iterating this vec in
    /// `match_prefix` yields the validator chain that gates the match
    /// boundary. Hot-path per-node LRU operations DO NOT go through
    /// this vec — they stay statically dispatched on
    /// `tree_node_lru.rs::*LRUSlot` markers.
    ///
    /// Mirrors OSS `unified_radix_cache.py`'s `self._components_tuple`
    /// shape (config-driven list of configured components).
    components: Vec<Box<dyn Component<K>>>,

    /// Whether SWA is configured. Captured at construction from
    /// `sliding_window_size.is_some()`. Stored separately from
    /// `components` so it doesn't require iterating + downcasting to
    /// check, and so `reset()` can pass it back to the rebuilt pool
    /// without re-deriving.
    has_swa_component: bool,

    /// Sliding-window size when SWA is configured.
    sliding_window_size: Option<usize>,

    /// Whether Mamba is configured. Parallel to `has_swa_component`.
    has_mamba_component: bool,

    /// Mamba chunk size. Mamba state checkpoints are saved only at
    /// positions that are multiples of `chunk_size`.
    mamba_cache_chunk_size: Option<usize>,

    /// Whether the host (CPU/L2) cache tier is active.
    enable_hicache: bool,

    /// Whether the HiCache write-back policy is active
    hicache_write_back: bool,
}

impl<K: ChildKeyType> RadixCache<K> {
    /// Construct a radix cache from per-cache config.
    ///
    /// - `init_node_capacity`: initial size of the tree node pool;
    /// - `sliding_window_size`: pass the per-token SWA window to enable
    ///   Sliding Window Attention;
    /// - `mamba_cache_chunk_size`: pass the SSM checkpoint chunk size to
    ///   enable Mamba (Linear Attention);
    /// - `enable_hicache`: toggle the host (CPU/L2) cache tier.
    /// - `hicache_write_back`: toggle hicache write_back policy (e.g. backup on eviction)
    pub fn new(
        device: Device,
        page_size: usize,
        init_node_capacity: usize,
        sliding_window_size: Option<usize>,
        mamba_cache_chunk_size: Option<usize>,
        enable_hicache: bool,
        hicache_write_back: bool,
    ) -> Result<Self, RadixCacheInitError> {
        let page_size = PageSize::new(page_size)?;
        let has_swa_component = sliding_window_size.is_some();
        let has_mamba_component = mamba_cache_chunk_size.is_some();
        let components = build_components(
            sliding_window_size,
            mamba_cache_chunk_size,
            page_size.get(),
            enable_hicache,
            hicache_write_back,
        )?;
        let mut tree_node_pool = TreeNodePool::<K>::new(
            page_size,
            init_node_capacity,
            has_swa_component,
            has_mamba_component,
        );
        let default_root = tree_node_pool.alloc(TreeNode::new_root());
        let empty_tensor = Tensor::empty([0], (Kind::Int64, device));
        Ok(Self {
            tree_node_pool,
            default_root,
            named_roots: HashMap::new(),
            page_size,
            init_node_capacity,
            empty_tensor,
            components,
            has_swa_component,
            sliding_window_size,
            has_mamba_component,
            mamba_cache_chunk_size,
            enable_hicache,
            hicache_write_back,
        })
    }

    /// Recreate a new empty Radix Cache.
    pub fn reset(&mut self) {
        // Infallible: page_size was validated at construction.
        let mut new_tree_node_pool = TreeNodePool::<K>::new(
            self.page_size,
            self.init_node_capacity,
            self.has_swa_component,
            self.has_mamba_component,
        );
        let new_default_root = new_tree_node_pool.alloc(TreeNode::new_root());
        self.tree_node_pool = new_tree_node_pool;
        self.default_root = new_default_root;
        self.named_roots.clear();
    }

    /// Node index for the default namespace root (`extra_key = None`).
    pub fn default_root_idx(&self) -> NodeIdx {
        self.default_root
    }

    fn empty_match_result(&self, last_device_node_idx: NodeIdx) -> MatchResult {
        MatchResult {
            device_indices: self.empty_tensor.shallow_clone(),
            last_device_node_idx,
            last_host_node_idx: last_device_node_idx,
            best_match_node_idx: last_device_node_idx,
            host_only_length: 0,
            swa_host_hit_length: 0,
            mamba_host_hit_length: 0,
            mamba_branching_seqlen: None,
            mamba_value: None,
        }
    }

    fn full_host_hit_length(&self, best_match: NodeIdx, last_device: NodeIdx) -> usize {
        let mut total = 0usize;
        let mut cur = best_match;
        while cur != last_device {
            let node = self.tree_node_pool.get(cur);
            if node.is_root() {
                break;
            }
            if let Some(host_value) = HostFullLRUSlot::value(node) {
                total += host_value.size()[0] as usize;
            }
            let Some(parent) = node.parent() else { break };
            cur = parent;
        }
        total
    }

    fn swa_host_hit_length(&self, best_match: NodeIdx) -> usize {
        let Some(window) = self.sliding_window_size else {
            return 0;
        };
        let mut total = 0usize;
        let mut seen = 0usize;
        let mut cur = best_match;
        while seen < window {
            let node = self.tree_node_pool.get(cur);
            if node.is_root() {
                break;
            }
            if SwaLRUSlot::has_value(node) {
                seen += SwaLRUSlot::value_len(node);
            } else if HostSwaLRUSlot::has_value(node) {
                let len = HostSwaLRUSlot::value_len(node);
                total += len;
                seen += len;
            } else {
                break;
            }
            let Some(parent) = node.parent() else { break };
            cur = parent;
        }
        total
    }

    fn mamba_host_hit_length(&self, best_match: NodeIdx) -> usize {
        if !self.has_mamba_component {
            return 0;
        }
        let node = self.tree_node_pool.get(best_match);
        usize::from(!MambaLRUSlot::has_value(node) && HostMambaLRUSlot::has_value(node))
    }

    /// Resolve the namespace root for `extra_key`, lazily creating it on
    /// first use. Used by both `match_prefix` and `insert` so the caller
    /// always gets back a `last_device_node_idx` that lives in the same
    /// namespace as the query (important for follow-up `inc_lock_ref` and
    /// for the next `insert` reusing this namespace).
    fn get_or_create_root(&mut self, extra_key: Option<&str>) -> NodeIdx {
        match extra_key {
            None => self.default_root,
            Some(ek) => {
                if let Some(&idx) = self.named_roots.get(ek) {
                    idx
                } else {
                    let new_root = self.tree_node_pool.alloc(TreeNode::new_root());
                    self.named_roots.insert(ek.to_owned(), new_root);
                    new_root
                }
            }
        }
    }

    /// Find the longest cached prefix of `key` in the namespace selected
    /// by `extra_key` (or default namespace if `None`). Mutates the tree
    /// when the match ends mid-node by splitting that node — same as
    /// Python's `_match_prefix_helper`. Lazily creates the namespace
    /// root for an unseen `extra_key` so the returned
    /// `last_device_node_idx` is always a valid handle into the
    /// namespace.
    ///
    /// After the walk, the validator-approved path is bumped to MRU
    /// on every configured component via the per-component
    /// `Component::bump_mru_walk` dispatch. Mirrors Python
    /// `unified_radix_cache._match_post_processor`'s
    /// `reset_node_and_parents_mru` — recency is access-order, not
    /// insertion-order.
    ///
    /// **Per-component validation** (when SWA is configured): the
    /// structural walk is unchanged, but the returned boundary
    /// (`last_device_node_idx` and the slice of `device_indices`) only
    /// advances past visited nodes whose ALL configured components'
    /// validators approve. FULL is always-true; SWA gates on
    /// contiguous-SWA-present run length reaching `sliding_window_size`
    /// (with tombstone-reset and pre-tombstone shortcut). If no
    /// validator-approved boundary is found, the result is empty and
    /// the boundary is the namespace root.
    pub fn match_prefix(
        &mut self,
        key: &[K::Atom],
        extra_key: Option<&str>,
    ) -> Result<MatchResult, RadixCacheRuntimeError> {
        let root = self.get_or_create_root(extra_key);
        let aligned_key = match PageAlignedQueryKey::new(key, self.page_size) {
            Some(k) => k,
            None => return Ok(self.empty_match_result(root)),
        };
        self.match_prefix_helper(root, aligned_key)
    }

    /// Prefix match from `root` along `key`, splitting any node where the
    /// match ends mid-key.
    fn match_prefix_helper(
        &mut self,
        root: NodeIdx,
        key: PageAlignedQueryKey<K::Atom>,
    ) -> Result<MatchResult, RadixCacheRuntimeError> {
        let key = key.as_slice();
        let key_len = key.len();
        let mut node_idx = root;
        let mut consumed = 0usize;
        let mut values: Vec<Tensor> = Vec::new();

        // The match walk could be split into 2 parts: nodes with device values,
        // and nodes with host-only values.
        let mut last_matched_node_idx = root;
        let mut last_device_node_idx = root;
        let mut last_device_value_len: usize = 0;
        let mut passed_host_only = false;

        let mut validators: Vec<Box<dyn crate::components::MatchValidator<K>>> = self
            .components
            .iter()
            .filter_map(|c| c.create_match_validator(!self.enable_hicache))
            .collect();
        let mut device_validators: Vec<Box<dyn crate::components::MatchValidator<K>>> = self
            .components
            .iter()
            .filter_map(|c| c.create_match_validator(true))
            .collect();

        while consumed < key_len {
            let remaining_key = &key[consumed..];
            // Single match-walk step: no match, full match, or split a node to get
            // a full-match node (partial match).
            let (matched_node_idx, terminated) =
                match self.tree_node_pool.match_child(node_idx, remaining_key) {
                    MatchChildResult::NotFound => break,
                    MatchChildResult::FullMatch {
                        child_idx,
                        node_key_len,
                    } => {
                        consumed += node_key_len;
                        (child_idx, false)
                    }
                    MatchChildResult::PartialMatch(node_split) => (
                        self.tree_node_pool.split_node(&self.components, node_split),
                        true,
                    ),
                };

            // Ensure the matched path splits into 2 parts: nodes in the first part
            // all have a device value; the second part only has host value.
            match self.tree_node_pool.get(matched_node_idx).value() {
                Some(v) => {
                    assert!(
                        !passed_host_only,
                        "device value present after a host-only node — matched path \
                         is not a clean device-then-host cut",
                    );
                    values.push(v.shallow_clone());
                }
                None => {
                    if !self.enable_hicache {
                        return Err(RadixCacheRuntimeError::MatchPrefixMissingFullValue {
                            node_idx: matched_node_idx,
                        });
                    }
                    match HostFullLRUSlot::value(self.tree_node_pool.get(matched_node_idx)) {
                        Some(_) => {
                            passed_host_only = true;
                        }
                        // Dead node (device + host absent) — stop. TODO(Jialin):
                        // once husks are reclaimed and write-through keeps the host
                        // prefix contiguous, a device-absent matched node is always
                        // host-backed, so host-absent here is unexpected — make it
                        // an assert.
                        None => break,
                    }
                }
            }
            node_idx = matched_node_idx;

            let matched_node = self.tree_node_pool.get(matched_node_idx);
            let mut host_valid = true;
            let mut device_valid = !passed_host_only;
            // Due to statefulness, ensure all validators run without short circuit.
            for v in validators.iter_mut() {
                host_valid &= v.validate(matched_node);
            }
            for v in device_validators.iter_mut() {
                device_valid &= v.validate(matched_node);
            }
            if host_valid {
                last_matched_node_idx = matched_node_idx;
            }
            if device_valid {
                last_device_value_len = values.len();
                last_device_node_idx = matched_node_idx;
            }
            if terminated {
                break;
            }
        }

        // Bump the chain to root in device and host LRU; skip nodes whose tier
        // value is absent.
        self.bump_mru_walk(last_matched_node_idx);

        // Extract device values up to the validator-approved boundary.
        let device_indices = if last_device_value_len == 0 {
            self.empty_tensor.shallow_clone()
        } else {
            Tensor::cat(&values[..last_device_value_len], 0)
        };

        let (mamba_branching_seqlen, mamba_value) = match self.mamba_cache_chunk_size {
            Some(chunk_size) => {
                // branching_seqlen is populated only on a partial match (walk
                // extended past the device-validator-approved boundary).
                // Unified Mamba HiCache restores the missing host Mamba state
                // before extend, so it intentionally skips the synthetic
                // branch-state fill used by L1-only Mamba.
                let branching_seqlen =
                    if !self.enable_hicache && last_device_value_len < values.len() {
                        let total: usize = values.iter().map(|v| v.size()[0] as usize).sum();
                        let aligned = total / chunk_size * chunk_size;
                        (aligned > 0).then_some(aligned)
                    } else {
                        None
                    };
                let mv = MambaLRUSlot::value(self.tree_node_pool.get(last_matched_node_idx))
                    .map(|t| t.shallow_clone());
                (branching_seqlen, mv)
            }
            None => (None, None),
        };

        let host_only_length =
            self.full_host_hit_length(last_matched_node_idx, last_device_node_idx);
        let swa_host_hit_length = self.swa_host_hit_length(last_matched_node_idx);
        let mamba_host_hit_length = self.mamba_host_hit_length(last_matched_node_idx);
        let last_host_node_idx = if self.enable_hicache {
            last_matched_node_idx
        } else {
            last_device_node_idx
        };

        Ok(MatchResult {
            device_indices,
            last_device_node_idx,
            last_host_node_idx,
            best_match_node_idx: last_matched_node_idx,
            host_only_length,
            swa_host_hit_length,
            mamba_host_hit_length,
            mamba_branching_seqlen,
            mamba_value,
        })
    }

    /// Insert `(key, value)` into the namespace selected by `extra_key`, or
    /// the default namespace if `None`. `value` is a 1-D `Int64` tensor of
    /// KV slot indices whose length must be at least the page-aligned key
    /// length; excess length is silently truncated (symmetric with the
    /// page-aligned key truncation). Validation happens before any
    /// structural tree mutation, so a bad-input call never leaves a
    /// half-modified tree (or a stranded namespace root for an unseen
    /// `extra_key`). The cache deep-copies the slice it stores, so callers
    /// may freely mutate or drop their tensor after this call returns.
    ///
    /// Returns the prefix length already in the tree before this insert —
    /// the caller (e.g. scheduler) frees `value[:prefix_len]` since those
    /// slots are now redundant duplicates of the already-cached prefix.
    pub fn insert(
        &mut self,
        key: &[K::Atom],
        value: &Tensor,
        extra_key: Option<&str>,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        mamba_value: Option<Tensor>,
    ) -> Result<InsertResult, RadixCacheRuntimeError> {
        self.validate_insert_value(key.len(), value, mamba_value.as_ref())?;
        let aligned_key = match PageAlignedQueryKey::new(key, self.page_size) {
            Some(k) => k,
            None => {
                return Ok(InsertResult {
                    prefix_len: 0,
                    leaf_creation_skipped: false,
                    // Empty key: ownership NOT taken; caller should free mamba_value.
                    mamba_value_exists: true,
                    deferred_actions: Vec::new(),
                });
            }
        };
        let root = self.get_or_create_root(extra_key);
        self.insert_helper(
            root,
            aligned_key,
            value,
            prev_prefix_len,
            swa_evicted_seqlen,
            mamba_value,
        )
    }

    /// Validate insert inputs.
    fn validate_insert_value(
        &self,
        key_len: usize,
        value: &Tensor,
        mamba_value: Option<&Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if value.kind() != Kind::Int64 {
            return Err(RadixCacheRuntimeError::InsertValueWrongDtype { got: value.kind() });
        }
        let shape = value.size();
        if shape.len() != 1 {
            return Err(RadixCacheRuntimeError::InsertValueWrongShape { got: shape });
        }
        let value_len = shape[0] as usize;
        let ps = self.page_size.get();
        let aligned_key_len = key_len / ps * ps;
        if value_len < aligned_key_len {
            return Err(RadixCacheRuntimeError::InsertValueLengthMismatch {
                aligned_key_len,
                value_len,
            });
        }
        let cache_device = self.empty_tensor.device();
        let value_device = value.device();
        if value_device != cache_device {
            return Err(RadixCacheRuntimeError::InsertValueWrongDevice {
                expected: cache_device,
                got: value_device,
            });
        }
        if let Some(m) = mamba_value {
            if self.mamba_cache_chunk_size.is_none() {
                return Err(RadixCacheRuntimeError::InsertMambaValueWithoutMambaConfigured);
            }
            if m.kind() != Kind::Int64 {
                return Err(RadixCacheRuntimeError::InsertMambaValueWrongDtype { got: m.kind() });
            }
            let mshape = m.size();
            if mshape.len() != 1 || mshape[0] != 1 {
                return Err(RadixCacheRuntimeError::InsertMambaValueWrongShape { got: mshape });
            }
            let mamba_device = m.device();
            if mamba_device != cache_device {
                return Err(RadixCacheRuntimeError::InsertMambaValueWrongDevice {
                    expected: cache_device,
                    got: mamba_device,
                });
            }
        }
        Ok(())
    }

    /// Per-component LRU recency bump dispatcher. Iterates
    /// `self.components` and calls each one's `bump_mru_walk` to
    /// bring `node_idx` and its ancestors to MRU position in the
    /// component's LRU. Mirrors OSS
    /// `unified_radix_cache._for_each_component_lru` /
    /// `_match_post_processor`'s per-component `reset_node_and_parents_mru`.
    ///
    /// Called by `match_prefix_helper` with the validator-approved
    /// best node (only the matched subset is bumped) and by
    /// `insert_helper` with the deepest touched node (the new leaf or
    /// last overlap node). SWA tombstones are skipped at the slot
    /// level via `in_list` gating.
    fn bump_mru_walk(&mut self, node_idx: NodeIdx) {
        for comp in self.components.iter() {
            comp.bump_mru_walk(&mut self.tree_node_pool, node_idx);
        }
    }

    /// Per-overlap-node component dispatcher. Iterates `self.components`
    /// and takes `min(consumed_from)` across all returned values. Mirrors
    /// OSS `unified_radix_cache._insert_helper`'s
    /// `consumed_from = min(consumed_from, comp_consumed_from)` pattern
    /// — a single component claiming any slot vetoes its freeing as a
    /// duplicate. Default returns `node_key_len` (claim nothing) when no
    /// component claims; FULL inherits the default and SWA overrides
    /// for tombstone recovery.
    #[allow(clippy::too_many_arguments)]
    fn consume_value(
        &mut self,
        child_idx: NodeIdx,
        node_key_len: usize,
        total_prefix_len: usize,
        prev_prefix_len: usize,
        value_slice: &Tensor,
        swa_evicted_seqlen: usize,
        deferred: &mut Vec<DeferredAction>,
    ) -> Result<usize, RadixCacheRuntimeError> {
        let mut consumed_from = node_key_len;
        let full_had_value_at_entry = FullLRUSlot::has_value(self.tree_node_pool.get(child_idx));
        for comp in self.components.iter() {
            let comp_consumed = comp.consume_value(
                &mut self.tree_node_pool,
                &self.components,
                child_idx,
                node_key_len,
                total_prefix_len,
                prev_prefix_len,
                value_slice,
                swa_evicted_seqlen,
                full_had_value_at_entry,
                deferred,
            )?;
            consumed_from = consumed_from.min(comp_consumed);
        }
        Ok(consumed_from)
    }

    /// Pre-leaf-creation veto dispatcher. Iterates `self.components` and
    /// returns `true` if ANY component vetoes leaf creation. Mirrors OSS
    /// `unified_radix_cache._insert_helper`'s
    /// `any(comp.should_skip_leaf_creation(...))` pattern. Default
    /// returns `false` (no veto). FULL inherits the default; SWA
    /// overrides to skip when the entire suffix is outside the SWA
    /// window.
    fn should_skip_leaf_creation(
        &self,
        total_prefix_len: usize,
        key_len: usize,
        swa_evicted_seqlen: usize,
    ) -> bool {
        self.components
            .iter()
            .any(|c| c.should_skip_leaf_creation(total_prefix_len, key_len, swa_evicted_seqlen))
    }

    /// Post-leaf-creation hook dispatcher. Iterates `self.components`
    /// in order, letting each one inspect / split the new leaf and
    /// emit deferred actions. Mirrors OSS
    /// `unified_radix_cache._insert_helper`'s
    /// `for comp in components: comp.commit_insert_component_data(...)`
    /// loop. Default is a no-op. FULL inherits the default; SWA splits
    /// at the SWA boundary (if straddling) and emits `SwaStamp` for the
    /// in-window portion.
    fn commit_insert_data_on_new_leaf(
        &mut self,
        leaf_idx: NodeIdx,
        consumed: usize,
        swa_evicted_seqlen: usize,
        deferred: &mut Vec<DeferredAction>,
    ) {
        for comp in self.components.iter() {
            comp.commit_insert_data_on_new_leaf(
                &mut self.tree_node_pool,
                &self.components,
                leaf_idx,
                consumed,
                swa_evicted_seqlen,
                deferred,
            );
        }
    }

    /// HiCache write-through threshold: a hot node (hit count >= this) has
    /// its device value backed up to host.
    const WRITE_THROUGH_THRESHOLD: u64 = 1;

    /// Bump the node's hit count; once it reaches the write-through
    /// threshold, emit a backup of the device value to host.
    fn inc_hit_count(&mut self, node_idx: NodeIdx, deferred: &mut Vec<DeferredAction>) {
        if !self.enable_hicache {
            return;
        }
        // Write-back is triggered by eviction, not a hit-count threshold, so no
        // hit-count tracking is needed.
        if self.hicache_write_back {
            return;
        }
        let node = self.tree_node_pool.get_mut(node_idx);
        node.hit_count = node.hit_count.saturating_add(1);
        if HostFullLRUSlot::has_value(node) || node.hit_count < Self::WRITE_THROUGH_THRESHOLD {
            return;
        }
        if let Some(value) = FullLRUSlot::value(node) {
            let value = value.shallow_clone();
            deferred.push(DeferredAction::FullWriteThroughBackup { node_idx, value });
        }
    }

    /// Walk from `root` along `key`, splitting nodes on partial matches,
    /// then append a new leaf for the unmatched suffix.
    ///
    /// When `swa_evicted_seqlen` is `Some`, the walk also runs SWA
    /// component hooks at three points (mirroring OSS
    /// `unified_radix_cache._insert_helper` + `SWAComponent`):
    ///
    /// ```text
    ///   insert key layout (seqlen = swa_evicted_seqlen):
    ///
    ///   0              consumed                              key_len
    ///   |              |                                     |
    ///   v              v                                     v
    ///   [== overlap ==][============ new suffix =============]
    ///                       seqlen may fall anywhere:
    ///                       ← in overlap | in suffix | past end →
    ///
    ///   ┌────────────────────────────────────────────────────────────────┐
    ///   │                   insert_helper walk                           │
    ///   │                                                                │
    ///   │  for each overlap node:                                        │
    ///   │    ├─ FULL: emit FullDupFreed (free duplicate indices)         │
    ///   │    └─ SWA:  consume_value_on_full_match (tombstone recovery)   │
    ///   │             → compares seqlen vs node position                 │
    ///   │             → may emit SwaRecover, may split node              │
    ///   │                                                                │
    ///   │  before leaf creation (remaining = key_len - consumed):        │
    ///   │    └─ SWA:  should_skip_leaf_creation (veto check)             │
    ///   │             → skip if seqlen >= key_len                        │
    ///   │               (entire new suffix outside SWA window)           │
    ///   │                                                                │
    ///   │  after leaf creation:                                          │
    ///   │    └─ SWA:  commit_insert_data_on_new_leaf (boundary split)    │
    ///   │             → compares seqlen vs leaf position                 │
    ///   │             → may split leaf at seqlen into tombstone + child  │
    ///   │               and emit SwaStamp for the in-window child        │
    ///   └────────────────────────────────────────────────────────────────┘
    /// ```
    ///
    /// The FULL component's hooks are all defaults (free duplicates,
    /// never veto, no-op commit) — identical to today's FULL-only path.
    /// When `swa_evicted_seqlen` is `None`, no SWA hooks fire and the
    /// walk is identical to the pre-SWA implementation.
    fn insert_helper(
        &mut self,
        root: NodeIdx,
        key: PageAlignedQueryKey<K::Atom>,
        value: &Tensor,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        mamba_value: Option<Tensor>,
    ) -> Result<InsertResult, RadixCacheRuntimeError> {
        let key = key.as_slice();
        let key_len = key.len();
        let mut node_idx = root;
        let mut consumed = 0usize;
        let mut deferred: Vec<DeferredAction> = Vec::new();

        // ── Overlap walk ──
        //
        // Walk the tree to match the longest page-aligned prefix of the insert
        // key, one edge per step. Within each step:
        // - No match: create leaf node
        // - Full match: skip and continue walking down the tree
        // - Partial match: split node and create leaf node
        while consumed < key_len {
            let remaining_key = &key[consumed..];
            let (current_node_idx, step_len, last_step) =
                match self.tree_node_pool.match_child(node_idx, remaining_key) {
                    MatchChildResult::NotFound => break,
                    MatchChildResult::FullMatch {
                        child_idx,
                        node_key_len,
                    } => (child_idx, node_key_len, false),
                    MatchChildResult::PartialMatch(node_split) => {
                        let split_len = node_split.split_len();
                        let new_idx = self.tree_node_pool.split_node(&self.components, node_split);
                        (new_idx, split_len, true)
                    }
                };

            let value_slice = value.narrow(0, consumed as i64, step_len as i64);

            let consumed_from = self.consume_value(
                current_node_idx,
                step_len,
                consumed,
                prev_prefix_len,
                &value_slice,
                swa_evicted_seqlen,
                &mut deferred,
            )?;

            // Free the duplicate band of this segment: value indices that are
            // past the caller's locked prefix but not claimed by any component.
            //
            // Walk Indices:  consumed        prev_prefix_len                     consumed+step_len
            // Step Indices:  0               dup_start       consumed_from       step_len
            //  (Shifted)     |               |               |                   |
            //                v               v               v                   v
            //                +---------------+---------------+-------------------+
            //                | locked prefix | duplicate &   | claimed by a      |
            //                | (keep; caller | unlocked:     | component         |
            //                | frees it)     | >>> FREE <<<  | (keep)            |
            //                +---------------+---------------+-------------------+
            //
            // Within a step the value splits into 3 parts:
            // - locked by a shared prefix (kept; the caller frees it)
            // - recomputed but already populated -> duplicate, freed here
            // - recomputed and claimed by a component (e.g. SWA recovery; kept)
            let dup_start = prev_prefix_len.saturating_sub(consumed);
            if dup_start < consumed_from {
                deferred.push(DeferredAction::FullDupFreed {
                    freed_indices: value_slice.narrow(
                        0,
                        dup_start as i64,
                        (consumed_from - dup_start) as i64,
                    ),
                });
            }

            // Bump reuse on the extended node; back up to host at threshold.
            self.inc_hit_count(current_node_idx, &mut deferred);

            consumed += step_len;
            node_idx = current_node_idx;
            if last_step {
                break;
            }
        }

        // ── Leaf creation ──
        let remaining_len = key_len - consumed;
        let mut leaf_creation_skipped = false;
        let mut new_leaf_created = false;

        if remaining_len > 0 {
            let skip = self.should_skip_leaf_creation(consumed, remaining_len, swa_evicted_seqlen);

            if skip {
                deferred.push(DeferredAction::FullDupFreed {
                    freed_indices: value.narrow(0, consumed as i64, remaining_len as i64),
                });
                leaf_creation_skipped = true;
            } else {
                let remaining_key = key[consumed..].to_vec();
                let remaining_value = value
                    .narrow(0, consumed as i64, remaining_len as i64)
                    .copy();
                let leaf = TreeNode::new_child(remaining_key, node_idx, Some(remaining_value));
                #[allow(clippy::expect_used, reason = "child key just confirmed absent above")]
                let leaf_idx = self.tree_node_pool.insert_leaf(node_idx, leaf).expect(
                    "first-page child key was just confirmed absent \
                         at this parent",
                );

                self.commit_insert_data_on_new_leaf(
                    leaf_idx,
                    consumed,
                    swa_evicted_seqlen,
                    &mut deferred,
                );

                // Bump hit on the new leaf; back up to host if needed
                self.inc_hit_count(leaf_idx, &mut deferred);

                node_idx = leaf_idx;
                new_leaf_created = true;
            }
        }

        // ── Populate mamba value ──
        let mamba_value_exists = if let Some(mv) = mamba_value {
            assert!(
                !leaf_creation_skipped,
                "leaf_creation_skipped is unreachable for Mamba",
            );
            if new_leaf_created || !MambaLRUSlot::has_value(self.tree_node_pool.get(node_idx)) {
                MambaLRUSlot::set_value(&mut self.tree_node_pool, node_idx, mv)?;
                let delta = MambaLRUSlot::value_len(self.tree_node_pool.get(node_idx));
                MambaLRUSlot::bump_mru(&mut self.tree_node_pool, node_idx);
                MambaLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += delta;
                false
            } else {
                true
            }
        } else {
            false
        };

        // Bump the chain to root in device and host LRU; skip nodes whose tier
        // value is absent.
        self.bump_mru_walk(node_idx);

        Ok(InsertResult {
            prefix_len: consumed,
            leaf_creation_skipped,
            mamba_value_exists,
            deferred_actions: deferred,
        })
    }

    /// Configured page size (1 for token, >1 for page).
    pub fn page_size(&self) -> usize {
        self.page_size.get()
    }

    /// Number of live nodes in the underlying tree_node_pool (always >= 1 — root).
    pub fn active_tree_node_count(&self) -> usize {
        self.tree_node_pool.active_node_count()
    }

    // TODO: rename `evictable_token_size` → `full_evictable_token_size`,
    // `protected_token_size` → `full_protected_token_size`, and
    // `total_token_size` → `full_total_token_size` to match the
    // `swa_*` accessors below. The current asymmetry (FULL has no
    // prefix; SWA has the `swa_` prefix) reads as if FULL is the
    // "default" component and the others are exceptions, when
    // really both are first-class. After the rename a future Mamba
    // accessor lands as `mamba_*` symmetrically. Deferred — touches
    // `RustPageRadixCacheWrapper` PyO3 surface, the `__init__.pyi`
    // stubs, the orchestrator's `RustUnifiedRadixCache`,
    // `RustFullComponent.unlocked_size`, and every test call site
    // that reads these aggregates.

    /// Sum of `key.len()` across FULL device-value unreferenced nodes.
    pub fn evictable_token_size(&self) -> usize {
        FullLRUSlot::unlocked_size(&self.tree_node_pool)
    }

    /// Sum of `key.len()` across FULL device-value referenced nodes.
    pub fn protected_token_size(&self) -> usize {
        FullLRUSlot::locked_size(&self.tree_node_pool)
    }

    /// Total tokens (evictable + protected) across FULL and SWA components.
    pub fn total_token_size(&self) -> usize {
        let mut total = FullLRUSlot::total_size(&self.tree_node_pool);
        if self.has_swa_component {
            total += SwaLRUSlot::total_size(&self.tree_node_pool);
        }
        total
    }

    /// Total Mamba slots (evictable + protected); separate from `total_token_size` because Mamba's unit is slots, not tokens.
    pub fn mamba_total_size(&self) -> usize {
        if self.has_mamba_component {
            MambaLRUSlot::total_size(&self.tree_node_pool)
        } else {
            0
        }
    }

    /// Sum of `key.len()` across SWA device-value unreferenced nodes.
    pub fn swa_evictable_token_size(&self) -> usize {
        SwaLRUSlot::unlocked_size(&self.tree_node_pool)
    }

    /// Sum of `key.len()` across SWA device-value referenced nodes.
    pub fn swa_protected_token_size(&self) -> usize {
        SwaLRUSlot::locked_size(&self.tree_node_pool)
    }

    /// Count of unlocked nodes with a Mamba value populated.
    pub fn mamba_evictable_token_size(&self) -> usize {
        MambaLRUSlot::unlocked_size(&self.tree_node_pool)
    }

    /// Count of locked nodes with a Mamba value populated.
    pub fn mamba_protected_token_size(&self) -> usize {
        MambaLRUSlot::locked_size(&self.tree_node_pool)
    }

    fn recompute_device_slot_size<S: LRUSlot>(&self) -> (usize, usize) {
        let mut unlocked = 0usize;
        let mut locked = 0usize;
        for idx in self.tree_node_pool.live_node_indices() {
            let node = self.tree_node_pool.get(idx);
            if !S::has_value(node) {
                continue;
            }
            let len = S::value_len(node);
            if S::lock_ref(node) == 0 {
                unlocked += len;
            } else {
                locked += len;
            }
        }
        (unlocked, locked)
    }

    fn check_device_slot_size<S: LRUSlot>(&self, name: &str, failures: &mut Vec<String>) {
        let (actual_unlocked, actual_locked) = self.recompute_device_slot_size::<S>();
        let counter_unlocked = S::unlocked_size(&self.tree_node_pool);
        let counter_locked = S::locked_size(&self.tree_node_pool);
        if actual_unlocked != counter_unlocked || actual_locked != counter_locked {
            failures.push(format!(
                "{name}: actual unlocked/locked={actual_unlocked}/{actual_locked}, \
                 counters={counter_unlocked}/{counter_locked}"
            ));
        }
    }

    pub fn sanity_check(&self) -> Result<(), RadixCacheRuntimeError> {
        let mut failures = Vec::new();
        self.check_device_slot_size::<FullLRUSlot>("FullDevice", &mut failures);
        if self.has_swa_component {
            self.check_device_slot_size::<SwaLRUSlot>("SwaDevice", &mut failures);
        }
        if self.has_mamba_component {
            self.check_device_slot_size::<MambaLRUSlot>("MambaDevice", &mut failures);
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(RadixCacheRuntimeError::SanityCheckFailed {
                message: failures.join("; "),
            })
        }
    }

    /// Pure-dispatcher acquire. Iterates `self.components` forward
    /// (FULL first, then SWA — order set by `build_components` at
    /// construction, matching OSS `unified_radix_cache.py`'s
    /// `inc_lock_ref`) and aggregates each component's
    /// `IncLockRefResult` into a single result:
    ///   * `delta` — sum of per-component contributions.
    ///   * `swa_uuid_for_lock` — taken from the unique component that
    ///     produces it (SWA today; `None` for FULL-only configs).
    ///
    /// The actual walks live in `FullComponent::inc_lock_ref` /
    /// `SwaComponent::inc_lock_ref` — see those for per-component
    /// semantics. Forward iteration order is required so FULL's
    /// `lock_ref` is bumped before SWA's per-slot
    /// `SwaLRUSlot::inc_lock_ref` mutator-assert checks
    /// `prospective new swa_lock_ref <= full_lock_ref` on the same node.
    ///
    /// PyO3 surface returns this as `(delta, swa_uuid_for_lock)`
    /// (tuple-shaped — no PyO3 class for a 2-field POD). Caller must
    /// pass `swa_uuid_for_lock` back to `dec_lock_ref` so SWA's
    /// release stops at the right boundary.
    // TODO(perf): collapse the per-component walks into a single
    // coordinated leaf-to-root walk. Today each component walks
    // independently — FULL: leaf → root excl.; SWA: leaf → window-fill
    // boundary — so a SWA-configured cache visits the leaf-to-boundary
    // segment twice (once per walk). OSS `swa_radix_cache.py`'s
    // `inc_lock_ref` does it in one pass (FULL bumps always, SWA
    // bumps conditionally on window-not-yet-filled). Matching that
    // here would require either (a) a new "coordinated per-node" hook
    // on the Component trait that each component implements with a
    // "should keep walking" return, preserving the trait abstraction;
    // or (b) inlining per-component per-node logic directly in this
    // dispatcher, simpler but couples the dispatcher to component
    // internals. Defer until profiling flags the per-component walk
    // as hot.
    pub fn inc_lock_ref(&mut self, node_idx: NodeIdx) -> IncLockRefResult {
        let mut delta: i64 = 0;
        let mut swa_uuid_for_lock: Option<u64> = None;
        for c in self.components.iter() {
            if let Some(r) = c.inc_lock_ref(&mut self.tree_node_pool, node_idx) {
                delta += r.delta;
                // Per-field merge: only SWA produces `swa_uuid_for_lock`
                // today (FULL and future components return `None` per
                // Component trait contract). `.or()` keeps the first
                // `Some` observed across the iteration — order-
                // independent under the at-most-one contract, so future
                // component additions can't accidentally overwrite SWA's
                // value with their `None`.
                swa_uuid_for_lock = swa_uuid_for_lock.or(r.swa_uuid_for_lock);
            }
        }
        IncLockRefResult {
            delta,
            swa_uuid_for_lock,
        }
    }

    /// Pure-dispatcher release. Iterates `self.components` in
    /// **reverse** (SWA first, then FULL) and sums each component's
    /// `delta` contribution. `swa_uuid_for_lock` is threaded to every
    /// component (FULL ignores; SWA matches against the boundary
    /// node's stamp to know when to stop). Pass back the value
    /// returned by the matching `inc_lock_ref`.
    ///
    /// Reverse iteration is a Rust-side deviation from OSS
    /// `unified_radix_cache.py`'s `dec_lock_ref` forward iteration.
    /// The reason:
    /// our per-slot `FullLRUSlot::dec_lock_ref` mutator-assert checks
    /// `swa_lock_ref <= prospective new full_lock_ref`. With forward
    /// iter (FULL dec'd first), if `full_lock_ref == swa_lock_ref`
    /// before the call, FULL's dec produces `full = swa - 1` which
    /// fires the assert. SWA-first dec keeps the gap valid through
    /// every step. OSS doesn't have these per-slot asserts (its
    /// invariant is enforced by a single coordinated walk in
    /// `swa_radix_cache.py`), so OSS works in either order; we picked
    /// reverse-iter to keep the assert load-bearing as a Rust-side
    /// regression net WITHOUT paying for an extra path-walk to verify
    /// the invariant post-dispatch.
    ///
    /// Panics on lock_ref underflow on any node in the walk — callers
    /// must match dec calls to inc calls exactly (per-slot
    /// mutator-asserts catch underflow at the actual mutation site).
    // TODO(perf): same single-walk optimization opportunity as
    // `inc_lock_ref` (see TODO there) — today FULL and SWA dec walks
    // are independent and the leaf-to-boundary segment is visited
    // twice.
    pub fn dec_lock_ref(&mut self, node_idx: NodeIdx, swa_uuid_for_lock: Option<u64>) -> i64 {
        let mut delta: i64 = 0;
        for c in self.components.iter().rev() {
            if let Some(d) = c.dec_lock_ref(&mut self.tree_node_pool, node_idx, swa_uuid_for_lock) {
                delta += d;
            }
        }
        delta
    }

    fn inc_full_lock_ref(&mut self, node_idx: NodeIdx) -> i64 {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = self.tree_node_pool.get(current).parent() {
            delta += FullLRUSlot::inc_lock_ref(&mut self.tree_node_pool, current);
            current = parent;
        }
        delta
    }

    fn dec_full_lock_ref(&mut self, node_idx: NodeIdx) -> i64 {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = self.tree_node_pool.get(current).parent() {
            delta += FullLRUSlot::dec_lock_ref(&mut self.tree_node_pool, current);
            current = parent;
        }
        delta
    }

    /// Lock the node-local device values used as D2H backup sources.
    pub fn inc_backup_lock_ref(&mut self, node_idx: NodeIdx) {
        FullLRUSlot::inc_lock_ref(&mut self.tree_node_pool, node_idx);
        if SwaLRUSlot::has_value(self.tree_node_pool.get(node_idx)) {
            SwaLRUSlot::inc_lock_ref(&mut self.tree_node_pool, node_idx);
        }
        if MambaLRUSlot::has_value(self.tree_node_pool.get(node_idx)) {
            MambaLRUSlot::inc_lock_ref(&mut self.tree_node_pool, node_idx);
        }
    }

    /// Release `inc_backup_lock_ref`.
    pub fn dec_backup_lock_ref(&mut self, node_idx: NodeIdx) {
        if MambaLRUSlot::lock_ref(self.tree_node_pool.get(node_idx)) > 0 {
            MambaLRUSlot::dec_lock_ref(&mut self.tree_node_pool, node_idx);
        }
        if SwaLRUSlot::lock_ref(self.tree_node_pool.get(node_idx)) > 0 {
            SwaLRUSlot::dec_lock_ref(&mut self.tree_node_pool, node_idx);
        }
        FullLRUSlot::dec_lock_ref(&mut self.tree_node_pool, node_idx);
    }

    /// Best-effort to evict at least `num_tokens` per component.
    pub fn evict(&mut self, request: EvictRequest) -> EvictResult {
        let mut result = EvictResult::default();
        // FULL eviction runs first: it can evict SWA values too, shrinking
        // SWA's residual budget.
        for c in self.components.iter() {
            c.evict(&mut self.tree_node_pool, &request, &mut result);
        }
        result
    }

    /// Best-effort to evict at least `num_tokens` host FULL values.
    pub fn evict_host(&mut self, num_tokens: usize) -> EvictResult {
        let mut result = EvictResult::default();
        evict_host_full(&mut self.tree_node_pool, num_tokens, &mut result);
        result
    }

    /// Best-effort to evict at least `num_tokens` host SWA values.
    pub fn evict_host_swa(&mut self, num_tokens: usize) -> EvictResult {
        let mut result = EvictResult::default();
        evict_host_non_full::<K, HostSwaLRUSlot>(&mut self.tree_node_pool, num_tokens, &mut result);
        result
    }

    /// Best-effort to evict at least `num_tokens` host Mamba values.
    pub fn evict_host_mamba(&mut self, num_tokens: usize) -> EvictResult {
        let mut result = EvictResult::default();
        evict_host_non_full::<K, HostMambaLRUSlot>(
            &mut self.tree_node_pool,
            num_tokens,
            &mut result,
        );
        result
    }

    pub fn get_full_device_value(&self, node_idx: NodeIdx) -> Option<Tensor> {
        FullLRUSlot::value(self.tree_node_pool.get(node_idx)).map(|v| v.shallow_clone())
    }

    pub fn get_swa_device_value(&self, node_idx: NodeIdx) -> Option<Tensor> {
        SwaLRUSlot::value(self.tree_node_pool.get(node_idx)).map(|v| v.shallow_clone())
    }

    pub fn get_mamba_device_value(&self, node_idx: NodeIdx) -> Option<Tensor> {
        MambaLRUSlot::value(self.tree_node_pool.get(node_idx)).map(|v| v.shallow_clone())
    }

    /// Walk the host-only (device-evicted, host-backed) chain from `node_idx`
    /// up to the first device-present node. Also lock the chain to avoid eviction.
    pub fn prepare_load_back(
        &mut self,
        node_idx: NodeIdx,
    ) -> Result<PrepareLoadBackResult, RadixCacheRuntimeError> {
        let mut chain: Vec<NodeIdx> = Vec::new();
        let mut host_values: Vec<Tensor> = Vec::new();
        let mut cur_node_idx = node_idx;
        loop {
            let node = self.tree_node_pool.get(cur_node_idx);
            // Stop at root or device-present node.
            let Some(parent) = node.parent() else {
                break;
            };
            if FullLRUSlot::value(node).is_some() {
                break;
            }
            // A device-evicted non-root chain node must be host-backed; absence
            // means host-prefix contiguity is broken (state corruption).
            let host_value = match HostFullLRUSlot::value(node) {
                Some(v) => v.shallow_clone(),
                None => {
                    return Err(RadixCacheRuntimeError::PrepareLoadBackMissingHostValue {
                        node_idx: cur_node_idx,
                    });
                }
            };
            chain.push(cur_node_idx);
            host_values.push(host_value);
            cur_node_idx = parent;
        }
        let ancestor_node_idx = cur_node_idx;
        // root->leaf order to prioritize loadback from root side.
        chain.reverse();
        host_values.reverse();
        // Lock ancestor node and chain host values to avoid eviction.
        for &idx in &chain {
            HostFullLRUSlot::inc_lock_ref(&mut self.tree_node_pool, idx);
        }
        self.inc_full_lock_ref(ancestor_node_idx);

        let host_indices = if host_values.is_empty() {
            self.empty_tensor.shallow_clone()
        } else {
            Tensor::cat(&host_values, 0)
        };
        Ok(PrepareLoadBackResult {
            chain,
            host_indices,
            ancestor_node_idx,
        })
    }

    /// Build aux host load-back transfers and lock their host source values.
    pub fn prepare_aux_load_back(&mut self, node_idx: NodeIdx) -> PrepareAuxLoadBackResult {
        let (swa_chain, swa_values) = self.prepare_swa_load_back(node_idx);
        let (mamba_chain, mamba_values) = self.prepare_mamba_load_back(node_idx);
        PrepareAuxLoadBackResult {
            swa_chain,
            swa_host_indices: cat_or_empty(&self.empty_tensor, swa_values),
            mamba_chain,
            mamba_host_indices: cat_or_empty(&self.empty_tensor, mamba_values),
        }
    }

    /// Collect device FULL values on the path `(stop_node_idx, node_idx]`.
    ///
    /// This mirrors Python UnifiedRadixCache's load-back post-processor: an
    /// auxiliary-only load-back (SWA/Mamba host state with all FULL KV already
    /// on device) still advances the request's usable prefix. The returned
    /// tensor is root-to-leaf ordered and excludes `stop_node_idx`, which is
    /// the request's old `last_device_node`.
    pub fn collect_full_device_values_between(
        &self,
        node_idx: NodeIdx,
        stop_node_idx: NodeIdx,
    ) -> Result<Tensor, RadixCacheRuntimeError> {
        if node_idx == stop_node_idx {
            return Ok(self.empty_tensor.shallow_clone());
        }

        let mut values: Vec<Tensor> = Vec::new();
        let mut cur = node_idx;
        loop {
            if cur == stop_node_idx {
                break;
            }
            let node = self.tree_node_pool.get(cur);
            let value = FullLRUSlot::value(node).ok_or(
                RadixCacheRuntimeError::CollectFullDeviceValuesMissingValue { node_idx: cur },
            )?;
            values.push(value.shallow_clone());
            let Some(parent) = node.parent() else {
                return Err(
                    RadixCacheRuntimeError::CollectFullDeviceValuesStopNotAncestor {
                        node_idx,
                        stop_node_idx,
                    },
                );
            };
            cur = parent;
        }

        values.reverse();
        Ok(cat_or_empty(&self.empty_tensor, values))
    }

    fn prepare_swa_load_back(&mut self, node_idx: NodeIdx) -> (Vec<NodeIdx>, Vec<Tensor>) {
        let Some(window) = self.sliding_window_size else {
            return (Vec::new(), Vec::new());
        };
        let mut seen = 0usize;
        let mut chain: Vec<NodeIdx> = Vec::new();
        let mut host_values: Vec<Tensor> = Vec::new();
        let mut cur = node_idx;
        while seen < window {
            let node = self.tree_node_pool.get(cur);
            if node.is_root() {
                break;
            }
            if SwaLRUSlot::has_value(node) {
                seen += SwaLRUSlot::value_len(node);
            } else if let Some(host_value) = HostSwaLRUSlot::value(node) {
                seen += host_value.size()[0] as usize;
                chain.push(cur);
                host_values.push(host_value.shallow_clone());
            } else {
                break;
            }
            let Some(parent) = self.tree_node_pool.get(cur).parent() else {
                break;
            };
            cur = parent;
        }
        chain.reverse();
        host_values.reverse();
        for &idx in &chain {
            HostSwaLRUSlot::inc_lock_ref(&mut self.tree_node_pool, idx);
        }
        (chain, host_values)
    }

    fn prepare_mamba_load_back(&mut self, node_idx: NodeIdx) -> (Vec<NodeIdx>, Vec<Tensor>) {
        if !self.has_mamba_component {
            return (Vec::new(), Vec::new());
        }
        let node = self.tree_node_pool.get(node_idx);
        if MambaLRUSlot::has_value(node) {
            return (Vec::new(), Vec::new());
        }
        let Some(host_value) = HostMambaLRUSlot::value(node) else {
            return (Vec::new(), Vec::new());
        };
        let value = host_value.shallow_clone();
        HostMambaLRUSlot::inc_lock_ref(&mut self.tree_node_pool, node_idx);
        (vec![node_idx], vec![value])
    }

    /// Write per-node SWA values back into the tree.
    pub fn apply_swa_writes(
        &mut self,
        node_indices: Vec<NodeIdx>,
        swa_values: Vec<tch::Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if node_indices.len() != swa_values.len() {
            return Err(RadixCacheRuntimeError::ApplySwaWritesMismatch {
                indices: node_indices.len(),
                values: swa_values.len(),
            });
        }
        // Accumulate credit across all stamped nodes; commit to the
        // pool-state aggregate once after the loop. Saves N-1
        // pool_state_mut indexes for an N-element batch.
        let mut evictable_size_credit: usize = 0;
        for (idx, value) in node_indices.into_iter().zip(swa_values) {
            // Snapshot pre-mutation node state once (immutable borrow
            // released before the value-stamp's mutable borrow). The
            // stamp doesn't touch LRU membership or `key`, so these
            // snapshots stay valid for the credit decision below.
            let node = self.tree_node_pool.get(idx);
            let in_list_at_entry = SwaLRUSlot::data(node).in_list;
            let value_present_at_entry = SwaLRUSlot::has_value(node);
            let value_len = value.size()[0] as usize;
            // SWA value existence should be in sync with whether the node
            // is in the SWA LRU list.
            assert_eq!(
                in_list_at_entry, value_present_at_entry,
                "SWA invariant violated at apply_swa_writes entry for node_idx {idx}: \
                 in_list ({in_list_at_entry}) != value.is_some() ({value_present_at_entry})",
            );

            // Stamp the SWA value.
            SwaLRUSlot::replace_value(&mut self.tree_node_pool, idx, value);

            SwaLRUSlot::bump_mru(&mut self.tree_node_pool, idx);
            if !in_list_at_entry {
                evictable_size_credit += value_len;
            }
        }
        SwaLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += evictable_size_credit;
        Ok(())
    }

    /// Populate backup host values onto the given nodes.
    pub fn set_host_full_values(
        &mut self,
        node_indices: Vec<NodeIdx>,
        host_values: Vec<tch::Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if node_indices.len() != host_values.len() {
            return Err(RadixCacheRuntimeError::SetHostFullValuesMismatch {
                indices: node_indices.len(),
                values: host_values.len(),
            });
        }
        for (idx, value) in node_indices.into_iter().zip(host_values) {
            let node = self.tree_node_pool.get(idx);
            // Already backed up (host value present) → caller bug; restamping
            // would leak the prior host KV.
            if HostFullLRUSlot::has_value(node) {
                return Err(RadixCacheRuntimeError::HostValueAlreadyBackedUp { node_idx: idx });
            }
            if !self.hicache_write_back {
                // Contiguity: parent must be root or host-backed (write-through
                // grows the host set root-down).
                if let Some(parent_idx) = node.parent() {
                    let parent = self.tree_node_pool.get(parent_idx);
                    if !parent.is_root() && !HostFullLRUSlot::has_value(parent) {
                        return Err(RadixCacheRuntimeError::HostBackupParentNotBackedUp {
                            node_idx: idx,
                            parent_idx,
                        });
                    }
                }
            }
            let value_len = value.size()[0] as usize;
            // Update host value
            HostFullLRUSlot::set_value(&mut self.tree_node_pool, idx, value)?;
            // Update LRU
            HostFullLRUSlot::bump_mru(&mut self.tree_node_pool, idx);
            HostFullLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += value_len;
        }
        Ok(())
    }

    pub fn set_host_swa_values(
        &mut self,
        node_indices: Vec<NodeIdx>,
        host_values: Vec<tch::Tensor>,
    ) -> Result<Vec<tch::Tensor>, RadixCacheRuntimeError> {
        self.set_host_aux_values::<HostSwaLRUSlot, SwaLRUSlot>(node_indices, host_values)
    }

    pub fn has_host_swa_value(&self, node_idx: NodeIdx) -> bool {
        HostSwaLRUSlot::has_value(self.tree_node_pool.get(node_idx))
    }

    pub fn set_host_mamba_values(
        &mut self,
        node_indices: Vec<NodeIdx>,
        host_values: Vec<tch::Tensor>,
    ) -> Result<Vec<tch::Tensor>, RadixCacheRuntimeError> {
        self.set_host_aux_values::<HostMambaLRUSlot, MambaLRUSlot>(node_indices, host_values)
    }

    pub fn has_host_mamba_value(&self, node_idx: NodeIdx) -> bool {
        HostMambaLRUSlot::has_value(self.tree_node_pool.get(node_idx))
    }

    fn set_host_aux_values<HostSlot: LRUSlot, DeviceSlot: LRUSlot>(
        &mut self,
        node_indices: Vec<NodeIdx>,
        host_values: Vec<tch::Tensor>,
    ) -> Result<Vec<tch::Tensor>, RadixCacheRuntimeError> {
        if node_indices.len() != host_values.len() {
            return Err(RadixCacheRuntimeError::SetHostFullValuesMismatch {
                indices: node_indices.len(),
                values: host_values.len(),
            });
        }
        let mut replaced_values = Vec::new();
        for (idx, value) in node_indices.into_iter().zip(host_values) {
            let device_has_value = DeviceSlot::has_value(self.tree_node_pool.get(idx));
            let host_was_in_lru = HostSlot::data(self.tree_node_pool.get(idx)).in_list;
            let host_lock_ref = HostSlot::lock_ref(self.tree_node_pool.get(idx));
            let old_host_len = HostSlot::value_len(self.tree_node_pool.get(idx));

            if HostSlot::has_value(self.tree_node_pool.get(idx)) {
                if host_lock_ref > 0 {
                    // This host value may be the source of an in-flight H2D
                    // load-back. Keep it stable until the load ack releases
                    // the host lock; the newly allocated backup is returned to
                    // Python so the host pool can free it.
                    replaced_values.push(value);
                    continue;
                }
                if host_was_in_lru {
                    HostSlot::remove(&mut self.tree_node_pool, idx);
                    HostSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size -=
                        old_host_len;
                }
                let old_value = HostSlot::node_state_mut(self.tree_node_pool.get_mut(idx))
                    .value
                    .take()
                    .expect("host value checked above");
                replaced_values.push(old_value);
            }
            let value_len = value.size()[0] as usize;
            HostSlot::set_value(&mut self.tree_node_pool, idx, value)?;
            if !device_has_value {
                if host_lock_ref == 0 {
                    HostSlot::bump_mru(&mut self.tree_node_pool, idx);
                    HostSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += value_len;
                } else {
                    HostSlot::pool_state_mut(&mut self.tree_node_pool).locked_size += value_len;
                }
            }
        }
        Ok(replaced_values)
    }

    pub fn postprocess_aux_load_back(
        &mut self,
        swa_chain: Vec<NodeIdx>,
        swa_device_values: Option<Tensor>,
        mamba_chain: Vec<NodeIdx>,
        mamba_device_values: Option<Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if let Some(values) = swa_device_values {
            self.write_aux_load_back_values::<HostSwaLRUSlot, SwaLRUSlot>(&swa_chain, &values)?;
        }
        if let Some(values) = mamba_device_values {
            self.write_aux_load_back_values::<HostMambaLRUSlot, MambaLRUSlot>(
                &mamba_chain,
                &values,
            )?;
        }
        Ok(())
    }

    fn write_aux_load_back_values<HostSlot: LRUSlot, DeviceSlot: LRUSlot>(
        &mut self,
        chain: &[NodeIdx],
        device_values: &Tensor,
    ) -> Result<(), RadixCacheRuntimeError> {
        let expected: usize = chain
            .iter()
            .map(|&idx| HostSlot::value_len(self.tree_node_pool.get(idx)))
            .sum();
        let got = device_values.size()[0] as usize;
        if got != expected {
            return Err(RadixCacheRuntimeError::PostprocessLoadBackLengthMismatch {
                got,
                expected,
            });
        }
        let mut offset: i64 = 0;
        for &idx in chain {
            let len = HostSlot::value_len(self.tree_node_pool.get(idx));
            let device_value = device_values.narrow(0, offset, len as i64);
            offset += len as i64;
            DeviceSlot::set_value(&mut self.tree_node_pool, idx, device_value)?;
            DeviceSlot::bump_mru(&mut self.tree_node_pool, idx);
            DeviceSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += len;
            // Keep the freshly allocated H2D destination out of eviction
            // until the transfer ack arrives. Non-FULL component locks are
            // capped by the same-node FULL lock.
            FullLRUSlot::inc_lock_ref(&mut self.tree_node_pool, idx);
            DeviceSlot::inc_lock_ref(&mut self.tree_node_pool, idx);
        }
        Ok(())
    }

    fn release_aux_host_lock<HostSlot: LRUSlot, DeviceSlot: LRUSlot>(&mut self, idx: NodeIdx) {
        if DeviceSlot::has_value(self.tree_node_pool.get(idx))
            && DeviceSlot::lock_ref(self.tree_node_pool.get(idx)) > 0
        {
            DeviceSlot::dec_lock_ref(&mut self.tree_node_pool, idx);
            FullLRUSlot::dec_lock_ref(&mut self.tree_node_pool, idx);
        }

        let lock_ref = HostSlot::lock_ref(self.tree_node_pool.get(idx));
        if lock_ref == 0 {
            return;
        }
        if !DeviceSlot::has_value(self.tree_node_pool.get(idx)) {
            HostSlot::dec_lock_ref(&mut self.tree_node_pool, idx);
            return;
        }
        if lock_ref > 1 {
            HostSlot::set_lock_ref(self.tree_node_pool.get_mut(idx), lock_ref - 1);
            return;
        }

        if HostSlot::data(self.tree_node_pool.get(idx)).in_list {
            let len = HostSlot::value_len(self.tree_node_pool.get(idx));
            HostSlot::remove(&mut self.tree_node_pool, idx);
            HostSlot::pool_state_mut(&mut self.tree_node_pool).locked_size -= len;
        }
        HostSlot::set_lock_ref(self.tree_node_pool.get_mut(idx), 0);
    }

    pub fn release_aux_host_locks(&mut self, swa_chain: Vec<NodeIdx>, mamba_chain: Vec<NodeIdx>) {
        for idx in swa_chain {
            self.release_aux_host_lock::<HostSwaLRUSlot, SwaLRUSlot>(idx);
        }
        for idx in mamba_chain {
            self.release_aux_host_lock::<HostMambaLRUSlot, MambaLRUSlot>(idx);
        }
    }

    /// Commit the loadback results back to the radix tree. A `None`
    /// `device_values` indicates loadback failure.
    pub fn postprocess_load_back(
        &mut self,
        chain: Vec<NodeIdx>,
        ancestor_node_idx: NodeIdx,
        device_values: Option<Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        // Capture the write result to ensure lock release.
        let write_result = match &device_values {
            Some(device_values) => self.write_load_back_values(&chain, device_values),
            None => Ok(()),
        };
        // Release prepare_load_back's FULL device anchor lock. On success,
        // keep the host source chain locked until the H2D copy ack; on
        // failure/skip no copy will reference the host source, so release it
        // here.
        if write_result.is_err() || device_values.is_none() {
            self.release_host_chain_locks(&chain);
        }
        self.dec_full_lock_ref(ancestor_node_idx);
        // On success, hand the FULL device lock to the loaded prefix so it
        // survives until the request is scheduled.
        // The orchestrator releases this together with the host source locks
        // in `finish_load_back` after the load ack.
        if write_result.is_ok() && device_values.is_some() {
            if let Some(&deepest) = chain.last() {
                self.inc_full_lock_ref(deepest);
            }
        }
        write_result
    }

    /// Release the locks that protect a completed load-back: the host source
    /// chain locked by `prepare_load_back`, and the loaded device prefix lock
    /// handed off by `postprocess_load_back`.
    pub fn finish_load_back(&mut self, chain: Vec<NodeIdx>, loaded_node_idx: NodeIdx) {
        self.release_host_chain_locks(&chain);
        self.dec_full_lock_ref(loaded_node_idx);
    }

    fn release_host_chain_locks(&mut self, chain: &[NodeIdx]) {
        for &idx in chain {
            HostFullLRUSlot::dec_lock_ref(&mut self.tree_node_pool, idx);
        }
    }

    /// Restore device FULL values after an orchestrator-side write-back failure.
    pub fn restore_full_values(
        &mut self,
        node_indices: Vec<NodeIdx>,
        device_values: Vec<tch::Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if node_indices.len() != device_values.len() {
            return Err(RadixCacheRuntimeError::SetHostFullValuesMismatch {
                indices: node_indices.len(),
                values: device_values.len(),
            });
        }
        for (idx, value) in node_indices.into_iter().zip(device_values) {
            let value_len = value.size()[0] as usize;
            FullLRUSlot::set_value(&mut self.tree_node_pool, idx, value)?;
            FullLRUSlot::bump_mru(&mut self.tree_node_pool, idx);
            FullLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += value_len;
        }
        Ok(())
    }

    /// Remove aux device values after the Full value has been demoted to host.
    /// `full_value_cookie` is the Full index tensor Python's SWA allocator uses
    /// to release the translated SWA slots.
    pub fn demote_aux_device_values(
        &mut self,
        node_idx: NodeIdx,
        full_value_cookie: Tensor,
    ) -> EvictResult {
        let mut result = EvictResult::default();
        if self.has_swa_component && SwaLRUSlot::has_value(self.tree_node_pool.get(node_idx)) {
            let len = SwaLRUSlot::value_len(self.tree_node_pool.get(node_idx));
            SwaLRUSlot::node_state_mut(self.tree_node_pool.get_mut(node_idx))
                .value
                .take();
            if SwaLRUSlot::data(self.tree_node_pool.get(node_idx)).in_list {
                SwaLRUSlot::remove(&mut self.tree_node_pool, node_idx);
            }
            SwaLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size -= len;
            result.evicted[ComponentType::Swa as usize] += len;
            result.freed[ComponentType::Swa as usize].push(full_value_cookie.shallow_clone());
            self.promote_host_aux_lru_if_backed::<HostSwaLRUSlot>(node_idx);
        }
        if self.has_mamba_component && MambaLRUSlot::has_value(self.tree_node_pool.get(node_idx)) {
            let len = MambaLRUSlot::value_len(self.tree_node_pool.get(node_idx));
            if let Some(value) = MambaLRUSlot::node_state_mut(self.tree_node_pool.get_mut(node_idx))
                .value
                .take()
            {
                result.evicted[ComponentType::Mamba as usize] += len;
                result.freed[ComponentType::Mamba as usize].push(value);
            }
            if MambaLRUSlot::data(self.tree_node_pool.get(node_idx)).in_list {
                MambaLRUSlot::remove(&mut self.tree_node_pool, node_idx);
            }
            MambaLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size -= len;
            self.promote_host_aux_lru_if_backed::<HostMambaLRUSlot>(node_idx);
        }
        result
    }

    fn promote_host_aux_lru_if_backed<HostSlot: LRUSlot>(&mut self, node_idx: NodeIdx) {
        if HostSlot::has_value(self.tree_node_pool.get(node_idx))
            && !HostSlot::data(self.tree_node_pool.get(node_idx)).in_list
        {
            let len = HostSlot::value_len(self.tree_node_pool.get(node_idx));
            HostSlot::bump_mru(&mut self.tree_node_pool, node_idx);
            HostSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += len;
        }
    }

    /// Write loaded values onto the chain.
    fn write_load_back_values(
        &mut self,
        chain: &[NodeIdx],
        device_values: &Tensor,
    ) -> Result<(), RadixCacheRuntimeError> {
        // The loaded slice must cover exactly the chain's tokens, so each
        // per-node narrow below stays in-bounds.
        let expected: usize = chain
            .iter()
            .map(|&idx| HostFullLRUSlot::value_len(self.tree_node_pool.get(idx)))
            .sum();
        let got = device_values.size()[0] as usize;
        if got != expected {
            return Err(RadixCacheRuntimeError::PostprocessLoadBackLengthMismatch {
                got,
                expected,
            });
        }
        let mut offset: i64 = 0;
        for &idx in chain {
            // Host-only chain node => lock_ref==0, so its tokens credit
            // unlocked_size (postprocess re-locks the loaded prefix afterward).
            assert_eq!(
                FullLRUSlot::lock_ref(self.tree_node_pool.get(idx)),
                0,
                "load-back: host-only chain node {idx} has non-zero lock_ref",
            );
            let len = HostFullLRUSlot::value_len(self.tree_node_pool.get(idx));
            let device_value = device_values.narrow(0, offset, len as i64);
            offset += len as i64;
            // TODO(Jialin): fold set_value + bump_mru + unlocked_size credit into
            // a shared set_value(idx, value, update_lru) helper.
            FullLRUSlot::set_value(&mut self.tree_node_pool, idx, device_value)?;
            FullLRUSlot::bump_mru(&mut self.tree_node_pool, idx);
            FullLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += len;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(len: i64) -> Tensor {
        Tensor::zeros([len], (Kind::Int64, Device::Cpu))
    }

    fn ones(len: i64) -> Tensor {
        Tensor::ones([len], (Kind::Int64, Device::Cpu))
    }

    fn mamba_cache() -> RadixCache<Vec<i64>> {
        RadixCache::new(Device::Cpu, 1, 16, None, Some(1), true, false).expect("valid mamba cache")
    }

    fn mamba_l1_cache() -> RadixCache<Vec<i64>> {
        RadixCache::new(Device::Cpu, 1, 16, None, Some(1), false, false).expect("valid mamba cache")
    }

    fn swa_cache() -> RadixCache<Vec<i64>> {
        RadixCache::new(Device::Cpu, 1, 16, Some(2), None, false, false).expect("valid swa cache")
    }

    fn apply_swa_stamps(cache: &mut RadixCache<Vec<i64>>, actions: Vec<DeferredAction>) {
        let mut node_indices = Vec::new();
        let mut values = Vec::new();
        for action in actions {
            if let DeferredAction::SwaStamp {
                node_idx,
                source_value,
            } = action
            {
                node_indices.push(node_idx);
                values.push(source_value);
            }
        }
        cache
            .apply_swa_writes(node_indices, values)
            .expect("swa stamps apply");
    }

    fn apply_swa_stamps_and_recovers(
        cache: &mut RadixCache<Vec<i64>>,
        actions: Vec<DeferredAction>,
    ) {
        let mut node_indices = Vec::new();
        let mut values = Vec::new();
        for action in actions {
            match action {
                DeferredAction::SwaStamp {
                    node_idx,
                    source_value,
                }
                | DeferredAction::SwaRecover {
                    node_idx,
                    source_value,
                    ..
                } => {
                    node_indices.push(node_idx);
                    values.push(source_value);
                }
                _ => {}
            }
        }
        cache
            .apply_swa_writes(node_indices, values)
            .expect("swa writes apply");
    }

    fn insert_mamba_leaf(cache: &mut RadixCache<Vec<i64>>) -> NodeIdx {
        cache
            .insert(&[1], &tensor(1), None, 0, 0, Some(tensor(1)))
            .expect("insert succeeds");
        cache
            .match_prefix(&[1], None)
            .expect("match succeeds")
            .best_match_node_idx
    }

    #[test]
    fn swa_dec_lock_ref_skips_zero_ref_ancestors() {
        let mut cache = swa_cache();
        let insert = cache
            .insert(&[1, 2, 3], &tensor(3), None, 0, 0, None)
            .expect("insert succeeds");
        apply_swa_stamps(&mut cache, insert.deferred_actions);

        let leaf = cache
            .match_prefix(&[1, 2, 3], None)
            .expect("match succeeds")
            .best_match_node_idx;
        let lock = cache.inc_lock_ref(leaf);
        cache
            .insert(&[1, 2, 4], &tensor(3), None, 0, 0, None)
            .expect("split insert succeeds");
        let child = cache
            .match_prefix(&[1, 2, 3], None)
            .expect("match child succeeds")
            .best_match_node_idx;
        cache.dec_lock_ref(child, lock.swa_uuid_for_lock);

        let parent = cache
            .tree_node_pool
            .get(child)
            .parent()
            .expect("split child has parent");
        let stale_parent_uuid = cache
            .tree_node_pool
            .get(parent)
            .swa_uuid_for_lock()
            .expect("split parent keeps stale uuid");
        assert_eq!(SwaLRUSlot::lock_ref(cache.tree_node_pool.get(parent)), 0);

        // Simulate a release path that carries an older boundary uuid through
        // a zero-ref ancestor. Python Unified skips those nodes; Rust must do
        // the same for split/sidecar paths where structural ancestors are on
        // the route but outside the request's SWA lock window.
        FullLRUSlot::inc_lock_ref(&mut cache.tree_node_pool, child);
        SwaLRUSlot::inc_lock_ref(&mut cache.tree_node_pool, child);
        let swa = SwaComponent::new(2).expect("valid swa component");
        <SwaComponent as Component<Vec<i64>>>::dec_lock_ref(
            &swa,
            &mut cache.tree_node_pool,
            child,
            Some(stale_parent_uuid),
        )
        .expect("swa release returns delta");

        assert_eq!(SwaLRUSlot::lock_ref(cache.tree_node_pool.get(child)), 0);
        assert_eq!(SwaLRUSlot::lock_ref(cache.tree_node_pool.get(parent)), 0);
    }

    #[test]
    fn swa_dsv4_branching_decode_paths_remain_matchable() {
        let mut cache = RadixCache::new(Device::Cpu, 256, 16, Some(128), None, true, false)
            .expect("valid DSV4-like SWA cache");
        let mut first_turn_paths = Vec::new();
        for group in 0..3 {
            let base_start = group * 10_000;
            let base: Vec<i64> = (base_start..base_start + 512).collect();
            for branch in 0..3 {
                let output_start = base_start + 1_000 + branch * 1_000;
                let output: Vec<i64> = (output_start..output_start + 512).collect();
                let mut path = base.clone();
                path.extend(output);
                let value = tensor(path.len() as i64);
                let insert = cache
                    .insert(&path, &value, None, 0, 512, None)
                    .expect("insert first-turn branch succeeds");
                apply_swa_stamps_and_recovers(&mut cache, insert.deferred_actions);
                first_turn_paths.push(path);
            }
        }

        for (idx, path) in first_turn_paths.iter().enumerate() {
            let mut decode_path = path.clone();
            decode_path.extend(
                (50_000 + idx as i64 * 1_000..50_000 + idx as i64 * 1_000 + 512)
                    .collect::<Vec<_>>(),
            );
            let result = cache
                .match_prefix(&decode_path, None)
                .expect("decode path match succeeds");
            assert_eq!(
                result.device_indices.size()[0],
                1024,
                "branch {idx} should reuse first-turn prompt+output"
            );
        }
    }

    #[test]
    fn mamba_hicache_match_skips_branching_state_fill() {
        let mut cache = mamba_cache();
        let leaf = insert_mamba_leaf(&mut cache);
        cache
            .insert(&[1, 2], &tensor(2), None, 1, 0, Some(tensor(1)))
            .expect("insert second leaf succeeds");
        let child = cache
            .match_prefix(&[1, 2], None)
            .expect("match child succeeds")
            .best_match_node_idx;
        assert_ne!(leaf, child);

        let evicted = cache.demote_aux_device_values(child, tensor(1));
        assert_eq!(evicted.evicted[ComponentType::Mamba as usize], 1);

        let result = cache
            .match_prefix(&[1, 2, 3], None)
            .expect("match succeeds");
        assert_eq!(result.best_match_node_idx, leaf);
        assert_eq!(result.last_device_node_idx, leaf);
        assert_eq!(result.device_indices.size()[0], 1);
        assert_eq!(result.mamba_branching_seqlen, None);
    }

    #[test]
    fn mamba_l1_match_keeps_branching_state_fill() {
        let mut cache = mamba_l1_cache();
        let leaf = insert_mamba_leaf(&mut cache);
        cache
            .insert(&[1, 2], &tensor(2), None, 1, 0, Some(tensor(1)))
            .expect("insert second leaf succeeds");
        let child = cache
            .match_prefix(&[1, 2], None)
            .expect("match child succeeds")
            .best_match_node_idx;
        assert_ne!(leaf, child);

        let evicted = cache.demote_aux_device_values(child, tensor(1));
        assert_eq!(evicted.evicted[ComponentType::Mamba as usize], 1);

        let result = cache
            .match_prefix(&[1, 2, 3], None)
            .expect("match succeeds");
        assert_eq!(result.best_match_node_idx, leaf);
        assert_eq!(result.last_device_node_idx, leaf);
        assert_eq!(result.device_indices.size()[0], 1);
        assert_eq!(result.mamba_branching_seqlen, Some(2));
    }

    #[test]
    fn prepare_load_back_locks_full_anchor_without_swa_value() {
        let mut cache = RadixCache::new(Device::Cpu, 1, 16, Some(1), None, true, false)
            .expect("valid SWA HiCache");
        let first = cache
            .insert(&[1], &tensor(1), None, 0, 1, None)
            .expect("insert parent succeeds");
        apply_swa_stamps_and_recovers(&mut cache, first.deferred_actions);
        let second = cache
            .insert(&[1, 2], &tensor(2), None, 1, 1, None)
            .expect("insert child succeeds");
        apply_swa_stamps_and_recovers(&mut cache, second.deferred_actions);

        let leaf = cache
            .match_prefix(&[1, 2], None)
            .expect("match child succeeds")
            .best_match_node_idx;
        let parent = cache
            .tree_node_pool
            .get(leaf)
            .parent()
            .expect("child has parent");
        assert!(FullLRUSlot::has_value(cache.tree_node_pool.get(parent)));
        assert!(!SwaLRUSlot::has_value(cache.tree_node_pool.get(parent)));
        assert!(SwaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));

        cache
            .set_host_full_values(vec![parent, leaf], vec![tensor(1), tensor(1)])
            .expect("host full backup succeeds");
        let mut request = EvictRequest::default();
        request.num_tokens[ComponentType::Full as usize] = 1;
        cache.evict(request);
        assert!(!FullLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert!(HostFullLRUSlot::has_value(cache.tree_node_pool.get(leaf)));

        let plan = cache
            .prepare_load_back(leaf)
            .expect("prepare loadback should not require SWA on full anchor");
        assert_eq!(plan.chain, vec![leaf]);
        assert_eq!(plan.ancestor_node_idx, parent);
        assert_eq!(FullLRUSlot::lock_ref(cache.tree_node_pool.get(parent)), 1);
        assert_eq!(SwaLRUSlot::lock_ref(cache.tree_node_pool.get(parent)), 0);

        cache
            .postprocess_load_back(plan.chain, plan.ancestor_node_idx, None)
            .expect("rollback releases locks");
        assert_eq!(FullLRUSlot::lock_ref(cache.tree_node_pool.get(parent)), 0);
        assert_eq!(HostFullLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)), 0);
    }

    #[test]
    fn swa_recover_after_full_unevict_does_not_free_restored_full_value() {
        let mut cache = RadixCache::new(Device::Cpu, 1, 16, Some(2), None, true, false)
            .expect("valid SWA HiCache");
        let inserted = cache
            .insert(&[1, 2], &tensor(2), None, 0, 0, None)
            .expect("insert succeeds");
        apply_swa_stamps_and_recovers(&mut cache, inserted.deferred_actions);

        let leaf = cache
            .match_prefix(&[1, 2], None)
            .expect("match succeeds")
            .best_match_node_idx;
        assert!(FullLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert!(SwaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));

        cache
            .set_host_swa_values(vec![leaf], vec![tensor(2)])
            .expect("host SWA backup succeeds");
        let mut evict_swa = EvictRequest::default();
        evict_swa.num_tokens[ComponentType::Swa as usize] = 2;
        cache.evict(evict_swa);
        assert!(FullLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert!(!SwaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));

        cache
            .set_host_full_values(vec![leaf], vec![tensor(2)])
            .expect("host backup succeeds");
        let mut evict_full = EvictRequest::default();
        evict_full.num_tokens[ComponentType::Full as usize] = 2;
        cache.evict(evict_full);
        assert!(!FullLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert!(HostFullLRUSlot::has_value(cache.tree_node_pool.get(leaf)));

        let recovered = cache
            .insert(&[1, 2], &ones(2), None, 0, 0, None)
            .expect("unevict insert succeeds");
        assert!(
            recovered
                .deferred_actions
                .iter()
                .any(|action| matches!(action, DeferredAction::SwaStamp { .. })),
            "SWA should be rebuilt from the restored Full value"
        );
        assert!(
            recovered
                .deferred_actions
                .iter()
                .all(|action| !matches!(action, DeferredAction::SwaRecover { .. })),
            "Full value restored by the same insert must not be freed as an old tombstone"
        );
    }

    #[test]
    fn aux_load_back_keeps_mamba_host_source_locked_until_ack() {
        let mut cache = mamba_cache();
        let leaf = insert_mamba_leaf(&mut cache);
        cache
            .set_host_mamba_values(vec![leaf], vec![tensor(1)])
            .expect("host mamba backup succeeds");

        cache.demote_aux_device_values(leaf, tensor(1));
        assert_eq!(HostMambaLRUSlot::unlocked_size(&cache.tree_node_pool), 1);

        let aux_plan = cache.prepare_aux_load_back(leaf);
        assert_eq!(aux_plan.mamba_chain, vec![leaf]);
        assert_eq!(
            HostMambaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)),
            1
        );
        assert_eq!(HostMambaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);
        assert_eq!(HostMambaLRUSlot::locked_size(&cache.tree_node_pool), 1);

        cache
            .postprocess_aux_load_back(
                Vec::new(),
                None,
                aux_plan.mamba_chain.clone(),
                Some(tensor(1)),
            )
            .expect("aux postprocess succeeds");
        assert!(MambaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert!(HostMambaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert_eq!(
            HostMambaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)),
            1
        );

        let evicted = cache.evict_host_mamba(1);
        assert_eq!(evicted.evicted[ComponentType::Mamba as usize], 0);
        assert!(HostMambaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));

        cache.release_aux_host_locks(Vec::new(), aux_plan.mamba_chain);
        assert_eq!(
            HostMambaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)),
            0
        );
        assert!(!HostMambaLRUSlot::data(cache.tree_node_pool.get(leaf)).in_list);
        assert_eq!(HostMambaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);
        assert_eq!(HostMambaLRUSlot::locked_size(&cache.tree_node_pool), 0);
    }

    #[test]
    fn aux_load_back_accounts_swa_device_until_ack() {
        let mut cache = RadixCache::new(Device::Cpu, 1, 16, Some(2), None, true, false)
            .expect("valid SWA HiCache");
        let insert = cache
            .insert(&[1, 2], &tensor(2), None, 0, 0, None)
            .expect("insert succeeds");
        apply_swa_stamps_and_recovers(&mut cache, insert.deferred_actions);
        let leaf = cache
            .match_prefix(&[1, 2], None)
            .expect("match succeeds")
            .best_match_node_idx;
        cache
            .set_host_swa_values(vec![leaf], vec![tensor(2)])
            .expect("host swa backup succeeds");
        cache.demote_aux_device_values(leaf, tensor(2));
        assert_eq!(SwaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);
        assert_eq!(HostSwaLRUSlot::unlocked_size(&cache.tree_node_pool), 2);

        let aux_plan = cache.prepare_aux_load_back(leaf);
        assert_eq!(aux_plan.swa_chain, vec![leaf]);
        assert_eq!(HostSwaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)), 1);
        cache
            .postprocess_aux_load_back(
                aux_plan.swa_chain.clone(),
                Some(tensor(2)),
                Vec::new(),
                None,
            )
            .expect("aux postprocess succeeds");

        assert_eq!(FullLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)), 1);
        assert_eq!(SwaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)), 1);
        assert_eq!(SwaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);
        assert_eq!(SwaLRUSlot::locked_size(&cache.tree_node_pool), 2);

        cache.release_aux_host_locks(aux_plan.swa_chain, Vec::new());
        assert_eq!(FullLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)), 0);
        assert_eq!(SwaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)), 0);
        assert_eq!(SwaLRUSlot::unlocked_size(&cache.tree_node_pool), 2);
        assert_eq!(SwaLRUSlot::locked_size(&cache.tree_node_pool), 0);
        assert_eq!(HostSwaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);
        assert_eq!(HostSwaLRUSlot::locked_size(&cache.tree_node_pool), 0);
    }

    #[test]
    fn host_swa_split_keeps_s2_backup_off_host_lru() {
        let mut cache = RadixCache::new(Device::Cpu, 1, 16, Some(2), None, true, false)
            .expect("valid SWA HiCache");
        let insert = cache
            .insert(&[1, 2, 3, 4], &tensor(4), None, 0, 0, None)
            .expect("insert succeeds");
        apply_swa_stamps_and_recovers(&mut cache, insert.deferred_actions);
        let leaf = cache
            .match_prefix(&[1, 2, 3, 4], None)
            .expect("match succeeds")
            .best_match_node_idx;
        cache
            .set_host_swa_values(vec![leaf], vec![tensor(4)])
            .expect("host swa backup succeeds");
        assert_eq!(HostSwaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);

        let parent = {
            let components = &cache.components;
            let pool = &mut cache.tree_node_pool;
            pool.split_node(
                components,
                crate::tree_node_pool::NodeSplit {
                    child_idx: leaf,
                    split_len: 2,
                },
            )
        };

        assert_eq!(FullLRUSlot::value_len(cache.tree_node_pool.get(parent)), 2);
        assert_eq!(FullLRUSlot::value_len(cache.tree_node_pool.get(leaf)), 2);
        assert_eq!(SwaLRUSlot::value_len(cache.tree_node_pool.get(parent)), 2);
        assert_eq!(SwaLRUSlot::value_len(cache.tree_node_pool.get(leaf)), 2);
        assert_eq!(
            HostSwaLRUSlot::value_len(cache.tree_node_pool.get(parent)),
            2
        );
        assert_eq!(HostSwaLRUSlot::value_len(cache.tree_node_pool.get(leaf)), 2);
        assert!(!HostSwaLRUSlot::data(cache.tree_node_pool.get(parent)).in_list);
        assert!(!HostSwaLRUSlot::data(cache.tree_node_pool.get(leaf)).in_list);
        assert_eq!(HostSwaLRUSlot::unlocked_size(&cache.tree_node_pool), 0);
    }

    #[test]
    fn host_swa_split_redistributes_host_only_lru_order() {
        let mut cache = RadixCache::new(Device::Cpu, 1, 16, Some(2), None, true, false)
            .expect("valid SWA HiCache");
        let insert = cache
            .insert(&[1, 2, 3, 4], &tensor(4), None, 0, 0, None)
            .expect("insert succeeds");
        apply_swa_stamps_and_recovers(&mut cache, insert.deferred_actions);
        let leaf = cache
            .match_prefix(&[1, 2, 3, 4], None)
            .expect("match succeeds")
            .best_match_node_idx;
        cache
            .set_host_swa_values(vec![leaf], vec![tensor(4)])
            .expect("host swa backup succeeds");
        cache.demote_aux_device_values(leaf, tensor(4));
        assert_eq!(HostSwaLRUSlot::unlocked_size(&cache.tree_node_pool), 4);

        let parent = {
            let components = &cache.components;
            let pool = &mut cache.tree_node_pool;
            pool.split_node(
                components,
                crate::tree_node_pool::NodeSplit {
                    child_idx: leaf,
                    split_len: 2,
                },
            )
        };

        assert!(!SwaLRUSlot::has_value(cache.tree_node_pool.get(parent)));
        assert!(!SwaLRUSlot::has_value(cache.tree_node_pool.get(leaf)));
        assert_eq!(
            HostSwaLRUSlot::value_len(cache.tree_node_pool.get(parent)),
            2
        );
        assert_eq!(HostSwaLRUSlot::value_len(cache.tree_node_pool.get(leaf)), 2);
        assert!(HostSwaLRUSlot::data(cache.tree_node_pool.get(parent)).in_list);
        assert!(HostSwaLRUSlot::data(cache.tree_node_pool.get(leaf)).in_list);
        assert_eq!(HostSwaLRUSlot::unlocked_size(&cache.tree_node_pool), 4);
        assert_eq!(HostSwaLRUSlot::locked_size(&cache.tree_node_pool), 0);
    }

    #[test]
    fn set_host_mamba_values_does_not_replace_locked_source() {
        let mut cache = mamba_cache();
        let leaf = insert_mamba_leaf(&mut cache);
        cache
            .set_host_mamba_values(vec![leaf], vec![tensor(1)])
            .expect("host mamba backup succeeds");
        cache.demote_aux_device_values(leaf, tensor(1));

        let aux_plan = cache.prepare_aux_load_back(leaf);
        assert_eq!(
            HostMambaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)),
            1
        );

        let values_to_free = cache
            .set_host_mamba_values(vec![leaf], vec![ones(1)])
            .expect("locked replacement is skipped");
        assert_eq!(values_to_free.len(), 1);
        assert_eq!(values_to_free[0].int64_value(&[0]), 1);
        assert_eq!(
            HostMambaLRUSlot::value(cache.tree_node_pool.get(leaf))
                .expect("old host value remains")
                .int64_value(&[0]),
            0
        );

        cache.release_aux_host_locks(Vec::new(), aux_plan.mamba_chain);
    }

    #[test]
    fn release_aux_host_locks_without_device_value_restores_host_lru() {
        let mut cache = mamba_cache();
        let leaf = insert_mamba_leaf(&mut cache);
        cache
            .set_host_mamba_values(vec![leaf], vec![tensor(1)])
            .expect("host mamba backup succeeds");
        cache.demote_aux_device_values(leaf, tensor(1));

        let aux_plan = cache.prepare_aux_load_back(leaf);
        cache.release_aux_host_locks(Vec::new(), aux_plan.mamba_chain);

        assert_eq!(
            HostMambaLRUSlot::lock_ref(cache.tree_node_pool.get(leaf)),
            0
        );
        assert!(HostMambaLRUSlot::data(cache.tree_node_pool.get(leaf)).in_list);
        assert_eq!(HostMambaLRUSlot::unlocked_size(&cache.tree_node_pool), 1);
        assert_eq!(HostMambaLRUSlot::locked_size(&cache.tree_node_pool), 0);
    }
}

/// Production radix cache: children keyed by token page (`Vec<i64>`).
/// Handles `page_size >= 1` — `page_size=1` uses one-element page keys.
pub type PageRadixCache = RadixCache<Vec<i64>>;

/// Bigram-keyed radix cache: children keyed by `(t[i], t[i+1])` pairs.
/// Today used by EAGLE speculative decoding, but the abstraction is
/// "bigram", not Eagle-specific — keeps the door open for any other
/// caller that wants overlap-pair keys. Currently exposed only through
/// the bench wrapper; production wiring lands in a follow-up PR.
pub type BigramRadixCache = RadixCache<Vec<(i64, i64)>>;
