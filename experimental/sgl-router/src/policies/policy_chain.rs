// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::CacheAwareConfig;
use crate::policies::cache_aware_zmq::CacheAwareZmqPolicy;
use crate::policies::kv_events::{BlockSizeOracle, HashTree};
use crate::policies::load_based::LoadBasedPolicy;
use crate::policies::sticky_session::StickySessionPolicy;
use crate::policies::{Policy, SelectionContext};
use crate::tokenizer::TokenizerRegistry;
use crate::workers::Worker;
use std::sync::Arc;

#[derive(Debug)]
enum PolicyChainStep {
    StickySession(StickySessionPolicy),
    CacheAwareZmq(CacheAwareZmqPolicy),
    LoadBased(LoadBasedPolicy),
}

/// Ordered policy chain for request routing.
///
/// The shipped chain is:
///
/// 1. sticky session: reuse an existing routing-key assignment;
/// 2. cache-aware-zmq: pick a worker with a strong KV-cache prefix hit;
/// 3. load-based: deterministic lowest-active-load fallback.
#[derive(Debug)]
pub struct PolicyChain {
    steps: Vec<PolicyChainStep>,
}

impl PolicyChain {
    pub fn sticky_session_cache_aware_load_based(
        cache_config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Self {
        Self {
            steps: vec![
                PolicyChainStep::StickySession(StickySessionPolicy::new()),
                PolicyChainStep::CacheAwareZmq(CacheAwareZmqPolicy::new(
                    cache_config,
                    tree,
                    tokenizers,
                    block_size_oracle,
                )),
                PolicyChainStep::LoadBased(LoadBasedPolicy::new()),
            ],
        }
    }

    fn remember_sticky(&self, ctx: &SelectionContext<'_>, worker: &Worker) {
        for step in &self.steps {
            if let PolicyChainStep::StickySession(sticky) = step {
                sticky.remember(ctx, worker);
            }
        }
    }
}

impl Policy for PolicyChain {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }

        for step in &self.steps {
            let selected = match step {
                PolicyChainStep::StickySession(sticky) => sticky.select_existing(workers, ctx),
                PolicyChainStep::CacheAwareZmq(cache) => cache.select_cache_hit(workers, ctx),
                PolicyChainStep::LoadBased(load) => load.select(workers, ctx),
            };
            if let Some(worker) = selected {
                self.remember_sticky(ctx, &worker);
                return Some(worker);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::kv_events::HashTree;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    fn chain() -> PolicyChain {
        PolicyChain::sticky_session_cache_aware_load_based(
            CacheAwareConfig::default(),
            Arc::new(HashTree::new()),
            Arc::new(TokenizerRegistry::default()),
            BlockSizeOracle::new(),
        )
    }

    #[test]
    fn sticky_assignment_overrides_later_load_balance() {
        let chain = chain();
        let model = ModelId("tiny".into());
        let key = "session-a";
        let ctx = SelectionContext::with_routing_key(&model, None, Some(key));
        let w0 = worker("w0");
        let w1 = worker("w1");

        // First request: make w1 busy, so load-based fallback assigns w0.
        let w1_busy = w1.load_guard();
        let first = chain
            .select(&[Arc::clone(&w0), Arc::clone(&w1)], &ctx)
            .expect("first request should select");
        assert_eq!(first.id, w0.id);
        drop(w1_busy);

        // Later, w0 is hotter than w1. Sticky must still route the same
        // session to w0 instead of letting load-based move it.
        let _w0_hot = [w0.load_guard(), w0.load_guard(), w0.load_guard()];
        let second = chain
            .select(&[Arc::clone(&w0), Arc::clone(&w1)], &ctx)
            .expect("sticky request should select");
        assert_eq!(second.id, w0.id);

        let other_key = "session-b";
        let other_ctx = SelectionContext::with_routing_key(&model, None, Some(other_key));
        let other = chain
            .select(&[w0, Arc::clone(&w1)], &other_ctx)
            .expect("other session should select");
        assert_eq!(other.id, w1.id);
    }
}
