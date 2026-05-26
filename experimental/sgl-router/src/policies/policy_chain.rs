// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::load_based::LoadBasedPolicy;
use crate::policies::sticky_session::StickySessionPolicy;
use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Two-step routing chain:
///
/// 1. sticky session: reuse an existing routing-key assignment;
/// 2. load-based: deterministic lowest-active-load fallback.
#[derive(Debug)]
pub struct PolicyChain {
    sticky: StickySessionPolicy,
}

impl PolicyChain {
    pub fn sticky_session_load_based() -> Self {
        Self {
            sticky: StickySessionPolicy::new(),
        }
    }
}

impl Policy for PolicyChain {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }
        if let Some(worker) = self.sticky.select_existing(workers, ctx) {
            return Some(worker);
        }
        let worker = LoadBasedPolicy::pick_min_load(workers)?;
        self.sticky.remember(ctx, &worker);
        Some(worker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn sticky_assignment_overrides_later_load_balance() {
        let chain = PolicyChain::sticky_session_load_based();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("session-a"));
        let w0 = worker("w0");
        let w1 = worker("w1");

        let w1_busy = w1.load_guard();
        let first = chain
            .select(&[Arc::clone(&w0), Arc::clone(&w1)], &ctx)
            .expect("first request should select");
        assert_eq!(first.id, w0.id);
        drop(w1_busy);

        let _w0_hot = [w0.load_guard(), w0.load_guard(), w0.load_guard()];
        let second = chain
            .select(&[Arc::clone(&w0), Arc::clone(&w1)], &ctx)
            .expect("sticky request should select");
        assert_eq!(second.id, w0.id);

        let other_ctx = SelectionContext::with_routing_key(&model, None, Some("session-b"));
        let other = chain
            .select(&[w0, Arc::clone(&w1)], &other_ctx)
            .expect("other session should select");
        assert_eq!(other.id, w1.id);
    }
}
