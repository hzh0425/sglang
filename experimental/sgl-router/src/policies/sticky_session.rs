// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::WorkerId;
use crate::policies::SelectionContext;
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_SESSION_TTL: Duration = Duration::from_secs(60 * 60);
const DEFAULT_REAP_INTERVAL: Duration = Duration::from_secs(30);

struct Assignment {
    worker_id: WorkerId,
    expires_at: Instant,
}

/// Per-model sticky-session table.
///
/// The owning policy registry is already keyed by model, so the table only
/// needs to map request routing keys to worker ids.
pub struct StickySessionPolicy {
    assignments: DashMap<String, Assignment>,
    ttl: Duration,
    reap_interval: Duration,
    next_reap_at: Mutex<Instant>,
}

impl std::fmt::Debug for StickySessionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StickySessionPolicy")
            .field("assignments", &self.assignments.len())
            .finish()
    }
}

impl StickySessionPolicy {
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_SESSION_TTL)
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self::with_ttl_and_reap_interval(ttl, DEFAULT_REAP_INTERVAL)
    }

    fn with_ttl_and_reap_interval(ttl: Duration, reap_interval: Duration) -> Self {
        Self {
            assignments: DashMap::new(),
            ttl,
            reap_interval,
            next_reap_at: Mutex::new(Instant::now() + reap_interval),
        }
    }

    /// Return the previously assigned worker when the routing key is known
    /// and the worker is still present in the candidate set.
    pub fn select_existing(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        let key = ctx.routing_key()?;
        let now = Instant::now();
        self.reap_expired(now);

        let mut assignment = self.assignments.get_mut(key)?;
        if assignment.expires_at <= now {
            drop(assignment);
            self.assignments.remove(key);
            return None;
        }

        let worker_id = assignment.worker_id.clone();
        if let Some(worker) = workers.iter().find(|w| w.id == worker_id) {
            assignment.expires_at = now + self.ttl;
            return Some(Arc::clone(worker));
        }
        drop(assignment);
        self.assignments.remove(key);
        None
    }

    /// Bind the routing key to `worker`. Calls without a routing key are
    /// no-ops, so callers can invoke this unconditionally after fallback
    /// selection succeeds.
    pub fn remember(&self, ctx: &SelectionContext<'_>, worker: &Worker) {
        let Some(key) = ctx.routing_key() else {
            return;
        };
        let now = Instant::now();
        self.reap_expired(now);
        self.assignments.insert(
            key.to_string(),
            Assignment {
                worker_id: worker.id.clone(),
                expires_at: now + self.ttl,
            },
        );
    }

    fn reap_expired(&self, now: Instant) {
        let Ok(mut next_reap_at) = self.next_reap_at.try_lock() else {
            return;
        };
        if now < *next_reap_at {
            return;
        }
        *next_reap_at = now + self.reap_interval;
        self.assignments
            .retain(|_, assignment| assignment.expires_at > now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerMode, WorkerSpec};

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
    fn returns_existing_assignment() {
        let sticky = StickySessionPolicy::new();
        let model = ModelId("tiny".into());
        let key = "session-a";
        let ctx = SelectionContext::with_routing_key(&model, None, Some(key));
        let w0 = worker("w0");
        let w1 = worker("w1");
        sticky.remember(&ctx, &w1);

        let selected = sticky
            .select_existing(&[w0, Arc::clone(&w1)], &ctx)
            .expect("sticky assignment should resolve");
        assert_eq!(selected.id, w1.id);
    }

    #[test]
    fn drops_stale_assignment() {
        let sticky = StickySessionPolicy::new();
        let model = ModelId("tiny".into());
        let key = "session-a";
        let ctx = SelectionContext::with_routing_key(&model, None, Some(key));
        let stale = worker("gone");
        sticky.remember(&ctx, &stale);

        assert!(sticky.select_existing(&[worker("w0")], &ctx).is_none());
        assert!(
            sticky.assignments.get(key).is_none(),
            "stale assignment should be removed"
        );
    }

    #[test]
    fn expires_assignment() {
        let sticky = StickySessionPolicy::with_ttl_and_reap_interval(
            Duration::from_millis(1),
            Duration::from_millis(1),
        );
        let model = ModelId("tiny".into());
        let key = "session-a";
        let ctx = SelectionContext::with_routing_key(&model, None, Some(key));
        let w0 = worker("w0");
        sticky.remember(&ctx, &w0);
        std::thread::sleep(Duration::from_millis(5));

        assert!(sticky.select_existing(&[w0], &ctx).is_none());
        assert!(
            sticky.assignments.get(key).is_none(),
            "expired assignment should be removed"
        );
    }

    #[test]
    fn refreshes_ttl_on_hit() {
        let sticky = StickySessionPolicy::with_ttl_and_reap_interval(
            Duration::from_millis(50),
            Duration::from_secs(60),
        );
        let model = ModelId("tiny".into());
        let key = "session-a";
        let ctx = SelectionContext::with_routing_key(&model, None, Some(key));
        let w0 = worker("w0");
        sticky.remember(&ctx, &w0);
        std::thread::sleep(Duration::from_millis(30));

        assert!(sticky.select_existing(&[w0], &ctx).is_some());
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            sticky.select_existing(&[worker("w0")], &ctx).is_some(),
            "hit should refresh the assignment TTL"
        );
    }
}
