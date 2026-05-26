// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PdBucketConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlGroupConfig,
    StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{LoadGuard, Worker, WorkerRegistry};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::StickySessionCacheAwareLoadBased,
            circuit_breaker: None,
            cache_aware: None,
            pd_bucket: None,
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
                worker_groups: Vec::new(),
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

fn build_ctx(
    worker_a_url: &str,
    worker_b_url: &str,
) -> (Arc<AppContext>, Arc<Worker>, Arc<Worker>) {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (id, url) in [("w0", worker_a_url), ("w1", worker_b_url)] {
        registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: url.to_string(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let w0 = registry.get(&WorkerId("w0".into())).unwrap();
    let w1 = registry.get(&WorkerId("w1".into())).unwrap();
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    (ctx, w0, w1)
}

fn chat_request(key: &str, prompt: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-sglang-routing-key", key)
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4,
            }))
            .unwrap(),
        ))
        .unwrap()
}

fn request_count(worker: &crate::common::mock_worker::MockWorker) -> usize {
    worker.captured.lock().unwrap().request_count
}

async fn wait_for_request_count(worker: &crate::common::mock_worker::MockWorker, expected: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline {
        if request_count(worker) == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(request_count(worker), expected);
}

#[tokio::test]
async fn sticky_session_overrides_later_load_based_fallback() {
    let worker_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let worker_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let (ctx, w0, w1) = build_ctx(&worker_a.url, &worker_b.url);
    let app = build_router(ctx);

    // First request: make w1 busy so the terminal load-based step assigns
    // session-a to w0.
    let w1_busy = w1.load_guard();
    let res = app
        .clone()
        .oneshot(chat_request("session-a", "first"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();
    assert_eq!(request_count(&worker_a), 1);
    assert_eq!(request_count(&worker_b), 0);
    drop(w1_busy);

    // Now w0 is objectively hotter than w1. Sticky should still keep
    // session-a on w0.
    let _w0_hot: Vec<LoadGuard> = (0..4).map(|_| w0.load_guard()).collect();
    let res = app
        .clone()
        .oneshot(chat_request("session-a", "second"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();
    assert_eq!(request_count(&worker_a), 2);
    assert_eq!(request_count(&worker_b), 0);

    // A different routing key is not sticky yet, so it should follow the
    // load-based fallback and land on the cooler worker.
    let res = app
        .oneshot(chat_request("session-b", "third"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();
    assert_eq!(request_count(&worker_a), 2);
    assert_eq!(request_count(&worker_b), 1);
}

fn pd_bucket_config(workers: &[(&str, &str)]) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::StickySessionCacheAwareLoadBased,
            circuit_breaker: None,
            cache_aware: None,
            pd_bucket: Some(PdBucketConfig {
                groups: Vec::new(),
                short_group: "short".into(),
                long_group: "long".into(),
                prefill_long_threshold: 128,
                decode_long_threshold: 128,
            }),
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: workers.iter().map(|(url, _)| (*url).to_string()).collect(),
                worker_groups: workers
                    .iter()
                    .map(|(url, group)| StaticUrlGroupConfig {
                        url: (*url).to_string(),
                        group: (*group).to_string(),
                    })
                    .collect(),
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

fn build_pd_bucket_ctx(workers: &[(&str, &str, WorkerMode)]) -> Arc<AppContext> {
    let worker_groups = workers
        .iter()
        .map(|(url, group, _)| (*url, *group))
        .collect::<Vec<_>>();
    let cfg = pd_bucket_config(&worker_groups);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (url, _, mode) in workers {
        registry
            .add(WorkerSpec {
                id: WorkerId(url.to_string()),
                url: (*url).to_string(),
                mode: *mode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: match *mode {
                    WorkerMode::Prefill => Some(8999),
                    WorkerMode::Plain | WorkerMode::Decode => None,
                },
            })
            .unwrap();
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

#[tokio::test]
async fn pd_bucket_groups_filter_prefill_and_decode_pools() {
    let short_prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let short_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let long_prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let long_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_pd_bucket_ctx(&[
        (&short_prefill.url, "short", WorkerMode::Prefill),
        (&short_decode.url, "short", WorkerMode::Decode),
        (&long_prefill.url, "long", WorkerMode::Prefill),
        (&long_decode.url, "long", WorkerMode::Decode),
    ]);
    let app = build_router(ctx);

    let res = app
        .clone()
        .oneshot(chat_request("session-short", "short request"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers()
            .get("x-sglang-prefill-group")
            .and_then(|v| v.to_str().ok()),
        Some("short")
    );
    assert_eq!(
        res.headers()
            .get("x-sglang-decode-group")
            .and_then(|v| v.to_str().ok()),
        Some("short")
    );
    let _ = res.into_body().collect().await.unwrap();
    wait_for_request_count(&short_prefill, 1).await;
    assert_eq!(request_count(&short_decode), 1);
    assert_eq!(request_count(&long_prefill), 0);
    assert_eq!(request_count(&long_decode), 0);

    let long_prompt = "long ".repeat(700);
    let res = app
        .oneshot(chat_request("session-long", &long_prompt))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers()
            .get("x-sglang-prefill-group")
            .and_then(|v| v.to_str().ok()),
        Some("long")
    );
    assert_eq!(
        res.headers()
            .get("x-sglang-decode-group")
            .and_then(|v| v.to_str().ok()),
        Some("long")
    );
    let _ = res.into_body().collect().await.unwrap();
    wait_for_request_count(&long_prefill, 1).await;
    assert_eq!(request_count(&short_prefill), 1);
    assert_eq!(request_count(&short_decode), 1);
    assert_eq!(request_count(&long_decode), 1);
}
