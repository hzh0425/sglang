// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
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
            policy: PolicyKind::StickySessionLoadBased,
            circuit_breaker: None,
            cache_aware: None,
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
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

#[tokio::test]
async fn sticky_session_overrides_later_load_based_fallback() {
    let worker_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let worker_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let (ctx, w0, w1) = build_ctx(&worker_a.url, &worker_b.url);
    let app = build_router(ctx);

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

    let res = app
        .oneshot(chat_request("session-b", "third"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();
    assert_eq!(request_count(&worker_a), 2);
    assert_eq!(request_count(&worker_b), 1);
}
