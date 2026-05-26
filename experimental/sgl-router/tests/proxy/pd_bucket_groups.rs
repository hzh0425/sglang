// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PdBucketConfig, PdBucketGroupConfig, PolicyKind, ProxyConfig, ServerConfig,
    StaticUrlGroupConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn pd_bucket_config_with_bucket(workers: &[(&str, &str)], bucket: PdBucketConfig) -> Config {
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
            pd_bucket: Some(bucket),
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
    build_pd_bucket_ctx_with_bucket(
        workers,
        PdBucketConfig {
            groups: Vec::new(),
            short_group: "short".into(),
            long_group: "long".into(),
            prefill_long_threshold: 128,
            decode_long_threshold: 128,
        },
    )
}

fn build_pd_bucket_ctx_with_bucket(
    workers: &[(&str, &str, WorkerMode)],
    bucket: PdBucketConfig,
) -> Arc<AppContext> {
    let worker_groups = workers
        .iter()
        .map(|(url, group, _)| (*url, *group))
        .collect::<Vec<_>>();
    let cfg = pd_bucket_config_with_bucket(&worker_groups, bucket);
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
    assert_eq!(
        res.headers()
            .get("x-sglang-prefill-worker")
            .and_then(|v| v.to_str().ok()),
        Some(short_prefill.url.as_str())
    );
    assert_eq!(
        res.headers()
            .get("x-sglang-decode-worker")
            .and_then(|v| v.to_str().ok()),
        Some(short_decode.url.as_str())
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

#[tokio::test]
async fn pd_bucket_groups_route_three_ordered_buckets() {
    let small_prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let small_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let medium_prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let medium_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let large_prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let large_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let bucket = PdBucketConfig {
        groups: vec![
            PdBucketGroupConfig {
                group: "ctx0_16k".into(),
                max_tokens: 16,
            },
            PdBucketGroupConfig {
                group: "ctx16_32k".into(),
                max_tokens: 96,
            },
            PdBucketGroupConfig {
                group: "ctx32_64k".into(),
                max_tokens: 1024,
            },
        ],
        ..PdBucketConfig::default()
    };
    let ctx = build_pd_bucket_ctx_with_bucket(
        &[
            (&small_prefill.url, "ctx0_16k", WorkerMode::Prefill),
            (&small_decode.url, "ctx0_16k", WorkerMode::Decode),
            (&medium_prefill.url, "ctx16_32k", WorkerMode::Prefill),
            (&medium_decode.url, "ctx16_32k", WorkerMode::Decode),
            (&large_prefill.url, "ctx32_64k", WorkerMode::Prefill),
            (&large_decode.url, "ctx32_64k", WorkerMode::Decode),
        ],
        bucket,
    );
    let app = build_router(ctx);

    let cases = [
        (
            "session-small",
            "short bucket request".to_string(),
            "ctx0_16k",
            &small_prefill,
            &small_decode,
        ),
        (
            "session-medium",
            "middle bucket routing validation ".repeat(12),
            "ctx16_32k",
            &medium_prefill,
            &medium_decode,
        ),
        (
            "session-large",
            "long bucket routing validation ".repeat(80),
            "ctx32_64k",
            &large_prefill,
            &large_decode,
        ),
    ];

    for (key, prompt, group, prefill, decode) in cases {
        let res = app
            .clone()
            .oneshot(chat_request(key, &prompt))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers()
                .get("x-sglang-prefill-group")
                .and_then(|v| v.to_str().ok()),
            Some(group)
        );
        assert_eq!(
            res.headers()
                .get("x-sglang-decode-group")
                .and_then(|v| v.to_str().ok()),
            Some(group)
        );
        assert_eq!(
            res.headers()
                .get("x-sglang-prefill-worker")
                .and_then(|v| v.to_str().ok()),
            Some(prefill.url.as_str())
        );
        assert_eq!(
            res.headers()
                .get("x-sglang-decode-worker")
                .and_then(|v| v.to_str().ok()),
            Some(decode.url.as_str())
        );
        let _ = res.into_body().collect().await.unwrap();
    }

    wait_for_request_count(&small_prefill, 1).await;
    wait_for_request_count(&medium_prefill, 1).await;
    wait_for_request_count(&large_prefill, 1).await;
    assert_eq!(request_count(&small_decode), 1);
    assert_eq!(request_count(&medium_decode), 1);
    assert_eq!(request_count(&large_decode), 1);
}
