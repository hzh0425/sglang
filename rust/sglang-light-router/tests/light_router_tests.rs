use std::sync::{Arc, Mutex};

use axum::{
    Json,
    body::{Body, to_bytes},
    extract::State,
    http::{Request, StatusCode},
    routing::{get, post},
};
use clap::{CommandFactory, Parser};
use serde_json::{Value, json};
use sglang_light_router::{AppState, Cli, ContextGroup, RouterConfig, RouterRole, app};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};
use tower::ServiceExt;

fn valid_args() -> Vec<&'static str> {
    vec![
        "sglang-light-router",
        "--pd-disaggregation",
        "--prefill",
        "short=http://127.0.0.1:30000,bootstrap=8998",
        "--decode",
        "short=http://127.0.0.1:31000",
        "--prefill",
        "long=http://127.0.0.1:30001,bootstrap=8999",
        "--decode",
        "long=http://127.0.0.1:31001",
    ]
}

fn valid_config() -> RouterConfig {
    let cli = Cli::try_parse_from(valid_args()).expect("valid CLI args");
    RouterConfig::try_from(cli).expect("valid router config")
}

fn mock_config(short_decode_url: &str, long_decode_url: &str) -> RouterConfig {
    let args = vec![
        "sglang-light-router".to_owned(),
        "--pd-disaggregation".to_owned(),
        "--enable-routing-debug-headers".to_owned(),
        "--enable-test-load-override".to_owned(),
        "--prefill-long-threshold".to_owned(),
        "8".to_owned(),
        "--decode-long-threshold".to_owned(),
        "16".to_owned(),
        "--prefill".to_owned(),
        "short=http://127.0.0.1:30000,bootstrap=8998".to_owned(),
        "--decode".to_owned(),
        format!("short={short_decode_url}"),
        "--prefill".to_owned(),
        "long=http://127.0.0.1:30001,bootstrap=8999".to_owned(),
        "--decode".to_owned(),
        format!("long={long_decode_url}"),
    ];
    let cli = Cli::try_parse_from(args).expect("valid mock CLI args");
    RouterConfig::try_from(cli).expect("valid mock router config")
}

#[derive(Clone)]
struct MockDecodeState {
    sender: Arc<Mutex<Option<oneshot::Sender<Value>>>>,
    status: StatusCode,
}

async fn spawn_decode_mock(
    status: StatusCode,
) -> (String, oneshot::Receiver<Value>, JoinHandle<()>) {
    let (sender, receiver) = oneshot::channel();
    let state = MockDecodeState {
        sender: Arc::new(Mutex::new(Some(sender))),
        status,
    };
    let mock_app = axum::Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .route("/v1/loads", get(mock_loads))
        .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock server should bind");
    let addr = listener.local_addr().expect("mock address should exist");
    let handle = tokio::spawn(async move {
        axum::serve(listener, mock_app)
            .await
            .expect("mock server should run");
    });

    (format!("http://{addr}"), receiver, handle)
}

async fn mock_chat(
    State(state): State<MockDecodeState>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    if let Some(sender) = state
        .sender
        .lock()
        .expect("mock sender lock should not be poisoned")
        .take()
    {
        let _ = sender.send(body);
    }

    (state.status, Json(json!({ "mock": true })))
}

async fn mock_loads() -> Json<Value> {
    Json(json!({
        "aggregate": {
            "total_running_reqs": 2,
            "total_used_tokens": 64,
            "total_tokens": 512,
            "avg_token_usage": 0.125
        }
    }))
}

fn chat_http_request(prompt: &str, max_tokens: usize) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-sglang-routing-key", "mock-session")
        .body(Body::from(
            json!({
                "model": "mock",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            })
            .to_string(),
        ))
        .expect("request should build")
}

async fn response_json(response: axum::response::Response) -> Value {
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("response body should be readable");
    serde_json::from_slice(&body).expect("response body should be JSON")
}

#[test]
fn help_exposes_light_router_options() {
    let help = Cli::command().render_long_help().to_string();

    assert!(help.contains("--prefill"));
    assert!(help.contains("--decode"));
    assert!(help.contains("--prefill-long-threshold"));
    assert!(help.contains("--enable-routing-debug-headers"));
}

#[test]
fn valid_short_long_pool_configures_engines() {
    let config = valid_config();

    assert_eq!(config.engines.len(), 4);
    assert_eq!(
        config
            .engines_for(RouterRole::Prefill, ContextGroup::Short)
            .first()
            .and_then(|engine| engine.bootstrap_port),
        Some(8998),
    );
    assert_eq!(
        config
            .engines_for(RouterRole::Decode, ContextGroup::Long)
            .first()
            .map(|engine| engine.url.as_str()),
        Some("http://127.0.0.1:31001/"),
    );
}

#[test]
fn missing_group_role_fails_validation() {
    let mut args = valid_args();
    args.truncate(args.len() - 2);
    let cli = Cli::try_parse_from(args).expect("CLI parsing should defer role validation");
    let error = RouterConfig::try_from(cli).expect_err("missing long decode must fail");

    assert!(
        error
            .to_string()
            .contains("missing decode engine for long group")
    );
}

#[test]
fn invalid_group_fails_validation() {
    let mut args = valid_args();
    args[3] = "medium=http://127.0.0.1:30000,bootstrap=8998";
    let cli = Cli::try_parse_from(args).expect("CLI parsing should defer group validation");
    let error = RouterConfig::try_from(cli).expect_err("invalid group must fail");

    assert!(error.to_string().contains("invalid group"));
}

#[test]
fn prefill_requires_bootstrap_port() {
    let mut args = valid_args();
    args[3] = "short=http://127.0.0.1:30000";
    let cli = Cli::try_parse_from(args).expect("CLI parsing should defer endpoint validation");
    let error = RouterConfig::try_from(cli).expect_err("missing bootstrap must fail");

    assert!(
        error
            .to_string()
            .contains("prefill group short requires bootstrap port"),
    );
}

#[tokio::test]
async fn health_reports_local_config_readiness() {
    let response = app(AppState::new(valid_config()))
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("health request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024)
        .await
        .expect("health body should be readable");
    let json: serde_json::Value =
        serde_json::from_slice(&body).expect("health body should be JSON");

    assert_eq!(json["status"], "ok");
    assert_eq!(json["readiness"], "local_config");
    assert_eq!(json["engines_total"], 4);
}

#[tokio::test]
async fn chat_proxy_injects_bootstrap_and_debug_headers() {
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config(
        &decode_url,
        "http://127.0.0.1:39999",
    )));

    let response = router
        .clone()
        .oneshot(chat_http_request("short validation request", 4))
        .await
        .expect("chat request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-sglang-prefill-group"),
        Some(&axum::http::HeaderValue::from_static("short")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-group"),
        Some(&axum::http::HeaderValue::from_static("short")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-worker"),
        Some(&axum::http::HeaderValue::from_static("short-decode-0")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-policy-branch"),
        Some(&axum::http::HeaderValue::from_static("load_based_fallback")),
    );

    let json = response_json(response).await;
    assert_eq!(json["mock"], true);

    let forwarded_body = forwarded_body
        .await
        .expect("mock decode should receive forwarded body");
    assert_eq!(forwarded_body["bootstrap_host"], "127.0.0.1");
    assert_eq!(forwarded_body["bootstrap_port"], 8998);
    assert!(forwarded_body["bootstrap_room"].as_u64().is_some());

    let stats_response = router
        .oneshot(
            Request::builder()
                .uri("/debug/routing_stats")
                .body(Body::empty())
                .expect("stats request should build"),
        )
        .await
        .expect("stats request should succeed");
    let stats_json = response_json(stats_response).await;

    assert_eq!(stats_json["route_counts"]["prefill=short,decode=short"], 1);
    assert_eq!(
        stats_json["policy_counts"]["decode"]["load_based_fallback"],
        1
    );
    let short_decode = stats_json["engines"]
        .as_array()
        .expect("engines should be an array")
        .iter()
        .find(|engine| engine["id"] == "short-decode-0")
        .expect("short decode stats should exist");
    assert_eq!(short_decode["local_dispatch_total"], 1);
    assert_eq!(short_decode["in_flight_requests"], 0);

    handle.abort();
}

#[tokio::test]
async fn upstream_error_response_does_not_leak_local_load() {
    let (decode_url, forwarded_body, handle) =
        spawn_decode_mock(StatusCode::INTERNAL_SERVER_ERROR).await;
    let router = app(AppState::new(mock_config(
        &decode_url,
        "http://127.0.0.1:39999",
    )));

    let response = router
        .clone()
        .oneshot(chat_http_request("short validation request", 4))
        .await
        .expect("chat request should reach router");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let _ = response_json(response).await;
    let _ = forwarded_body
        .await
        .expect("mock decode should receive forwarded body");

    let stats_response = router
        .oneshot(
            Request::builder()
                .uri("/debug/routing_stats")
                .body(Body::empty())
                .expect("stats request should build"),
        )
        .await
        .expect("stats request should succeed");
    let stats_json = response_json(stats_response).await;
    for engine in stats_json["engines"]
        .as_array()
        .expect("engines should be an array")
    {
        assert_eq!(engine["in_flight_requests"], 0);
    }

    handle.abort();
}

#[tokio::test]
async fn dropped_proxy_response_body_does_not_leak_local_load() {
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config(
        &decode_url,
        "http://127.0.0.1:39999",
    )));

    let response = router
        .clone()
        .oneshot(chat_http_request("short validation request", 4))
        .await
        .expect("chat request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
    let _ = forwarded_body
        .await
        .expect("mock decode should receive forwarded body");

    let stats_response = router
        .oneshot(
            Request::builder()
                .uri("/debug/routing_stats")
                .body(Body::empty())
                .expect("stats request should build"),
        )
        .await
        .expect("stats request should succeed");
    let stats_json = response_json(stats_response).await;
    for engine in stats_json["engines"]
        .as_array()
        .expect("engines should be an array")
    {
        assert_eq!(engine["in_flight_requests"], 0);
    }

    handle.abort();
}

#[tokio::test]
async fn load_polling_updates_stats_snapshot() {
    let (engine_url, _forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let state = AppState::new(mock_config(&engine_url, "http://127.0.0.1:39999"));
    state.poll_loads_once().await;
    let router = app(state);

    let stats_response = router
        .oneshot(
            Request::builder()
                .uri("/debug/routing_stats")
                .body(Body::empty())
                .expect("stats request should build"),
        )
        .await
        .expect("stats request should succeed");
    let stats_json = response_json(stats_response).await;
    let short_decode = stats_json["engines"]
        .as_array()
        .expect("engines should be an array")
        .iter()
        .find(|engine| engine["id"] == "short-decode-0")
        .expect("short decode stats should exist");

    assert_eq!(short_decode["last_load_poll_ok"], true);
    assert_eq!(short_decode["reported_total_tokens"], 64);
    assert_eq!(short_decode["reported_decode_batch_size"], 2);
    assert_eq!(short_decode["reported_token_usage"], 0.125);

    handle.abort();
}
