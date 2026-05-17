use std::sync::{Arc, Mutex};

use axum::{
    Json,
    body::{Body, to_bytes},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
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

fn mock_config_for_workers(
    short_prefill_url: &str,
    short_decode_url: &str,
    long_prefill_url: &str,
    long_decode_url: &str,
) -> RouterConfig {
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
        format!("short={short_prefill_url},bootstrap=8998"),
        "--decode".to_owned(),
        format!("short={short_decode_url}"),
        "--prefill".to_owned(),
        format!("long={long_prefill_url},bootstrap=8999"),
        "--decode".to_owned(),
        format!("long={long_decode_url}"),
    ];
    let cli = Cli::try_parse_from(args).expect("valid mock CLI args");
    RouterConfig::try_from(cli).expect("valid mock router config")
}

fn mock_config(short_decode_url: &str, long_decode_url: &str) -> RouterConfig {
    mock_config_for_workers(
        "http://127.0.0.1:30000",
        short_decode_url,
        "http://127.0.0.1:30001",
        long_decode_url,
    )
}

#[derive(Clone)]
struct MockDecodeState {
    sender: Arc<Mutex<Option<oneshot::Sender<Value>>>>,
    status: StatusCode,
}

#[derive(Clone)]
struct MockMetadataState {
    sender: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    status: StatusCode,
}

#[derive(Debug, Eq, PartialEq)]
struct CapturedRequestHeaders {
    authorization: Option<String>,
    cookie: Option<String>,
}

#[derive(Clone)]
struct MockHeaderState {
    sender: Arc<Mutex<Option<oneshot::Sender<CapturedRequestHeaders>>>>,
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
        .route("/generate", post(mock_generate))
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

async fn spawn_metadata_mock(
    status: StatusCode,
) -> (String, oneshot::Receiver<()>, JoinHandle<()>) {
    let (sender, receiver) = oneshot::channel();
    let state = MockMetadataState {
        sender: Arc::new(Mutex::new(Some(sender))),
        status,
    };
    let mock_app = axum::Router::new()
        .route("/get_model_info", get(mock_model_info))
        .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock metadata server should bind");
    let addr = listener
        .local_addr()
        .expect("mock metadata address should exist");
    let handle = tokio::spawn(async move {
        axum::serve(listener, mock_app)
            .await
            .expect("mock metadata server should run");
    });

    (format!("http://{addr}"), receiver, handle)
}

async fn spawn_header_mock(
    status: StatusCode,
) -> (
    String,
    oneshot::Receiver<CapturedRequestHeaders>,
    JoinHandle<()>,
) {
    let (sender, receiver) = oneshot::channel();
    let state = MockHeaderState {
        sender: Arc::new(Mutex::new(Some(sender))),
        status,
    };
    let mock_app = axum::Router::new()
        .route("/v1/chat/completions", post(mock_header_chat))
        .route("/generate", post(mock_header_generate))
        .route("/v1/loads", get(mock_loads))
        .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock header server should bind");
    let addr = listener
        .local_addr()
        .expect("mock header address should exist");
    let handle = tokio::spawn(async move {
        axum::serve(listener, mock_app)
            .await
            .expect("mock header server should run");
    });

    (format!("http://{addr}"), receiver, handle)
}

async fn mock_chat(
    State(state): State<MockDecodeState>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    capture_mock_body(&state, body);

    (state.status, Json(json!({ "mock": "chat" })))
}

async fn mock_generate(
    State(state): State<MockDecodeState>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    capture_mock_body(&state, body);

    (state.status, Json(json!({ "mock": "generate" })))
}

fn capture_mock_body(state: &MockDecodeState, body: Value) {
    if let Some(sender) = state
        .sender
        .lock()
        .expect("mock sender lock should not be poisoned")
        .take()
    {
        let _ = sender.send(body);
    }
}

async fn mock_model_info(State(state): State<MockMetadataState>) -> (StatusCode, Json<Value>) {
    if let Some(sender) = state
        .sender
        .lock()
        .expect("mock metadata sender lock should not be poisoned")
        .take()
    {
        let _ = sender.send(());
    }

    (
        state.status,
        Json(json!({
            "model_path": "mock-model",
            "tokenizer_path": "mock-tokenizer",
            "is_generation": true
        })),
    )
}

async fn mock_header_chat(
    State(state): State<MockHeaderState>,
    headers: HeaderMap,
    Json(_body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    capture_mock_headers(&state, &headers);

    (state.status, Json(json!({ "mock": "chat" })))
}

async fn mock_header_generate(
    State(state): State<MockHeaderState>,
    headers: HeaderMap,
    Json(_body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    capture_mock_headers(&state, &headers);

    (state.status, Json(json!({ "mock": "generate" })))
}

fn capture_mock_headers(state: &MockHeaderState, headers: &HeaderMap) {
    if let Some(sender) = state
        .sender
        .lock()
        .expect("mock header sender lock should not be poisoned")
        .take()
    {
        let _ = sender.send(CapturedRequestHeaders {
            authorization: header_to_string(headers, "authorization"),
            cookie: header_to_string(headers, "cookie"),
        });
    }
}

fn header_to_string(headers: &HeaderMap, name: &'static str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
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

fn generate_http_request(text: &str, max_new_tokens: usize) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .header("x-sglang-routing-key", "mock-session")
        .body(Body::from(
            json!({
                "text": text,
                "sampling_params": {"max_new_tokens": max_new_tokens}
            })
            .to_string(),
        ))
        .expect("request should build")
}

fn generate_input_ids_request(token_count: usize, max_new_tokens: usize) -> Request<Body> {
    let input_ids: Vec<usize> = (0..token_count).collect();
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .header("x-sglang-routing-key", "mock-session")
        .body(Body::from(
            json!({
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": max_new_tokens}
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
async fn get_model_info_proxies_to_short_decode() {
    let (decode_url, metadata_hit, handle) = spawn_metadata_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config(
        &decode_url,
        "http://127.0.0.1:39999",
    )));

    let response = router
        .oneshot(
            Request::builder()
                .uri("/get_model_info")
                .body(Body::empty())
                .expect("metadata request should build"),
        )
        .await
        .expect("metadata request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    let metadata = response_json(response).await;
    assert_eq!(metadata["model_path"], "mock-model");
    metadata_hit
        .await
        .expect("short decode should receive metadata request");

    handle.abort();
}

#[tokio::test]
async fn chat_proxy_injects_bootstrap_and_debug_headers() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
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
    assert_eq!(json["mock"], "chat");

    let forwarded_body = forwarded_body
        .await
        .expect("mock decode should receive forwarded body");
    let prefill_body = prefill_body
        .await
        .expect("mock prefill should receive forwarded body");
    assert_eq!(forwarded_body["bootstrap_host"], "127.0.0.1");
    assert_eq!(forwarded_body["bootstrap_port"], 8998);
    assert!(forwarded_body["bootstrap_room"].as_u64().is_some());
    assert_eq!(
        prefill_body["bootstrap_host"],
        forwarded_body["bootstrap_host"]
    );
    assert_eq!(
        prefill_body["bootstrap_port"],
        forwarded_body["bootstrap_port"]
    );
    assert_eq!(
        prefill_body["bootstrap_room"],
        forwarded_body["bootstrap_room"]
    );

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
    prefill_handle.abort();
}

#[tokio::test]
async fn chat_proxy_forwards_authorization_to_prefill_and_decode() {
    let (prefill_url, prefill_headers, prefill_handle) = spawn_header_mock(StatusCode::OK).await;
    let (decode_url, decode_headers, decode_handle) = spawn_header_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
        "http://127.0.0.1:39999",
    )));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test-token")
        .header("cookie", "session=do-not-forward")
        .body(Body::from(
            json!({
                "model": "mock",
                "messages": [{"role": "user", "content": "short auth request"}],
                "max_tokens": 4
            })
            .to_string(),
        ))
        .expect("request should build");

    let response = router
        .oneshot(request)
        .await
        .expect("chat request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        prefill_headers
            .await
            .expect("prefill should receive request headers"),
        CapturedRequestHeaders {
            authorization: Some("Bearer test-token".to_owned()),
            cookie: None,
        }
    );
    assert_eq!(
        decode_headers
            .await
            .expect("decode should receive request headers"),
        CapturedRequestHeaders {
            authorization: Some("Bearer test-token".to_owned()),
            cookie: None,
        }
    );

    prefill_handle.abort();
    decode_handle.abort();
}

#[tokio::test]
async fn generate_proxy_injects_bootstrap_and_debug_headers() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
        "http://127.0.0.1:39999",
    )));

    let response = router
        .clone()
        .oneshot(generate_http_request("short native request", 4))
        .await
        .expect("generate request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-sglang-prefill-group"),
        Some(&axum::http::HeaderValue::from_static("short")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-worker"),
        Some(&axum::http::HeaderValue::from_static("short-decode-0")),
    );

    let response_body = response_json(response).await;
    assert_eq!(response_body["mock"], "generate");

    let forwarded_body = forwarded_body
        .await
        .expect("mock decode should receive forwarded generate body");
    let prefill_body = prefill_body
        .await
        .expect("mock prefill should receive forwarded generate body");
    assert_eq!(forwarded_body["bootstrap_host"], "127.0.0.1");
    assert_eq!(forwarded_body["bootstrap_port"], 8998);
    assert!(forwarded_body["bootstrap_room"].as_u64().is_some());
    assert_eq!(forwarded_body["sampling_params"]["max_new_tokens"], 4);
    assert_eq!(
        prefill_body["bootstrap_room"],
        forwarded_body["bootstrap_room"]
    );
    assert_eq!(prefill_body["sampling_params"]["max_new_tokens"], 4);

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

    handle.abort();
    prefill_handle.abort();
}

#[tokio::test]
async fn generate_long_dispatch_uses_text_and_sampling_params() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        "http://127.0.0.1:30000",
        "http://127.0.0.1:39998",
        &prefill_url,
        &decode_url,
    )));

    let response = router
        .clone()
        .oneshot(generate_http_request(
            "one two three four five six seven eight",
            8,
        ))
        .await
        .expect("generate request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-sglang-prefill-group"),
        Some(&axum::http::HeaderValue::from_static("long")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-group"),
        Some(&axum::http::HeaderValue::from_static("long")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-worker"),
        Some(&axum::http::HeaderValue::from_static("long-decode-1")),
    );

    let response_body = response_json(response).await;
    assert_eq!(response_body["mock"], "generate");
    let forwarded_body = forwarded_body
        .await
        .expect("mock long decode should receive forwarded generate body");
    let prefill_body = prefill_body
        .await
        .expect("mock long prefill should receive forwarded generate body");
    assert_eq!(forwarded_body["bootstrap_port"], 8999);
    assert_eq!(prefill_body["bootstrap_port"], 8999);
    assert_eq!(
        prefill_body["bootstrap_room"],
        forwarded_body["bootstrap_room"]
    );

    handle.abort();
    prefill_handle.abort();
}

#[tokio::test]
async fn generate_long_dispatch_uses_input_ids_length() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        "http://127.0.0.1:30000",
        "http://127.0.0.1:39998",
        &prefill_url,
        &decode_url,
    )));

    let response = router
        .clone()
        .oneshot(generate_input_ids_request(8, 8))
        .await
        .expect("tokenized generate request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-sglang-prefill-group"),
        Some(&axum::http::HeaderValue::from_static("long")),
    );
    assert_eq!(
        response.headers().get("x-sglang-decode-group"),
        Some(&axum::http::HeaderValue::from_static("long")),
    );

    let forwarded_body = forwarded_body
        .await
        .expect("mock long decode should receive forwarded generate body");
    let prefill_body = prefill_body
        .await
        .expect("mock long prefill should receive forwarded generate body");
    assert_eq!(
        forwarded_body["input_ids"].as_array().map(Vec::len),
        Some(8)
    );
    assert_eq!(prefill_body["input_ids"].as_array().map(Vec::len), Some(8));
    assert_eq!(forwarded_body["bootstrap_port"], 8999);
    assert_eq!(prefill_body["bootstrap_port"], 8999);

    handle.abort();
    prefill_handle.abort();
}

#[tokio::test]
async fn upstream_error_response_does_not_leak_local_load() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) =
        spawn_decode_mock(StatusCode::INTERNAL_SERVER_ERROR).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
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
    let _ = prefill_body
        .await
        .expect("mock prefill should receive forwarded body");

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
    prefill_handle.abort();
}

#[tokio::test]
async fn generate_upstream_error_response_does_not_leak_local_load() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) =
        spawn_decode_mock(StatusCode::INTERNAL_SERVER_ERROR).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
        "http://127.0.0.1:39999",
    )));

    let response = router
        .clone()
        .oneshot(generate_http_request("short native request", 4))
        .await
        .expect("generate request should reach router");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let _ = response_json(response).await;
    let _ = forwarded_body
        .await
        .expect("mock decode should receive forwarded generate body");
    let _ = prefill_body
        .await
        .expect("mock prefill should receive forwarded generate body");

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
    prefill_handle.abort();
}

#[tokio::test]
async fn dropped_proxy_response_body_does_not_leak_local_load() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
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
    let _ = prefill_body
        .await
        .expect("mock prefill should receive forwarded body");

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
    prefill_handle.abort();
}

#[tokio::test]
async fn dropped_generate_response_body_does_not_leak_local_load() {
    let (prefill_url, prefill_body, prefill_handle) = spawn_decode_mock(StatusCode::OK).await;
    let (decode_url, forwarded_body, handle) = spawn_decode_mock(StatusCode::OK).await;
    let router = app(AppState::new(mock_config_for_workers(
        &prefill_url,
        &decode_url,
        "http://127.0.0.1:30001",
        "http://127.0.0.1:39999",
    )));

    let response = router
        .clone()
        .oneshot(generate_http_request("short native request", 4))
        .await
        .expect("generate request should reach router");

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
    let _ = forwarded_body
        .await
        .expect("mock decode should receive forwarded generate body");
    let _ = prefill_body
        .await
        .expect("mock prefill should receive forwarded generate body");

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
    prefill_handle.abort();
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
