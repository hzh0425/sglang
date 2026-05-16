use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use clap::{CommandFactory, Parser};
use sglang_light_router::{AppState, Cli, ContextGroup, RouterConfig, RouterRole, app};
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
