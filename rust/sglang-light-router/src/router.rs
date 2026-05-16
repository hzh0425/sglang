use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    str::FromStr,
    sync::Arc,
};

use anyhow::Context;
use axum::{Json, Router, extract::State, response::IntoResponse, routing::get};
use clap::Parser;
use serde::Serialize;
use thiserror::Error;
use tokio::net::TcpListener;
use url::Url;

const REQUIRED_GROUPS: [ContextGroup; 2] = [ContextGroup::Short, ContextGroup::Long];

#[derive(Parser, Debug, Clone)]
#[command(
    name = "sglang-light-router",
    about = "Lightweight SGLang short/long PD router",
    long_about = "Lightweight SGLang router for short/long prefill and decode pools."
)]
pub struct Cli {
    /// Host address for the router HTTP server.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::UNSPECIFIED))]
    pub host: IpAddr,

    /// Port for the router HTTP server.
    #[arg(long, default_value_t = 20_000)]
    pub port: u16,

    /// Enable PD routing mode. The lightweight router currently requires this flag.
    #[arg(long, default_value_t = false)]
    pub pd_disaggregation: bool,

    /// Prefill endpoint, e.g. short=http://127.0.0.1:30000,bootstrap=8998.
    #[arg(long = "prefill", value_name = "GROUP=URL,bootstrap=PORT")]
    pub prefill: Vec<String>,

    /// Decode endpoint, e.g. short=http://127.0.0.1:31000.
    #[arg(long = "decode", value_name = "GROUP=URL")]
    pub decode: Vec<String>,

    /// Requests at or above this uncached prefill length use the long prefill group.
    #[arg(long, default_value_t = 32_768)]
    pub prefill_long_threshold: usize,

    /// Requests at or above this estimated sequence length use the long decode group.
    #[arg(long, default_value_t = 32_768)]
    pub decode_long_threshold: usize,

    /// Prefill policy chain. MVP accepts `sticky,power_of_two`.
    #[arg(long, default_value = "sticky,power_of_two")]
    pub prefill_routing_policy_chain: String,

    /// Decode policy chain. MVP accepts `sticky,power_of_two`.
    #[arg(long, default_value = "sticky,power_of_two")]
    pub decode_routing_policy_chain: String,

    /// Absolute score delta required before sticky can be broken.
    #[arg(long, default_value_t = 32.0)]
    pub balance_abs_threshold: f64,

    /// Relative score multiplier required before sticky can be broken.
    #[arg(long, default_value_t = 3.0)]
    pub balance_relative_upper_bound_limit: f64,

    /// Weight for token usage penalty in load score.
    #[arg(long, default_value_t = 1.0)]
    pub load_score_token_usage_weight: f64,

    /// Include routing decisions in response headers.
    #[arg(long, default_value_t = false)]
    pub enable_routing_debug_headers: bool,

    /// Expose test-only load override endpoints. Disabled by default.
    #[arg(long, default_value_t = false)]
    pub enable_test_load_override: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextGroup {
    Short,
    Long,
}

impl ContextGroup {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Long => "long",
        }
    }
}

impl fmt::Display for ContextGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ContextGroup {
    type Err = RouterError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "short" => Ok(Self::Short),
            "long" => Ok(Self::Long),
            _ => Err(RouterError::InvalidGroup(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterRole {
    Prefill,
    Decode,
}

impl RouterRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

impl fmt::Display for RouterRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineSpec {
    pub id: String,
    pub role: RouterRole,
    pub group: ContextGroup,
    pub url: Url,
    pub bootstrap_port: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub listen_addr: SocketAddr,
    pub prefill_long_threshold: usize,
    pub decode_long_threshold: usize,
    pub prefill_routing_policy_chain: String,
    pub decode_routing_policy_chain: String,
    pub balance_abs_threshold: f64,
    pub balance_relative_upper_bound_limit: f64,
    pub load_score_token_usage_weight: f64,
    pub enable_routing_debug_headers: bool,
    pub enable_test_load_override: bool,
    pub engines: Vec<EngineSpec>,
}

impl RouterConfig {
    #[must_use]
    pub fn engines_for(&self, role: RouterRole, group: ContextGroup) -> Vec<&EngineSpec> {
        self.engines
            .iter()
            .filter(|engine| engine.role == role && engine.group == group)
            .collect()
    }

    fn validate_policy_chain(name: &str, value: &str) -> Result<(), RouterError> {
        if value == "sticky,power_of_two" {
            Ok(())
        } else {
            Err(RouterError::InvalidPolicyChain {
                name: name.to_owned(),
                value: value.to_owned(),
            })
        }
    }
}

impl TryFrom<Cli> for RouterConfig {
    type Error = RouterError;

    fn try_from(cli: Cli) -> Result<Self, Self::Error> {
        if !cli.pd_disaggregation {
            return Err(RouterError::PdDisaggregationRequired);
        }

        Self::validate_policy_chain(
            "prefill-routing-policy-chain",
            &cli.prefill_routing_policy_chain,
        )?;
        Self::validate_policy_chain(
            "decode-routing-policy-chain",
            &cli.decode_routing_policy_chain,
        )?;

        let mut engines = Vec::with_capacity(cli.prefill.len() + cli.decode.len());
        for raw in &cli.prefill {
            let index = engines
                .iter()
                .filter(|engine: &&EngineSpec| engine.role == RouterRole::Prefill)
                .count();
            engines.push(parse_endpoint(raw, RouterRole::Prefill, index)?);
        }
        for raw in &cli.decode {
            let index = engines
                .iter()
                .filter(|engine: &&EngineSpec| engine.role == RouterRole::Decode)
                .count();
            engines.push(parse_endpoint(raw, RouterRole::Decode, index)?);
        }

        validate_required_engines(&engines)?;

        Ok(Self {
            listen_addr: SocketAddr::new(cli.host, cli.port),
            prefill_long_threshold: cli.prefill_long_threshold,
            decode_long_threshold: cli.decode_long_threshold,
            prefill_routing_policy_chain: cli.prefill_routing_policy_chain,
            decode_routing_policy_chain: cli.decode_routing_policy_chain,
            balance_abs_threshold: cli.balance_abs_threshold,
            balance_relative_upper_bound_limit: cli.balance_relative_upper_bound_limit,
            load_score_token_usage_weight: cli.load_score_token_usage_weight,
            enable_routing_debug_headers: cli.enable_routing_debug_headers,
            enable_test_load_override: cli.enable_test_load_override,
            engines,
        })
    }
}

#[derive(Debug)]
struct RawEndpoint {
    group: ContextGroup,
    url: Url,
    bootstrap_port: Option<u16>,
}

fn parse_endpoint(raw: &str, role: RouterRole, index: usize) -> Result<EngineSpec, RouterError> {
    let endpoint = RawEndpoint::parse(raw)?;
    if role == RouterRole::Prefill && endpoint.bootstrap_port.is_none() {
        return Err(RouterError::MissingBootstrapPort {
            group: endpoint.group,
        });
    }

    Ok(EngineSpec {
        id: format!("{}-{}-{index}", endpoint.group, role),
        role,
        group: endpoint.group,
        url: endpoint.url,
        bootstrap_port: endpoint.bootstrap_port,
    })
}

impl RawEndpoint {
    fn parse(raw: &str) -> Result<Self, RouterError> {
        let (group, rest) = raw
            .split_once('=')
            .ok_or_else(|| RouterError::InvalidEndpoint(raw.to_owned()))?;
        let group = ContextGroup::from_str(group)?;

        let (url, bootstrap_port) = parse_endpoint_parts(rest, raw)?;
        validate_url(&url, raw)?;

        Ok(Self {
            group,
            url,
            bootstrap_port,
        })
    }
}

fn parse_endpoint_parts(
    raw_parts: &str,
    original: &str,
) -> Result<(Url, Option<u16>), RouterError> {
    let mut parts = raw_parts.split(',');
    let url_part = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| RouterError::InvalidEndpoint(original.to_owned()))?;
    let url = Url::parse(url_part).map_err(|source| RouterError::InvalidUrl {
        value: url_part.to_owned(),
        source,
    })?;

    let mut bootstrap_port = None;
    for part in parts {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| RouterError::InvalidEndpoint(original.to_owned()))?;
        if key != "bootstrap" {
            return Err(RouterError::UnknownEndpointOption(key.to_owned()));
        }
        bootstrap_port = Some(
            value
                .parse::<u16>()
                .map_err(|_| RouterError::InvalidBootstrapPort(value.to_owned()))?,
        );
    }

    Ok((url, bootstrap_port))
}

fn validate_url(url: &Url, original: &str) -> Result<(), RouterError> {
    match url.scheme() {
        "http" | "https" => {}
        scheme => return Err(RouterError::UnsupportedUrlScheme(scheme.to_owned())),
    }

    if url.host_str().is_none() || url.port_or_known_default().is_none() {
        return Err(RouterError::InvalidEndpoint(original.to_owned()));
    }

    Ok(())
}

fn validate_required_engines(engines: &[EngineSpec]) -> Result<(), RouterError> {
    let mut present = BTreeSet::new();
    for engine in engines {
        present.insert((engine.role, engine.group));
    }

    for group in REQUIRED_GROUPS {
        for role in [RouterRole::Prefill, RouterRole::Decode] {
            if !present.contains(&(role, group)) {
                return Err(RouterError::MissingRoleGroup { role, group });
            }
        }
    }

    Ok(())
}

#[derive(Error, Debug)]
pub enum RouterError {
    #[error("--pd-disaggregation is required for the lightweight router")]
    PdDisaggregationRequired,
    #[error("invalid group {0:?}; expected short or long")]
    InvalidGroup(String),
    #[error("invalid endpoint {0:?}; expected group=http://host:port[,bootstrap=port]")]
    InvalidEndpoint(String),
    #[error("invalid URL {value:?}: {source}")]
    InvalidUrl {
        value: String,
        source: url::ParseError,
    },
    #[error("unsupported URL scheme {0:?}; expected http or https")]
    UnsupportedUrlScheme(String),
    #[error("unknown endpoint option {0:?}")]
    UnknownEndpointOption(String),
    #[error("invalid bootstrap port {0:?}")]
    InvalidBootstrapPort(String),
    #[error("prefill group {group} requires bootstrap port")]
    MissingBootstrapPort { group: ContextGroup },
    #[error("missing {role} engine for {group} group")]
    MissingRoleGroup {
        role: RouterRole,
        group: ContextGroup,
    },
    #[error("invalid {name}: {value:?}; only sticky,power_of_two is supported")]
    InvalidPolicyChain { name: String, value: String },
}

#[derive(Clone)]
pub struct AppState {
    config: Arc<RouterConfig>,
}

impl AppState {
    #[must_use]
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .with_state(state)
}

/// Serve the router using the supplied local configuration.
///
/// # Errors
///
/// Returns an error if the TCP listener cannot bind or if the axum server exits
/// with an error.
pub async fn serve_config(config: RouterConfig) -> anyhow::Result<()> {
    let listen_addr = config.listen_addr;
    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("failed to bind router on {listen_addr}"))?;
    tracing::info!("sglang-light-router listening on {}", listen_addr);
    axum::serve(listener, app(AppState::new(config)))
        .await
        .context("router server failed")
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    readiness: &'static str,
    engines_total: usize,
    engines_by_role: BTreeMap<&'static str, usize>,
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let mut engines_by_role = BTreeMap::new();
    for role in [RouterRole::Prefill, RouterRole::Decode] {
        let count = state
            .config
            .engines
            .iter()
            .filter(|engine| engine.role == role)
            .count();
        engines_by_role.insert(role.as_str(), count);
    }

    Json(HealthResponse {
        status: "ok",
        readiness: "local_config",
        engines_total: state.config.engines.len(),
        engines_by_role,
    })
}
