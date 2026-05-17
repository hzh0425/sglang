use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    pin::Pin,
    str::FromStr,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use anyhow::Context as AnyhowContext;
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use clap::Parser;
use http_body::{Frame, SizeHint};
use num_traits::ToPrimitive;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::net::TcpListener;
use url::Url;

const REQUIRED_GROUPS: [ContextGroup; 2] = [ContextGroup::Short, ContextGroup::Long];
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const LOAD_USAGE_EPSILON: f64 = 0.000_001;
const LOAD_POLL_INTERVAL: Duration = Duration::from_secs(5);

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
    #[error("no healthy {role} engine for {group} group")]
    NoHealthyEngine {
        role: RouterRole,
        group: ContextGroup,
    },
    #[error("request body must be a JSON object")]
    RequestBodyNotObject,
    #[error("selected prefill engine {engine_id} has no bootstrap port")]
    MissingSelectedBootstrapPort { engine_id: String },
    #[error("selected prefill engine {engine_id} has no host")]
    MissingSelectedBootstrapHost { engine_id: String },
    #[error("failed to build upstream URL for {engine_id}: {source}")]
    UpstreamUrl {
        engine_id: String,
        source: url::ParseError,
    },
    #[error("upstream request to {engine_id} failed: {source}")]
    UpstreamRequest {
        engine_id: String,
        source: reqwest::Error,
    },
    #[error("unknown engine id {0:?}")]
    UnknownEngine(String),
}

impl RouterError {
    const fn status_code(&self) -> StatusCode {
        match self {
            Self::RequestBodyNotObject => StatusCode::BAD_REQUEST,
            Self::NoHealthyEngine { .. } | Self::UnknownEngine(_) => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::UpstreamRequest { .. } => StatusCode::BAD_GATEWAY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for RouterError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = Json(json!({ "error": self.to_string() }));
        (status, body).into_response()
    }
}

#[derive(Debug, Clone)]
struct RoutingFeatures {
    routing_key: Option<String>,
    prompt_tokens: usize,
    uncached_prefill_tokens: usize,
    max_new_tokens: usize,
    estimated_sequence_length: usize,
}

impl RoutingFeatures {
    fn from_request(headers: &HeaderMap, body: &Value) -> Self {
        let prompt_tokens = estimate_prompt_tokens(body);
        let max_new_tokens = max_new_tokens(body);

        Self {
            routing_key: extract_routing_key(headers),
            prompt_tokens,
            uncached_prefill_tokens: prompt_tokens,
            max_new_tokens,
            estimated_sequence_length: prompt_tokens.saturating_add(max_new_tokens),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ReportedLoad {
    total_tokens: Option<usize>,
    token_usage: Option<f64>,
    decode_batch_size: Option<usize>,
    last_ok: bool,
}

#[derive(Debug)]
struct EngineState {
    spec: EngineSpec,
    healthy: AtomicBool,
    in_flight_requests: AtomicUsize,
    in_flight_prefill_tokens: AtomicUsize,
    in_flight_decode_requests: AtomicUsize,
    local_dispatch_total: AtomicUsize,
    reported: RwLock<ReportedLoad>,
}

impl EngineState {
    fn new(spec: EngineSpec) -> Self {
        Self {
            spec,
            healthy: AtomicBool::new(true),
            in_flight_requests: AtomicUsize::new(0),
            in_flight_prefill_tokens: AtomicUsize::new(0),
            in_flight_decode_requests: AtomicUsize::new(0),
            local_dispatch_total: AtomicUsize::new(0),
            reported: RwLock::new(ReportedLoad::default()),
        }
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }

    fn local_work(&self, role: RouterRole) -> usize {
        match role {
            RouterRole::Prefill => self.in_flight_prefill_tokens.load(Ordering::Relaxed),
            RouterRole::Decode => self.in_flight_decode_requests.load(Ordering::Relaxed),
        }
    }

    fn reported(&self) -> ReportedLoad {
        self.reported
            .read()
            .expect("reported load lock poisoned")
            .clone()
    }

    fn set_reported(&self, reported: ReportedLoad) {
        *self.reported.write().expect("reported load lock poisoned") = reported;
    }

    fn mark_load_poll_failed(&self) {
        self.reported
            .write()
            .expect("reported load lock poisoned")
            .last_ok = false;
    }

    fn set_local_work_for_test(&self, work: usize) {
        match self.spec.role {
            RouterRole::Prefill => self.in_flight_prefill_tokens.store(work, Ordering::Relaxed),
            RouterRole::Decode => self
                .in_flight_decode_requests
                .store(work, Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct StickyKey {
    routing_key: String,
    role: RouterRole,
    group: ContextGroup,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PolicyBranch {
    StickyHit,
    LoadBasedFallback,
}

impl PolicyBranch {
    const fn as_str(self) -> &'static str {
        match self {
            Self::StickyHit => "sticky_hit",
            Self::LoadBasedFallback => "load_based_fallback",
        }
    }
}

#[derive(Debug, Clone)]
struct EngineSelection {
    engine: Arc<EngineState>,
    role: RouterRole,
    group: ContextGroup,
    branch: PolicyBranch,
}

#[derive(Debug, Clone)]
struct RoutePlan {
    features: RoutingFeatures,
    prefill_group: ContextGroup,
    decode_group: ContextGroup,
    prefill: EngineSelection,
    decode: EngineSelection,
}

#[derive(Debug)]
struct RouterRuntime {
    config: RouterConfig,
    client: reqwest::Client,
    engines: Vec<Arc<EngineState>>,
    engines_by_id: BTreeMap<String, Arc<EngineState>>,
    engines_by_role_group: BTreeMap<(RouterRole, ContextGroup), Vec<Arc<EngineState>>>,
    sticky: RwLock<BTreeMap<StickyKey, String>>,
    route_counts: Mutex<BTreeMap<String, usize>>,
    policy_counts: Mutex<BTreeMap<String, BTreeMap<String, usize>>>,
    bootstrap_room: AtomicU64,
}

impl RouterRuntime {
    fn new(config: RouterConfig) -> Self {
        let mut engines = Vec::with_capacity(config.engines.len());
        let mut engines_by_id = BTreeMap::new();
        let mut engines_by_role_group: BTreeMap<_, Vec<_>> = BTreeMap::new();

        for spec in &config.engines {
            let engine = Arc::new(EngineState::new(spec.clone()));
            engines_by_id.insert(spec.id.clone(), Arc::clone(&engine));
            engines_by_role_group
                .entry((spec.role, spec.group))
                .or_default()
                .push(Arc::clone(&engine));
            engines.push(engine);
        }

        Self {
            config,
            client: reqwest::Client::new(),
            engines,
            engines_by_id,
            engines_by_role_group,
            sticky: RwLock::new(BTreeMap::new()),
            route_counts: Mutex::new(BTreeMap::new()),
            policy_counts: Mutex::new(BTreeMap::new()),
            bootstrap_room: AtomicU64::new(1),
        }
    }

    fn healthy_engines(
        &self,
        role: RouterRole,
        group: ContextGroup,
    ) -> Result<Vec<Arc<EngineState>>, RouterError> {
        let healthy =
            self.engines_by_role_group
                .get(&(role, group))
                .map_or_else(Vec::new, |engines| {
                    engines
                        .iter()
                        .filter(|engine| engine.is_healthy())
                        .cloned()
                        .collect()
                });

        if healthy.is_empty() {
            Err(RouterError::NoHealthyEngine { role, group })
        } else {
            Ok(healthy)
        }
    }

    fn select_without_commit(
        &self,
        role: RouterRole,
        group: ContextGroup,
        features: &RoutingFeatures,
    ) -> Result<EngineSelection, RouterError> {
        let candidates = self.healthy_engines(role, group)?;
        let reference = self.load_based_reference(role, &candidates);
        let sticky = features
            .routing_key
            .as_ref()
            .and_then(|routing_key| self.sticky_candidate(routing_key, role, group, &candidates));

        let (engine, branch) = if sticky
            .as_ref()
            .is_some_and(|preferred| self.sticky_is_acceptable(role, preferred, &reference))
        {
            (
                sticky.expect("sticky candidate exists"),
                PolicyBranch::StickyHit,
            )
        } else {
            (reference, PolicyBranch::LoadBasedFallback)
        };

        Ok(EngineSelection {
            engine,
            role,
            group,
            branch,
        })
    }

    fn load_based_reference(
        &self,
        role: RouterRole,
        candidates: &[Arc<EngineState>],
    ) -> Arc<EngineState> {
        if let [only] = candidates {
            return Arc::clone(only);
        }

        let mut rng = rand::thread_rng();
        let sampled = candidates
            .choose_multiple(&mut rng, 2)
            .cloned()
            .collect::<Vec<_>>();
        let first = &sampled[0];
        let second = &sampled[1];
        let (first_score, second_score) = self.score_pair(role, first, second);
        if first_score <= second_score {
            Arc::clone(first)
        } else {
            Arc::clone(second)
        }
    }

    fn sticky_candidate(
        &self,
        routing_key: &str,
        role: RouterRole,
        group: ContextGroup,
        candidates: &[Arc<EngineState>],
    ) -> Option<Arc<EngineState>> {
        let sticky_key = StickyKey {
            routing_key: routing_key.to_owned(),
            role,
            group,
        };
        let engine_id = self
            .sticky
            .read()
            .expect("sticky table lock poisoned")
            .get(&sticky_key)
            .cloned()?;

        candidates
            .iter()
            .find(|engine| engine.spec.id == engine_id)
            .cloned()
    }

    fn sticky_is_acceptable(
        &self,
        role: RouterRole,
        preferred: &Arc<EngineState>,
        reference: &Arc<EngineState>,
    ) -> bool {
        let (preferred_score, reference_score) = self.score_pair(role, preferred, reference);
        let abs_bad = preferred_score - reference_score > self.config.balance_abs_threshold;
        let rel_bad =
            preferred_score > reference_score * self.config.balance_relative_upper_bound_limit;

        !(abs_bad && rel_bad)
    }

    fn score_pair(
        &self,
        role: RouterRole,
        first: &Arc<EngineState>,
        second: &Arc<EngineState>,
    ) -> (f64, f64) {
        let first_reported = first.reported();
        let second_reported = second.reported();
        let use_reported = can_use_reported_score(role, &first_reported)
            && can_use_reported_score(role, &second_reported);

        if use_reported {
            (
                score_for_engine(
                    role,
                    first,
                    &first_reported,
                    self.config.load_score_token_usage_weight,
                    true,
                ),
                score_for_engine(
                    role,
                    second,
                    &second_reported,
                    self.config.load_score_token_usage_weight,
                    true,
                ),
            )
        } else {
            (
                load_score(
                    first.local_work(role),
                    None,
                    self.config.load_score_token_usage_weight,
                ),
                load_score(
                    second.local_work(role),
                    None,
                    self.config.load_score_token_usage_weight,
                ),
            )
        }
    }

    fn commit_plan(&self, plan: &RoutePlan) {
        self.record_route(plan.prefill_group, plan.decode_group);
        let routing_key = plan.features.routing_key.as_ref();
        self.commit_selection(routing_key, &plan.prefill);
        self.commit_selection(routing_key, &plan.decode);
    }

    fn commit_selection(&self, routing_key: Option<&String>, selection: &EngineSelection) {
        if let Some(key) = routing_key {
            self.sticky
                .write()
                .expect("sticky table lock poisoned")
                .insert(
                    StickyKey {
                        routing_key: key.clone(),
                        role: selection.role,
                        group: selection.group,
                    },
                    selection.engine.spec.id.clone(),
                );
        }

        selection
            .engine
            .local_dispatch_total
            .fetch_add(1, Ordering::Relaxed);
        self.record_policy(selection.role, selection.branch);
    }

    fn record_route(&self, prefill_group: ContextGroup, decode_group: ContextGroup) {
        let key = format!("prefill={prefill_group},decode={decode_group}");
        increment_map_count(&self.route_counts, key);
    }

    fn record_policy(&self, role: RouterRole, branch: PolicyBranch) {
        let mut counts = self.policy_counts.lock().expect("policy lock poisoned");
        *counts
            .entry(role.as_str().to_owned())
            .or_default()
            .entry(branch.as_str().to_owned())
            .or_default() += 1;
    }

    fn next_bootstrap_room(&self) -> u64 {
        self.bootstrap_room.fetch_add(1, Ordering::Relaxed)
    }

    fn metadata_decode_engine(&self) -> Result<Arc<EngineState>, RouterError> {
        self.engines
            .iter()
            .find(|engine| {
                engine.spec.role == RouterRole::Decode
                    && engine.spec.group == ContextGroup::Short
                    && engine.is_healthy()
            })
            .or_else(|| {
                self.engines
                    .iter()
                    .find(|engine| engine.spec.role == RouterRole::Decode && engine.is_healthy())
            })
            .cloned()
            .ok_or(RouterError::NoHealthyEngine {
                role: RouterRole::Decode,
                group: ContextGroup::Short,
            })
    }
}

fn increment_map_count(counts: &Mutex<BTreeMap<String, usize>>, key: String) {
    *counts
        .lock()
        .expect("counter lock poisoned")
        .entry(key)
        .or_default() += 1;
}

fn can_use_reported_score(role: RouterRole, reported: &ReportedLoad) -> bool {
    reported.token_usage.is_some()
        && (role == RouterRole::Prefill || reported.decode_batch_size.is_some())
}

fn score_for_engine(
    role: RouterRole,
    engine: &EngineState,
    reported: &ReportedLoad,
    lambda: f64,
    use_reported: bool,
) -> f64 {
    if use_reported {
        let work = match role {
            RouterRole::Prefill => engine.local_work(role),
            RouterRole::Decode => reported
                .decode_batch_size
                .unwrap_or_else(|| engine.local_work(role)),
        };
        load_score(work, reported.token_usage, lambda)
    } else {
        load_score(engine.local_work(role), None, lambda)
    }
}

fn load_score(work: usize, token_usage: Option<f64>, lambda: f64) -> f64 {
    let work = work.to_f64().unwrap_or(f64::MAX);
    let usage = token_usage
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0 - LOAD_USAGE_EPSILON);
    let lambda = if lambda.is_finite() { lambda } else { 0.0 };

    work + lambda * usage / (1.0 - usage)
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<RouterRuntime>,
}

impl AppState {
    #[must_use]
    pub fn new(config: RouterConfig) -> Self {
        Self {
            inner: Arc::new(RouterRuntime::new(config)),
        }
    }

    fn config(&self) -> &RouterConfig {
        &self.inner.config
    }

    fn route_plan(&self, headers: &HeaderMap, body: &Value) -> Result<RoutePlan, RouterError> {
        let features = RoutingFeatures::from_request(headers, body);
        let prefill_group = select_prefill_group(self.config(), &features);
        let decode_group = select_decode_group(self.config(), &features);
        let prefill =
            self.inner
                .select_without_commit(RouterRole::Prefill, prefill_group, &features)?;
        let decode =
            self.inner
                .select_without_commit(RouterRole::Decode, decode_group, &features)?;

        let plan = RoutePlan {
            features,
            prefill_group,
            decode_group,
            prefill,
            decode,
        };
        self.inner.commit_plan(&plan);

        Ok(plan)
    }

    fn start_load_polling(&self) {
        let state = self.clone();
        drop(tokio::spawn(async move {
            let mut interval = tokio::time::interval(LOAD_POLL_INTERVAL);
            loop {
                interval.tick().await;
                state.poll_loads_once().await;
            }
        }));
    }

    pub async fn poll_loads_once(&self) {
        for engine in &self.inner.engines {
            let engine = Arc::clone(engine);
            let client = self.inner.client.clone();
            poll_engine_load(&client, &engine).await;
        }
    }
}

pub fn app(state: AppState) -> Router {
    let enable_test_load_override = state.config().enable_test_load_override;
    let router = Router::new()
        .route("/health", get(health))
        .route("/get_model_info", get(get_model_info))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/generate", post(generate))
        .route("/debug/routing_stats", get(routing_stats));

    if enable_test_load_override {
        router
            .route("/debug/test_load_override", post(test_load_override))
            .with_state(state)
    } else {
        router.with_state(state)
    }
}

/// Serve the router using the supplied local configuration.
///
/// # Errors
///
/// Returns an error if the TCP listener cannot bind or if the axum server exits
/// with an error.
pub async fn serve_config(config: RouterConfig) -> anyhow::Result<()> {
    let listen_addr = config.listen_addr;
    let state = AppState::new(config);
    state.start_load_polling();
    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("failed to bind router on {listen_addr}"))?;
    tracing::info!("sglang-light-router listening on {}", listen_addr);
    axum::serve(listener, app(state))
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
            .config()
            .engines
            .iter()
            .filter(|engine| engine.role == role)
            .count();
        engines_by_role.insert(role.as_str(), count);
    }

    Json(HealthResponse {
        status: "ok",
        readiness: "local_config",
        engines_total: state.config().engines.len(),
        engines_by_role,
    })
}

async fn get_model_info(State(state): State<AppState>) -> Result<Response, RouterError> {
    let decode = state.inner.metadata_decode_engine()?;
    forward_metadata_to_decode(&state, &decode, "get_model_info").await
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, RouterError> {
    route_and_forward(state, headers, body, ProxyEndpoint::ChatCompletions).await
}

async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, RouterError> {
    route_and_forward(state, headers, body, ProxyEndpoint::Generate).await
}

async fn route_and_forward(
    state: AppState,
    headers: HeaderMap,
    mut body: Value,
    endpoint: ProxyEndpoint,
) -> Result<Response, RouterError> {
    let plan = state.route_plan(&headers, &body)?;
    inject_bootstrap(
        &mut body,
        &plan.prefill.engine,
        state.inner.next_bootstrap_room(),
    )?;

    let guard = LoadGuard::new(
        Arc::clone(&plan.prefill.engine),
        Arc::clone(&plan.decode.engine),
        plan.features.uncached_prefill_tokens,
    );
    let response = forward_to_decode(&state, &plan, &body, endpoint).await?;
    let response = attach_guard(response, guard);

    Ok(add_debug_headers(response, state.config(), &plan))
}

#[derive(Debug, Clone, Copy)]
enum ProxyEndpoint {
    ChatCompletions,
    Generate,
}

impl ProxyEndpoint {
    const fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "v1/chat/completions",
            Self::Generate => "generate",
        }
    }
}

fn select_prefill_group(config: &RouterConfig, features: &RoutingFeatures) -> ContextGroup {
    if features.uncached_prefill_tokens >= config.prefill_long_threshold {
        ContextGroup::Long
    } else {
        ContextGroup::Short
    }
}

fn select_decode_group(config: &RouterConfig, features: &RoutingFeatures) -> ContextGroup {
    if features.estimated_sequence_length >= config.decode_long_threshold {
        ContextGroup::Long
    } else {
        ContextGroup::Short
    }
}

fn extract_routing_key(headers: &HeaderMap) -> Option<String> {
    routing_key_from_header(headers, "x-smg-routing-key")
        .or_else(|| routing_key_from_header(headers, "x-sglang-routing-key"))
}

fn routing_key_from_header(headers: &HeaderMap, name: &'static str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn estimate_prompt_tokens(body: &Value) -> usize {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        return messages
            .iter()
            .filter_map(|message| message.get("content"))
            .map(count_content_tokens)
            .sum();
    }

    body.get("prompt")
        .or_else(|| body.get("text"))
        .map_or(0, count_content_tokens)
}

fn count_content_tokens(value: &Value) -> usize {
    match value {
        Value::String(text) => text.split_whitespace().count(),
        Value::Array(values) => values.iter().map(count_content_tokens).sum(),
        Value::Object(object) => object.get("text").map_or(0, count_content_tokens),
        _ => 0,
    }
}

fn max_new_tokens(body: &Value) -> usize {
    ["max_tokens", "max_completion_tokens", "max_new_tokens"]
        .iter()
        .find_map(|key| body.get(*key).and_then(value_to_usize))
        .or_else(|| {
            body.get("sampling_params").and_then(|sampling_params| {
                ["max_new_tokens", "max_tokens"]
                    .iter()
                    .find_map(|key| sampling_params.get(*key).and_then(value_to_usize))
            })
        })
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS)
}

fn value_to_usize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
}

fn inject_bootstrap(
    body: &mut Value,
    prefill: &EngineState,
    bootstrap_room: u64,
) -> Result<(), RouterError> {
    let object = body
        .as_object_mut()
        .ok_or(RouterError::RequestBodyNotObject)?;
    let bootstrap_port =
        prefill
            .spec
            .bootstrap_port
            .ok_or_else(|| RouterError::MissingSelectedBootstrapPort {
                engine_id: prefill.spec.id.clone(),
            })?;
    let host =
        prefill
            .spec
            .url
            .host_str()
            .ok_or_else(|| RouterError::MissingSelectedBootstrapHost {
                engine_id: prefill.spec.id.clone(),
            })?;

    object.insert("bootstrap_host".to_owned(), Value::String(host.to_owned()));
    object.insert("bootstrap_port".to_owned(), Value::from(bootstrap_port));
    object.insert("bootstrap_room".to_owned(), Value::from(bootstrap_room));
    Ok(())
}

async fn forward_to_decode(
    state: &AppState,
    plan: &RoutePlan,
    body: &Value,
    endpoint: ProxyEndpoint,
) -> Result<Response, RouterError> {
    let decode = &plan.decode.engine;
    let upstream_url =
        decode
            .spec
            .url
            .join(endpoint.path())
            .map_err(|source| RouterError::UpstreamUrl {
                engine_id: decode.spec.id.clone(),
                source,
            })?;

    let upstream = state
        .inner
        .client
        .post(upstream_url)
        .json(body)
        .send()
        .await
        .map_err(|source| RouterError::UpstreamRequest {
            engine_id: decode.spec.id.clone(),
            source,
        })?;
    let status = upstream.status();
    let upstream_headers = upstream.headers().clone();
    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    copy_safe_response_headers(&upstream_headers, response.headers_mut());
    Ok(response)
}

async fn forward_metadata_to_decode(
    state: &AppState,
    decode: &EngineState,
    path: &'static str,
) -> Result<Response, RouterError> {
    let upstream_url = decode
        .spec
        .url
        .join(path)
        .map_err(|source| RouterError::UpstreamUrl {
            engine_id: decode.spec.id.clone(),
            source,
        })?;

    let upstream = state
        .inner
        .client
        .get(upstream_url)
        .send()
        .await
        .map_err(|source| RouterError::UpstreamRequest {
            engine_id: decode.spec.id.clone(),
            source,
        })?;
    let status = upstream.status();
    let upstream_headers = upstream.headers().clone();
    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    copy_safe_response_headers(&upstream_headers, response.headers_mut());
    Ok(response)
}

fn copy_safe_response_headers(source: &HeaderMap, target: &mut HeaderMap) {
    for (name, value) in source {
        if is_safe_response_header(name) {
            target.insert(name.clone(), value.clone());
        }
    }
}

fn is_safe_response_header(name: &HeaderName) -> bool {
    name == header::CONTENT_TYPE || name.as_str().eq_ignore_ascii_case("x-request-id")
}

fn add_debug_headers(mut response: Response, config: &RouterConfig, plan: &RoutePlan) -> Response {
    if !config.enable_routing_debug_headers {
        return response;
    }

    insert_static_header(
        response.headers_mut(),
        "x-sglang-prefill-group",
        plan.prefill_group.as_str(),
    );
    insert_static_header(
        response.headers_mut(),
        "x-sglang-decode-group",
        plan.decode_group.as_str(),
    );
    insert_header(
        response.headers_mut(),
        "x-sglang-prefill-worker",
        &plan.prefill.engine.spec.id,
    );
    insert_header(
        response.headers_mut(),
        "x-sglang-decode-worker",
        &plan.decode.engine.spec.id,
    );
    insert_static_header(
        response.headers_mut(),
        "x-sglang-prefill-policy-branch",
        plan.prefill.branch.as_str(),
    );
    insert_static_header(
        response.headers_mut(),
        "x-sglang-decode-policy-branch",
        plan.decode.branch.as_str(),
    );
    insert_header(
        response.headers_mut(),
        "x-sglang-prompt-tokens",
        &plan.features.prompt_tokens.to_string(),
    );
    insert_header(
        response.headers_mut(),
        "x-sglang-max-new-tokens",
        &plan.features.max_new_tokens.to_string(),
    );

    response
}

fn insert_static_header(headers: &mut HeaderMap, name: &'static str, value: &'static str) {
    headers.insert(
        HeaderName::from_static(name),
        HeaderValue::from_static(value),
    );
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

#[derive(Debug)]
struct LoadGuard {
    prefill: Arc<EngineState>,
    decode: Arc<EngineState>,
    prefill_tokens: usize,
}

impl LoadGuard {
    fn new(prefill: Arc<EngineState>, decode: Arc<EngineState>, prefill_tokens: usize) -> Self {
        prefill.in_flight_requests.fetch_add(1, Ordering::Relaxed);
        prefill
            .in_flight_prefill_tokens
            .fetch_add(prefill_tokens, Ordering::Relaxed);
        decode.in_flight_requests.fetch_add(1, Ordering::Relaxed);
        decode
            .in_flight_decode_requests
            .fetch_add(1, Ordering::Relaxed);

        Self {
            prefill,
            decode,
            prefill_tokens,
        }
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.prefill
            .in_flight_requests
            .fetch_sub(1, Ordering::Relaxed);
        self.prefill
            .in_flight_prefill_tokens
            .fetch_sub(self.prefill_tokens, Ordering::Relaxed);
        self.decode
            .in_flight_requests
            .fetch_sub(1, Ordering::Relaxed);
        self.decode
            .in_flight_decode_requests
            .fetch_sub(1, Ordering::Relaxed);
    }
}

struct AttachedBody<T> {
    inner: Body,
    _attached: T,
}

impl<T> AttachedBody<T> {
    const fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + Unpin + 'static> http_body::Body for AttachedBody<T> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

fn attach_guard(response: Response, guard: LoadGuard) -> Response {
    let (parts, body) = response.into_parts();
    Response::from_parts(parts, Body::new(AttachedBody::new(body, guard)))
}

#[derive(Debug, Serialize)]
struct RoutingStats {
    route_counts: BTreeMap<String, usize>,
    policy_counts: BTreeMap<String, BTreeMap<String, usize>>,
    engines: Vec<EngineStats>,
}

#[derive(Debug, Serialize)]
struct EngineStats {
    id: String,
    role: RouterRole,
    group: ContextGroup,
    url: Url,
    local_dispatch_total: usize,
    in_flight_requests: usize,
    work: usize,
    reported_total_tokens: Option<usize>,
    reported_token_usage: Option<f64>,
    reported_decode_batch_size: Option<usize>,
    last_load_poll_ok: bool,
}

async fn routing_stats(State(state): State<AppState>) -> Json<RoutingStats> {
    Json(state.routing_stats())
}

impl AppState {
    fn routing_stats(&self) -> RoutingStats {
        let engines = self
            .inner
            .engines
            .iter()
            .map(|engine| {
                let reported = engine.reported();
                EngineStats {
                    id: engine.spec.id.clone(),
                    role: engine.spec.role,
                    group: engine.spec.group,
                    url: engine.spec.url.clone(),
                    local_dispatch_total: engine.local_dispatch_total.load(Ordering::Relaxed),
                    in_flight_requests: engine.in_flight_requests.load(Ordering::Relaxed),
                    work: engine.local_work(engine.spec.role),
                    reported_total_tokens: reported.total_tokens,
                    reported_token_usage: reported.token_usage,
                    reported_decode_batch_size: reported.decode_batch_size,
                    last_load_poll_ok: reported.last_ok,
                }
            })
            .collect();

        RoutingStats {
            route_counts: self
                .inner
                .route_counts
                .lock()
                .expect("route counter lock poisoned")
                .clone(),
            policy_counts: self
                .inner
                .policy_counts
                .lock()
                .expect("policy counter lock poisoned")
                .clone(),
            engines,
        }
    }
}

#[derive(Debug, Deserialize)]
struct LoadOverrideRequest {
    id: String,
    healthy: Option<bool>,
    work: Option<usize>,
    total_tokens: Option<usize>,
    token_usage: Option<f64>,
    decode_batch_size: Option<usize>,
    last_load_poll_ok: Option<bool>,
}

#[derive(Debug, Serialize)]
struct LoadOverrideResponse {
    status: &'static str,
}

async fn test_load_override(
    State(state): State<AppState>,
    Json(request): Json<LoadOverrideRequest>,
) -> Result<Json<LoadOverrideResponse>, RouterError> {
    let engine = state
        .inner
        .engines_by_id
        .get(&request.id)
        .ok_or_else(|| RouterError::UnknownEngine(request.id.clone()))?;

    if let Some(healthy) = request.healthy {
        engine.set_healthy(healthy);
    }
    if let Some(work) = request.work {
        engine.set_local_work_for_test(work);
    }

    let mut reported = engine.reported();
    if request.total_tokens.is_some() {
        reported.total_tokens = request.total_tokens;
    }
    if request.token_usage.is_some() {
        reported.token_usage = request.token_usage;
    }
    if request.decode_batch_size.is_some() {
        reported.decode_batch_size = request.decode_batch_size;
    }
    if let Some(last_ok) = request.last_load_poll_ok {
        reported.last_ok = last_ok;
    }
    engine.set_reported(reported);

    Ok(Json(LoadOverrideResponse { status: "ok" }))
}

async fn poll_engine_load(client: &reqwest::Client, engine: &EngineState) {
    let load_url = match engine.spec.url.join("v1/loads?include=core") {
        Ok(url) => url,
        Err(error) => {
            tracing::warn!(engine_id = %engine.spec.id, %error, "failed to build load URL");
            engine.mark_load_poll_failed();
            return;
        }
    };

    let result = client.get(load_url).send().await;
    let Ok(response) = result else {
        engine.mark_load_poll_failed();
        return;
    };
    if !response.status().is_success() {
        engine.mark_load_poll_failed();
        return;
    }

    match response.json::<Value>().await {
        Ok(value) => engine.set_reported(parse_load_response(&value)),
        Err(error) => {
            tracing::warn!(engine_id = %engine.spec.id, %error, "failed to parse load response");
            engine.mark_load_poll_failed();
        }
    }
}

fn parse_load_response(value: &Value) -> ReportedLoad {
    let aggregate = value.get("aggregate").unwrap_or(value);
    let total_tokens = find_usize(
        aggregate,
        &[
            "total_used_tokens",
            "num_used_tokens",
            "reported_total_tokens",
        ],
    );
    let capacity = find_usize(
        aggregate,
        &["total_tokens", "num_total_tokens", "max_total_num_tokens"],
    );
    let token_usage = find_f64(
        aggregate,
        &["avg_token_usage", "token_usage", "reported_token_usage"],
    )
    .or_else(|| ratio(total_tokens, capacity));
    let decode_batch_size = find_usize(
        aggregate,
        &[
            "total_running_reqs",
            "num_running_reqs",
            "decode_batch_size",
        ],
    );

    ReportedLoad {
        total_tokens,
        token_usage,
        decode_batch_size,
        last_ok: true,
    }
}

fn find_usize(object: &Value, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(value_to_usize))
}

fn find_f64(object: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
    })
}

fn ratio(numerator: Option<usize>, denominator: Option<usize>) -> Option<f64> {
    let numerator = numerator?.to_f64()?;
    let denominator = denominator?.to_f64()?;
    (denominator > 0.0).then_some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RouterConfig {
        RouterConfig {
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 20_000)),
            prefill_long_threshold: 10,
            decode_long_threshold: 20,
            prefill_routing_policy_chain: "sticky,power_of_two".to_owned(),
            decode_routing_policy_chain: "sticky,power_of_two".to_owned(),
            balance_abs_threshold: 2.0,
            balance_relative_upper_bound_limit: 2.0,
            load_score_token_usage_weight: 1.0,
            enable_routing_debug_headers: true,
            enable_test_load_override: true,
            engines: vec![
                engine(
                    "short-prefill-0",
                    RouterRole::Prefill,
                    ContextGroup::Short,
                    30_000,
                    Some(8998),
                ),
                engine(
                    "short-prefill-1",
                    RouterRole::Prefill,
                    ContextGroup::Short,
                    30_002,
                    Some(8997),
                ),
                engine(
                    "short-decode-0",
                    RouterRole::Decode,
                    ContextGroup::Short,
                    31_000,
                    None,
                ),
                engine(
                    "short-decode-1",
                    RouterRole::Decode,
                    ContextGroup::Short,
                    31_002,
                    None,
                ),
                engine(
                    "long-prefill-0",
                    RouterRole::Prefill,
                    ContextGroup::Long,
                    30_001,
                    Some(8999),
                ),
                engine(
                    "long-decode-0",
                    RouterRole::Decode,
                    ContextGroup::Long,
                    31_001,
                    None,
                ),
            ],
        }
    }

    fn engine(
        id: &str,
        role: RouterRole,
        group: ContextGroup,
        port: u16,
        bootstrap_port: Option<u16>,
    ) -> EngineSpec {
        EngineSpec {
            id: id.to_owned(),
            role,
            group,
            url: Url::parse(&format!("http://127.0.0.1:{port}")).expect("valid URL"),
            bootstrap_port,
        }
    }

    fn features(prompt_tokens: usize, max_new_tokens: usize) -> RoutingFeatures {
        RoutingFeatures {
            routing_key: None,
            prompt_tokens,
            uncached_prefill_tokens: prompt_tokens,
            max_new_tokens,
            estimated_sequence_length: prompt_tokens + max_new_tokens,
        }
    }

    #[test]
    fn feature_extraction_prefers_smg_routing_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-sglang-routing-key", HeaderValue::from_static("fallback"));
        headers.insert("x-smg-routing-key", HeaderValue::from_static("primary"));
        let body = json!({
            "messages": [{"role": "user", "content": "one two three"}],
            "max_completion_tokens": 7
        });

        let features = RoutingFeatures::from_request(&headers, &body);

        assert_eq!(features.routing_key.as_deref(), Some("primary"));
        assert_eq!(features.prompt_tokens, 3);
        assert_eq!(features.uncached_prefill_tokens, 3);
        assert_eq!(features.max_new_tokens, 7);
        assert_eq!(features.estimated_sequence_length, 10);
    }

    #[test]
    fn dispatch_uses_independent_thresholds() {
        let config = test_config();

        assert_eq!(
            select_prefill_group(&config, &features(9, 1)),
            ContextGroup::Short
        );
        assert_eq!(
            select_prefill_group(&config, &features(10, 1)),
            ContextGroup::Long
        );
        assert_eq!(
            select_decode_group(&config, &features(10, 9)),
            ContextGroup::Short
        );
        assert_eq!(
            select_decode_group(&config, &features(10, 10)),
            ContextGroup::Long
        );
    }

    #[test]
    fn dispatch_allows_cross_group_route() {
        let config = test_config();
        let features = features(12, 1);

        assert_eq!(select_prefill_group(&config, &features), ContextGroup::Long);
        assert_eq!(select_decode_group(&config, &features), ContextGroup::Short);
    }

    #[test]
    fn score_clamps_full_token_usage() {
        let score = load_score(4, Some(1.0), 1.0);

        assert!(score.is_finite());
        assert!(score > 4.0);
    }

    #[test]
    fn sticky_is_scoped_by_role_and_group() {
        let state = AppState::new(test_config());
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-sglang-routing-key",
            HeaderValue::from_static("session-1"),
        );

        let short_body = json!({
            "messages": [{"role": "user", "content": "short"}],
            "max_tokens": 1
        });
        let first = state
            .route_plan(&headers, &short_body)
            .expect("short route should select");
        let second = state
            .route_plan(&headers, &short_body)
            .expect("short sticky route should select");

        assert_eq!(second.prefill.branch, PolicyBranch::StickyHit);
        assert_eq!(second.decode.branch, PolicyBranch::StickyHit);
        assert_eq!(first.prefill.engine.spec.id, second.prefill.engine.spec.id);
        assert_eq!(first.decode.engine.spec.id, second.decode.engine.spec.id);

        let long_body = json!({
            "messages": [{"role": "user", "content": "one two three four five six seven eight nine ten"}],
            "max_tokens": 10
        });
        let long = state
            .route_plan(&headers, &long_body)
            .expect("long route should select");

        assert_eq!(long.prefill_group, ContextGroup::Long);
        assert_eq!(long.decode_group, ContextGroup::Long);
        assert_ne!(second.prefill.engine.spec.id, long.prefill.engine.spec.id);
        assert_ne!(second.decode.engine.spec.id, long.decode.engine.spec.id);
    }

    #[test]
    fn unhealthy_sticky_falls_back_without_commit() {
        let state = AppState::new(test_config());
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-sglang-routing-key",
            HeaderValue::from_static("session-2"),
        );
        let body = json!({
            "messages": [{"role": "user", "content": "short"}],
            "max_tokens": 1
        });
        let first = state
            .route_plan(&headers, &body)
            .expect("first route should select");

        first.decode.engine.set_healthy(false);
        let second = state
            .route_plan(&headers, &body)
            .expect("second route should select another decode");

        assert_eq!(second.decode.branch, PolicyBranch::LoadBasedFallback);
        assert_ne!(first.decode.engine.spec.id, second.decode.engine.spec.id);
    }

    #[test]
    fn no_healthy_candidate_returns_typed_error() {
        let state = AppState::new(test_config());
        for engine in &state.inner.engines {
            if engine.spec.role == RouterRole::Prefill && engine.spec.group == ContextGroup::Short {
                engine.set_healthy(false);
            }
        }
        let body = json!({
            "messages": [{"role": "user", "content": "short"}],
            "max_tokens": 1
        });

        let error = state
            .route_plan(&HeaderMap::new(), &body)
            .expect_err("short prefill has no healthy candidate");

        assert!(matches!(
            error,
            RouterError::NoHealthyEngine {
                role: RouterRole::Prefill,
                group: ContextGroup::Short
            }
        ));
    }

    #[test]
    fn load_guard_lifetime_is_tied_to_attached_response_body() {
        let state = AppState::new(test_config());
        let prefill = Arc::clone(
            state
                .inner
                .engines_by_id
                .get("short-prefill-0")
                .expect("prefill engine exists"),
        );
        let decode = Arc::clone(
            state
                .inner
                .engines_by_id
                .get("short-decode-0")
                .expect("decode engine exists"),
        );
        let guard = LoadGuard::new(Arc::clone(&prefill), Arc::clone(&decode), 5);
        let response = attach_guard(Response::new(Body::empty()), guard);

        assert_eq!(prefill.in_flight_requests.load(Ordering::Relaxed), 1);
        assert_eq!(decode.in_flight_requests.load(Ordering::Relaxed), 1);

        drop(response);

        assert_eq!(prefill.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(decode.in_flight_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn load_response_parser_reads_sglang_aggregate_shape() {
        let reported = parse_load_response(&json!({
            "aggregate": {
                "total_running_reqs": 3,
                "total_used_tokens": 128,
                "total_tokens": 1024,
                "avg_token_usage": 0.125
            }
        }));

        assert_eq!(reported.total_tokens, Some(128));
        assert_eq!(reported.decode_batch_size, Some(3));
        assert_eq!(reported.token_usage, Some(0.125));
        assert!(reported.last_ok);
    }

    #[test]
    fn bootstrap_injection_requires_object_body() {
        let state = AppState::new(test_config());
        let prefill = state
            .inner
            .engines_by_id
            .get("short-prefill-0")
            .expect("prefill engine exists");
        let mut body = Value::Array(Vec::new());

        let error = inject_bootstrap(&mut body, prefill, 1).expect_err("array is invalid");

        assert!(matches!(error, RouterError::RequestBodyNotObject));
    }

    #[test]
    fn object_content_array_contributes_text_tokens() {
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "one two"},
                    {"type": "image_url", "image_url": {"url": "http://example.invalid/image.png"}}
                ]
            }]
        });

        assert_eq!(estimate_prompt_tokens(&body), 2);
    }

    #[test]
    fn native_generate_body_contributes_text_and_sampling_max_new_tokens() {
        let body = json!({
            "text": "one two three four",
            "sampling_params": {"max_new_tokens": 5}
        });
        let features = RoutingFeatures::from_request(&HeaderMap::new(), &body);

        assert_eq!(features.prompt_tokens, 4);
        assert_eq!(features.max_new_tokens, 5);
        assert_eq!(features.estimated_sequence_length, 9);
    }

    #[test]
    fn empty_routing_key_is_skipped() {
        let mut headers = HeaderMap::new();
        headers.insert("x-sglang-routing-key", HeaderValue::from_static("   "));

        assert_eq!(extract_routing_key(&headers), None);
    }
}
