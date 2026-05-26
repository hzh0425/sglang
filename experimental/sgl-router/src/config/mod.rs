pub mod types;
pub use types::*;

use anyhow::Context as _;
use anyhow::{anyhow, Result};
use std::path::Path;

impl Config {
    pub fn from_path(p: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(p).with_context(|| format!("read config {}", p.display()))?;
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        let cfg: Config = match ext {
            "yaml" | "yml" => serde_yaml::from_str(&raw)
                .map_err(|e| anyhow!("parse yaml {}: {e}", p.display()))?,
            "toml" => {
                toml::from_str(&raw).map_err(|e| anyhow!("parse toml {}: {e}", p.display()))?
            }
            other => {
                return Err(anyhow!(
                    "unsupported config extension {other:?}; want yaml/yml/toml"
                ))
            }
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        // Unknown policy names are rejected by serde via `PolicyKind`'s
        // `rename_all = "snake_case"`; threshold = 0 is rejected by
        // `NonZeroU32`.  Only fields without a type-system constraint are
        // checked here.
        for m in &self.models {
            if m.id.is_empty() {
                return Err(anyhow!("model.id must be non-empty"));
            }
            if let Some(bucket) = &m.pd_bucket {
                if bucket.groups.is_empty()
                    && (bucket.short_group.trim().is_empty() || bucket.long_group.trim().is_empty())
                {
                    return Err(anyhow!(
                        "model {:?} pd_bucket group names must be non-empty",
                        m.id
                    ));
                }
                if bucket.groups.is_empty() && bucket.short_group == bucket.long_group {
                    return Err(anyhow!(
                        "model {:?} pd_bucket short_group and long_group must differ",
                        m.id
                    ));
                }
                if bucket.groups.is_empty()
                    && (bucket.prefill_long_threshold == 0 || bucket.decode_long_threshold == 0)
                {
                    return Err(anyhow!(
                        "model {:?} pd_bucket thresholds must be greater than zero",
                        m.id
                    ));
                }
                let mut previous_max_tokens = 0;
                for group in &bucket.groups {
                    if group.group.trim().is_empty() {
                        return Err(anyhow!(
                            "model {:?} pd_bucket group names must be non-empty",
                            m.id
                        ));
                    }
                    if group.max_tokens <= previous_max_tokens {
                        return Err(anyhow!(
                            "model {:?} pd_bucket group max_tokens must be strictly increasing",
                            m.id
                        ));
                    }
                    previous_max_tokens = group.max_tokens;
                }
            }
        }
        match &self.discovery.backend {
            DiscoveryBackend::StaticUrls(s) => {
                if s.urls.is_empty() {
                    return Err(anyhow!(
                        "discovery.static_urls.urls must be a non-empty list"
                    ));
                }
                // Validate every entry up front so typos surface at
                // config-load with a precise diagnostic instead of as
                // per-worker introspect failures or as two registry
                // entries pointing at the same SGLang (trailing-slash
                // near-duplicates). Dedupe runs against a normalized
                // form (trimmed + trailing `/` stripped) so
                // `"http://x:30000"` and `"http://x:30000/"` collide.
                let mut seen = std::collections::HashSet::new();
                for raw in &s.urls {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        return Err(anyhow!(
                            "discovery.static_urls.urls contains an empty or whitespace-only entry"
                        ));
                    }
                    let parsed = url::Url::parse(trimmed).map_err(|e| {
                        anyhow!("discovery.static_urls.urls entry {raw:?} is not a valid URL: {e}")
                    })?;
                    match parsed.scheme() {
                        "http" | "https" => {}
                        other => {
                            return Err(anyhow!(
                                "discovery.static_urls.urls entry {raw:?} has unsupported scheme {other:?}; only http and https are supported"
                            ));
                        }
                    }
                    let normalized = parsed.as_str().trim_end_matches('/').to_string();
                    if !seen.insert(normalized.clone()) {
                        return Err(anyhow!(
                            "discovery.static_urls.urls contains duplicate entry {raw:?} (normalized: {normalized:?})"
                        ));
                    }
                }
                let mut grouped = std::collections::HashSet::new();
                for entry in &s.worker_groups {
                    let trimmed = entry.url.trim();
                    let group = entry.group.trim();
                    if group.is_empty() {
                        return Err(anyhow!(
                            "discovery.static_urls.worker_groups entry for {:?} has an empty group",
                            entry.url
                        ));
                    }
                    let parsed = url::Url::parse(trimmed).map_err(|e| {
                        anyhow!(
                            "discovery.static_urls.worker_groups url {:?} is not a valid URL: {e}",
                            entry.url
                        )
                    })?;
                    let normalized = parsed.as_str().trim_end_matches('/').to_string();
                    if !seen.contains(&normalized) {
                        return Err(anyhow!(
                            "discovery.static_urls.worker_groups url {:?} is not present in discovery.static_urls.urls",
                            entry.url
                        ));
                    }
                    if !grouped.insert(normalized.clone()) {
                        return Err(anyhow!(
                            "discovery.static_urls.worker_groups contains duplicate url {:?}",
                            entry.url
                        ));
                    }
                }
            }
            DiscoveryBackend::K8s(k) => {
                // Empty namespace is intentional: triggers `Api::all(client)`
                // for cluster-wide EndpointSlice watch (see
                // `discovery::k8s::spawn`). Only validate the selector
                // combination here.
                let _ = &k.namespace;
                k.mode().map_err(|e| anyhow!("{e}"))?;
            }
        }
        Ok(())
    }

    pub fn static_worker_group(&self, worker_url: &str) -> Option<&str> {
        let DiscoveryBackend::StaticUrls(static_urls) = &self.discovery.backend else {
            return None;
        };
        let normalized = normalize_url_for_config_match(worker_url)?;
        static_urls
            .worker_groups
            .iter()
            .find(|entry| {
                normalize_url_for_config_match(&entry.url).as_deref() == Some(&normalized)
            })
            .map(|entry| entry.group.as_str())
    }
}

fn normalize_url_for_config_match(raw: &str) -> Option<String> {
    url::Url::parse(raw.trim())
        .ok()
        .map(|parsed| parsed.as_str().trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write `body` to a temp file with the given extension and load it
    /// through `Config::from_path`. Failures still surface the offending
    /// config because each call site passes its body inline.
    fn load(ext: &str, body: &str) -> Result<Config> {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(format!("c.{ext}"));
        std::fs::write(&p, body).unwrap();
        Config::from_path(&p)
    }

    #[test]
    fn loads_minimal_yaml() {
        let c = load(
            "yaml",
            r#"
server:
  host: "0.0.0.0"
  port: 8090
models:
  - id: "qwen3-0.6b"
    tokenizer_path: "/tmp/qwen.json"
discovery:
  backend: static_urls
  static_urls:
    urls:
      - "http://10.0.0.1:30000"
"#,
        )
        .unwrap();
        assert_eq!(c.server.port, 8090);
        assert_eq!(c.models[0].id, "qwen3-0.6b");
        match &c.discovery.backend {
            DiscoveryBackend::StaticUrls(s) => {
                assert_eq!(s.urls, vec!["http://10.0.0.1:30000".to_string()])
            }
            _ => panic!("expected static_urls backend"),
        }
    }

    #[test]
    fn loads_minimal_toml() {
        let c = load(
            "toml",
            r#"
[server]
host = "0.0.0.0"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://10.0.0.1:30000"]
"#,
        )
        .unwrap();
        assert_eq!(c.server.port, 8090);
        match &c.discovery.backend {
            DiscoveryBackend::StaticUrls(s) => {
                assert_eq!(s.urls, vec!["http://10.0.0.1:30000".to_string()])
            }
            _ => panic!("expected static_urls backend"),
        }
    }

    #[test]
    fn rejects_missing_discovery_section() {
        let err = load(
            "yaml",
            "server:\n  host: \"0.0.0.0\"\n  port: 8090\nmodels: []\n",
        )
        .unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("discovery") || msg.contains("missing"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_unknown_extension() {
        let err = load("txt", "").unwrap_err();
        assert!(err.to_string().contains("yaml") && err.to_string().contains("toml"));
    }

    #[test]
    fn loads_static_urls_discovery() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
policy = "round_robin"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://10.0.0.1:30000", "http://10.0.0.2:30000"]
"#,
        )
        .unwrap();
        match &c.discovery.backend {
            DiscoveryBackend::StaticUrls(s) => {
                assert_eq!(
                    s.urls,
                    vec![
                        "http://10.0.0.1:30000".to_string(),
                        "http://10.0.0.2:30000".to_string(),
                    ],
                );
            }
            _ => panic!("expected static_urls backend"),
        }
        assert_eq!(c.models[0].policy, PolicyKind::RoundRobin);
    }

    #[test]
    fn loads_load_based_policy() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
policy = "load_based"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://10.0.0.1:30000"]
"#,
        )
        .unwrap();
        assert_eq!(c.models[0].policy, PolicyKind::LoadBased);
    }

    #[test]
    fn loads_sticky_session_load_based_policy() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
policy = "sticky_session_load_based"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://10.0.0.1:30000"]
"#,
        )
        .unwrap();
        assert_eq!(c.models[0].policy, PolicyKind::StickySessionLoadBased);
    }

    #[test]
    fn loads_pd_bucket_worker_groups() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
policy = "sticky_session_load_based"
[models.pd_bucket]
short_group = "short"
long_group = "long"
prefill_long_threshold = 512
decode_long_threshold = 768
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://10.0.0.1:30000", "http://10.0.0.2:30000"]
[[discovery.static_urls.worker_groups]]
url = "http://10.0.0.1:30000"
group = "short"
[[discovery.static_urls.worker_groups]]
url = "http://10.0.0.2:30000"
group = "long"
"#,
        )
        .unwrap();
        let bucket = c.models[0].pd_bucket.as_ref().unwrap();
        assert_eq!(bucket.short_group, "short");
        assert_eq!(bucket.long_group, "long");
        assert_eq!(bucket.prefill_long_threshold, 512);
        assert_eq!(bucket.decode_long_threshold, 768);
        assert_eq!(
            c.static_worker_group("http://10.0.0.1:30000"),
            Some("short")
        );
        assert_eq!(c.static_worker_group("http://10.0.0.2:30000"), Some("long"));
    }

    #[test]
    fn loads_multi_pd_bucket_groups() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
policy = "sticky_session_load_based"
[models.pd_bucket]
[[models.pd_bucket.groups]]
group = "ctx0_16k"
max_tokens = 16384
[[models.pd_bucket.groups]]
group = "ctx16_32k"
max_tokens = 32768
[[models.pd_bucket.groups]]
group = "ctx32_64k"
max_tokens = 65536
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://10.0.0.1:30000"]
"#,
        )
        .unwrap();
        let bucket = c.models[0].pd_bucket.as_ref().unwrap();
        assert_eq!(bucket.group_for_prefill(8192), "ctx0_16k");
        assert_eq!(bucket.group_for_prefill(16384), "ctx16_32k");
        assert_eq!(bucket.group_for_decode(32768), "ctx32_64k");
        assert_eq!(bucket.group_for_decode(131072), "ctx32_64k");
    }

    #[test]
    fn rejects_static_urls_with_empty_list() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = []
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("non-empty"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_with_duplicate_entry() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://x:30000", "http://x:30000"]
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("duplicate"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_with_empty_entry() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://x:30000", ""]
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    /// Whitespace-only entries are user typos that previously slipped
    /// through `is_empty()` checks and surfaced as "introspect against
    /// `   /server_info` failed" at runtime. Catch at load.
    #[test]
    fn rejects_static_urls_with_whitespace_only_entry() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://x:30000", "   "]
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("whitespace"), "got: {err}");
    }

    /// `"10.0.0.1:30000"` (missing scheme) used to pass validation; the
    /// scheme/`http://` would only fail (or worse, silently degrade
    /// because of the `parse_bootstrap_host` localhost fallback) at
    /// introspect time. Reject at load.
    #[test]
    fn rejects_static_urls_with_schemeless_entry() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["10.0.0.1:30000"]
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("not a valid URL") || err.contains("unsupported scheme"),
            "got: {err}"
        );
    }

    /// Non-http(s) schemes are rejected. The router speaks HTTP to
    /// workers; a `tcp://` or `ws://` entry is almost certainly an
    /// operator typo.
    #[test]
    fn rejects_static_urls_with_non_http_scheme() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["ws://x:30000"]
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unsupported scheme"), "got: {err}");
    }

    /// Trailing-slash near-duplicates collide in the registry but used
    /// to pass byte-equality dedupe. Normalize before checking so two
    /// pointers at the same SGLang surface as a config error.
    #[test]
    fn rejects_static_urls_with_trailing_slash_near_duplicate() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "m"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://x:30000", "http://x:30000/"]
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("duplicate"), "got: {err}");
    }

    #[test]
    fn loads_k8s_discovery() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
policy = "round_robin"
[discovery]
backend = "k8s"
[discovery.k8s]
namespace = "default"
label_selector = "app=sglang"
"#,
        )
        .unwrap();
        match &c.discovery.backend {
            DiscoveryBackend::K8s(k) => {
                assert_eq!(k.namespace, "default");
                assert_eq!(k.label_selector.as_deref(), Some("app=sglang"));
                assert!(k.prefill_selector.is_none());
                assert!(k.decode_selector.is_none());
            }
            _ => panic!("expected k8s backend"),
        }
    }

    /// K8s PD selectors drive slice-classification only; per-worker
    /// bootstrap_port comes from `/server_info` post-discovery
    /// (`crate::workers::introspect`). This test pins the wire-shape;
    /// the selector grammar itself is covered in `types.rs`.
    #[test]
    fn loads_k8s_pd_discovery_with_prefill_and_decode_selectors() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen3-0.6b"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "k8s"
[discovery.k8s]
namespace = "default"
prefill_selector = "app=sglang,role=prefill"
decode_selector = "app=sglang,role=decode"
"#,
        )
        .expect("k8s PD config must load");
        match &c.discovery.backend {
            DiscoveryBackend::K8s(k) => {
                assert_eq!(k.namespace, "default");
                assert_eq!(
                    k.prefill_selector.as_deref(),
                    Some("app=sglang,role=prefill")
                );
                assert_eq!(k.decode_selector.as_deref(), Some("app=sglang,role=decode"));
                assert!(k.label_selector.is_none());
            }
            _ => panic!("expected k8s backend"),
        }
    }

    #[test]
    fn rejects_k8s_config_with_no_selector() {
        let err = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "k8s"
[discovery.k8s]
namespace = "default"
"#,
        )
        .unwrap_err();
        // Pin the specific variant: `ConfigError::NoSelector` ("none were
        // set"). A bare `contains("selector")` would also pass for
        // EmptyPdSelector / PartialPdSelectors / IdenticalPdSelectors /
        // UnsupportedSelectorGrammar — variants that have semantically
        // different error wording but all mention "selector". A future
        // regression that returned, say, `PartialPdSelectors` for the
        // all-None input would be caught here.
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("none were set"),
            "expected NoSelector wording (\"none were set\"); got: {err}",
        );
    }

    // Direct `K8sDiscoveryConfig::mode()` unit tests live alongside the
    // type in `src/config/types.rs::k8s_discovery_config_tests`.
    // The tests in this module exercise the `Config::from_path` ↔ K8s
    // selector wiring, not the selector grammar itself.

    #[test]
    fn rejects_unknown_policy_name() {
        let err = load(
            "yaml",
            "
server:
  host: 0.0.0.0
  port: 8090
discovery:
  backend: static_urls
  static_urls:
    urls:
      - http://x:30000
models:
  - id: qwen
    tokenizer_path: /tmp/qwen.json
    policy: bogus_policy
",
        )
        .unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("bogus_policy") || msg.contains("policy"),
            "got: {err}"
        );
    }

    #[test]
    fn defaults_policy_to_round_robin() {
        let c = load(
            "toml",
            r#"
[server]
host = "127.0.0.1"
port = 8090
[[models]]
id = "qwen"
tokenizer_path = "/tmp/qwen.json"
[discovery]
backend = "static_urls"
[discovery.static_urls]
urls = ["http://x:30000"]
"#,
        )
        .unwrap();
        assert_eq!(c.models[0].policy, PolicyKind::RoundRobin);
    }
}
