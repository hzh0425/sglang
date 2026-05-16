use anyhow::Context;
use clap::Parser;
use sglang_light_router::{Cli, RouterConfig, serve_config};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sglang_light_router=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let config = RouterConfig::try_from(cli).context("invalid router configuration")?;
    serve_config(config).await
}
