#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTER_MANIFEST="${ROOT_DIR}/rust/sglang-light-router/Cargo.toml"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
TP="${TP:-1}"
RDMA_DEVICES="${RDMA_DEVICES:-}"

SHORT_PREFILL_PORT="${SHORT_PREFILL_PORT:-30000}"
SHORT_DECODE_PORT="${SHORT_DECODE_PORT:-31000}"
LONG_PREFILL_PORT="${LONG_PREFILL_PORT:-30001}"
LONG_DECODE_PORT="${LONG_DECODE_PORT:-31001}"
SHORT_BOOTSTRAP_PORT="${SHORT_BOOTSTRAP_PORT:-8998}"
LONG_BOOTSTRAP_PORT="${LONG_BOOTSTRAP_PORT:-8999}"
ROUTER_PORT="${ROUTER_PORT:-20000}"

SHORT_PREFILL_CUDA_VISIBLE_DEVICES="${SHORT_PREFILL_CUDA_VISIBLE_DEVICES:-0}"
SHORT_DECODE_CUDA_VISIBLE_DEVICES="${SHORT_DECODE_CUDA_VISIBLE_DEVICES:-1}"
LONG_PREFILL_CUDA_VISIBLE_DEVICES="${LONG_PREFILL_CUDA_VISIBLE_DEVICES:-2}"
LONG_DECODE_CUDA_VISIBLE_DEVICES="${LONG_DECODE_CUDA_VISIBLE_DEVICES:-3}"

PREFILL_LONG_THRESHOLD="${PREFILL_LONG_THRESHOLD:-512}"
DECODE_LONG_THRESHOLD="${DECODE_LONG_THRESHOLD:-512}"
GSM8K_MIN_ACCURACY="${GSM8K_MIN_ACCURACY:-0.50}"
MMLU_MIN_SCORE="${MMLU_MIN_SCORE:-0.50}"
MMLU_NUM_EXAMPLES="${MMLU_NUM_EXAMPLES:-200}"
MMLU_NUM_THREADS="${MMLU_NUM_THREADS:-32}"
MMLU_MAX_TOKENS="${MMLU_MAX_TOKENS:-512}"
MMLU_REPEAT="${MMLU_REPEAT:-1}"
MMLU_CHAT_TEMPLATE_KWARGS="${MMLU_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-32}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-4}"
SMOKE_REQUEST_TIMEOUT_SECS="${SMOKE_REQUEST_TIMEOUT_SECS:-300}"
RUN_BENCH_SERVING="${RUN_BENCH_SERVING:-1}"
RUN_ACCURACY_EVALS="${RUN_ACCURACY_EVALS:-1}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark/router/light_router_logs}"
mkdir -p "$LOG_DIR"

pids=()

die() {
  echo "error: $*" >&2
  exit 1
}

cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

split_gpu_ids() {
  local raw="$1"
  tr ',' '\n' <<<"$raw" | sed '/^[[:space:]]*$/d'
}

check_gpu_ids_unique() {
  local all_ids duplicate
  all_ids="$(
    split_gpu_ids "$SHORT_PREFILL_CUDA_VISIBLE_DEVICES"
    split_gpu_ids "$SHORT_DECODE_CUDA_VISIBLE_DEVICES"
    split_gpu_ids "$LONG_PREFILL_CUDA_VISIBLE_DEVICES"
    split_gpu_ids "$LONG_DECODE_CUDA_VISIBLE_DEVICES"
  )"
  duplicate="$(sort <<<"$all_ids" | uniq -d | head -n 1)"
  [[ -z "$duplicate" ]] || die "GPU id ${duplicate} is assigned to multiple roles"
}

check_bench_serving_flags() {
  local help_text
  help_text="$(python3 -m sglang.bench_serving --help 2>&1)"
  grep -q -- "--random-input-len" <<<"$help_text" || die "bench_serving missing --random-input-len"
  grep -q -- "--random-output-len" <<<"$help_text" || die "bench_serving missing --random-output-len"
  grep -q -- "--random-range-ratio" <<<"$help_text" || die "bench_serving missing --random-range-ratio"
  grep -q -- "--ready-check-timeout-sec" <<<"$help_text" || die "bench_serving missing --ready-check-timeout-sec"
  grep -q -- "--tokenize-prompt" <<<"$help_text" || die "bench_serving missing --tokenize-prompt"
}

check_mmlu_model_arg() {
  python3 - "$0" <<'PY'
import sys

script_path = sys.argv[1]
lines = open(script_path, encoding="utf-8").read().splitlines()
target = "python3 -m " + "sglang.test.run_eval"
eval_flag = "--eval-name " + "mmlu"
model_flag = "--model " + '"$MODEL_NAME"'
for index, line in enumerate(lines):
    if target not in line:
        continue
    command_block = "\n".join(lines[index : index + 12])
    if eval_flag in command_block and model_flag in command_block:
        raise SystemExit(0)
raise SystemExit("MMLU validation must pass --model to avoid /v1/models")
PY
}

wait_http() {
  local name="$1"
  local url="$2"
  local timeout_secs="${3:-600}"
  local start
  start="$(date +%s)"
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( "$(date +%s)" - start > timeout_secs )); then
      die "${name} did not become ready at ${url}"
    fi
    sleep 2
  done
}

common_sglang_args() {
  local args=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --page-size 64
    --tp "$TP"
    --mem-fraction-static 0.85
    --nnodes 1
    --node-rank 0
  )
  if [[ -n "$RDMA_DEVICES" ]]; then
    args+=(--disaggregation-ib-device "$RDMA_DEVICES")
  fi
  printf '%q ' "${args[@]}"
}

start_prefill() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local bootstrap_port="$4"
  local dist_init_addr="$5"
  local log_file="${LOG_DIR}/${name}.log"
  local common
  common="$(common_sglang_args)"
  CUDA_VISIBLE_DEVICES="$gpu" bash -lc "${common} --port ${port} --prefill-round-robin-balance --disaggregation-mode prefill --disaggregation-bootstrap-port ${bootstrap_port} --dist-init-addr ${dist_init_addr}" >"$log_file" 2>&1 &
  pids+=("$!")
}

start_decode() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local bootstrap_port="$4"
  local dist_init_addr="$5"
  local log_file="${LOG_DIR}/${name}.log"
  local common
  common="$(common_sglang_args)"
  CUDA_VISIBLE_DEVICES="$gpu" bash -lc "${common} --port ${port} --disaggregation-mode decode --disaggregation-bootstrap-port ${bootstrap_port} --dist-init-addr ${dist_init_addr}" >"$log_file" 2>&1 &
  pids+=("$!")
}

start_router() {
  local log_file="${LOG_DIR}/router.log"
  cargo run --release --manifest-path "$ROUTER_MANIFEST" --bin sglang-light-router -- \
    --pd-disaggregation \
    --prefill "short=http://127.0.0.1:${SHORT_PREFILL_PORT},bootstrap=${SHORT_BOOTSTRAP_PORT}" \
    --decode "short=http://127.0.0.1:${SHORT_DECODE_PORT}" \
    --prefill "long=http://127.0.0.1:${LONG_PREFILL_PORT},bootstrap=${LONG_BOOTSTRAP_PORT}" \
    --decode "long=http://127.0.0.1:${LONG_DECODE_PORT}" \
    --prefill-long-threshold "$PREFILL_LONG_THRESHOLD" \
    --decode-long-threshold "$DECODE_LONG_THRESHOLD" \
    --enable-routing-debug-headers \
    --port "$ROUTER_PORT" >"$log_file" 2>&1 &
  pids+=("$!")
}

chat_request() {
  local prompt="$1"
  local max_tokens="$2"
  local headers_file="$3"
  local body_file="$4"
  curl -fsS "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
    --max-time "$SMOKE_REQUEST_TIMEOUT_SECS" \
    -D "$headers_file" \
    -o "$body_file" \
    -H 'Content-Type: application/json' \
    -H 'X-SGLang-Routing-Key: validation' \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}"
}

header_value() {
  local headers_file="$1"
  local header_name="$2"
  grep -i "^${header_name}:" "$headers_file" | tail -n 1 | sed -E 's/^[^:]+:[[:space:]]*//; s/\r$//'
}

assert_header_equals() {
  local headers_file="$1"
  local header_name="$2"
  local expected="$3"
  local actual
  actual="$(header_value "$headers_file" "$header_name")"
  [[ "$actual" == "$expected" ]] || die "expected ${header_name}=${expected}, got ${actual:-<missing>}"
}

run_smoke_tests() {
  local short_headers="${LOG_DIR}/short_smoke.headers"
  local short_body="${LOG_DIR}/short_smoke.json"
  local long_headers="${LOG_DIR}/long_smoke.headers"
  local long_body="${LOG_DIR}/long_smoke.json"

  chat_request "short validation request" 16 "$short_headers" "$short_body"
  assert_header_equals "$short_headers" "x-sglang-prefill-group" "short"
  assert_header_equals "$short_headers" "x-sglang-decode-group" "short"
  assert_header_equals "$short_headers" "x-sglang-prefill-policy-branch" "load_based_fallback"
  assert_header_equals "$short_headers" "x-sglang-decode-policy-branch" "load_based_fallback"

  chat_request "$(printf 'long %.0s' {1..700})" 32 "$long_headers" "$long_body"
  assert_header_equals "$long_headers" "x-sglang-prefill-group" "long"
  assert_header_equals "$long_headers" "x-sglang-decode-group" "long"
}

fetch_stats() {
  local output_file="$1"
  curl -fsS "http://127.0.0.1:${ROUTER_PORT}/debug/routing_stats" -o "$output_file"
}

assert_stats_contract() {
  local stats_file="$1"
  local min_short_routes="$2"
  local min_long_routes="$3"
  python3 - "$stats_file" "$min_short_routes" "$min_long_routes" <<'PY'
import json
import sys

stats_path = sys.argv[1]
min_short = int(sys.argv[2])
min_long = int(sys.argv[3])
stats = json.load(open(stats_path, encoding="utf-8"))

routes = stats.get("route_counts", {})
if routes.get("prefill=short,decode=short", 0) < min_short:
    raise SystemExit("short route count below expected minimum")
if routes.get("prefill=long,decode=long", 0) < min_long:
    raise SystemExit("long route count below expected minimum")
unexpected_routes = {
    key: value
    for key, value in routes.items()
    if key not in {"prefill=short,decode=short", "prefill=long,decode=long"}
    and value > 0
}
if unexpected_routes:
    raise SystemExit(f"unexpected route counts indicate wrong-group routing: {unexpected_routes}")

policy_counts = stats.get("policy_counts", {})
for role in ("prefill", "decode"):
    if role not in policy_counts:
        raise SystemExit(f"missing policy counts for {role}")
    if not policy_counts[role]:
        raise SystemExit(f"empty policy counts for {role}")

engines = stats.get("engines", [])
required_ids = {
    "short-prefill-0",
    "short-decode-0",
    "long-prefill-1",
    "long-decode-1",
}
seen = {engine.get("id"): engine for engine in engines}
missing = required_ids - set(seen)
if missing:
    raise SystemExit(f"missing engine stats: {sorted(missing)}")

for engine_id, engine in seen.items():
    if engine_id in required_ids and engine.get("local_dispatch_total", 0) <= 0:
        raise SystemExit(f"{engine_id} has no local dispatches")
    if engine_id in required_ids and "last_load_poll_ok" not in engine:
        raise SystemExit(f"{engine_id} missing last_load_poll_ok")
    if engine_id in required_ids and engine.get("last_load_poll_ok") is not True:
        raise SystemExit(f"{engine_id} load polling has not succeeded")
    if engine_id in required_ids and "reported_token_usage" not in engine:
        raise SystemExit(f"{engine_id} missing reported_token_usage")
    if engine_id in required_ids and engine.get("reported_token_usage") is None:
        raise SystemExit(f"{engine_id} missing parsed token usage")
    if engine_id in required_ids and engine.get("reported_total_tokens") is None:
        raise SystemExit(f"{engine_id} missing parsed total tokens")
PY
}

wait_for_stats_contract() {
  local stats_file="${LOG_DIR}/routing_stats.json"
  local start
  start="$(date +%s)"
  until fetch_stats "$stats_file" && assert_stats_contract "$stats_file" 1 1; do
    if (( "$(date +%s)" - start > 60 )); then
      die "router stats did not satisfy smoke contract"
    fi
    sleep 2
  done
}

run_bench_serving() {
  [[ "$RUN_BENCH_SERVING" == "1" ]] || return 0

  python3 -m sglang.bench_serving \
    --backend sglang \
    --model "$MODEL_NAME" \
    --dataset-name random \
    --host 127.0.0.1 \
    --port "$ROUTER_PORT" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --request-rate "$BENCH_REQUEST_RATE" \
    --ready-check-timeout-sec 0 \
    --random-input-len 128 \
    --random-output-len 64 \
    --random-range-ratio 1.0 \
    --tokenize-prompt \
    --output-file "${LOG_DIR}/bench_short.jsonl" >"${LOG_DIR}/bench_short.out" 2>&1

  python3 -m sglang.bench_serving \
    --backend sglang \
    --model "$MODEL_NAME" \
    --dataset-name random \
    --host 127.0.0.1 \
    --port "$ROUTER_PORT" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --request-rate "$BENCH_REQUEST_RATE" \
    --ready-check-timeout-sec 0 \
    --random-input-len 700 \
    --random-output-len 128 \
    --random-range-ratio 1.0 \
    --tokenize-prompt \
    --output-file "${LOG_DIR}/bench_long.jsonl" >"${LOG_DIR}/bench_long.out" 2>&1

  fetch_stats "${LOG_DIR}/routing_stats_after_bench.json"
  assert_stats_contract "${LOG_DIR}/routing_stats_after_bench.json" 2 2
}

run_accuracy_evals() {
  [[ "$RUN_ACCURACY_EVALS" == "1" ]] || return 0
  python3 "${ROOT_DIR}/benchmark/gsm8k/bench_sglang.py" \
    --port "$ROUTER_PORT" \
    --num-questions 500 \
    --num-shots 24 \
    --parallel 50 >"${LOG_DIR}/gsm8k.out" 2>&1
  python3 -m sglang.test.run_eval \
    --eval-name mmlu \
    --model "$MODEL_NAME" \
    --port "$ROUTER_PORT" \
    --num-examples "$MMLU_NUM_EXAMPLES" \
    --num-threads "$MMLU_NUM_THREADS" \
    --max-tokens "$MMLU_MAX_TOKENS" \
    --repeat "$MMLU_REPEAT" \
    --chat-template-kwargs "$MMLU_CHAT_TEMPLATE_KWARGS" >"${LOG_DIR}/mmlu.out" 2>&1
  python3 - "$GSM8K_MIN_ACCURACY" "$MMLU_MIN_SCORE" "${LOG_DIR}/gsm8k.out" "${LOG_DIR}/mmlu.out" <<'PY'
import re
import sys

gsm_min = float(sys.argv[1])
mmlu_min = float(sys.argv[2])
gsm_text = open(sys.argv[3], encoding="utf-8").read()
mmlu_text = open(sys.argv[4], encoding="utf-8").read()

gsm_match = re.search(r"Accuracy:\s*([0-9.]+)", gsm_text)
if not gsm_match or float(gsm_match.group(1)) < gsm_min:
    raise SystemExit("GSM8K accuracy below threshold or missing")

mmlu_match = re.search(r"(?:Score|mean):\s*([0-9.]+)", mmlu_text)
if not mmlu_match or float(mmlu_match.group(1)) < mmlu_min:
    raise SystemExit("MMLU score below threshold or missing")
PY
}

main() {
  [[ -f "$ROUTER_MANIFEST" ]] || die "router manifest not found: ${ROUTER_MANIFEST}"
  check_gpu_ids_unique
  check_bench_serving_flags
  check_mmlu_model_arg

  start_prefill "short-prefill" "$SHORT_PREFILL_CUDA_VISIBLE_DEVICES" "$SHORT_PREFILL_PORT" "$SHORT_BOOTSTRAP_PORT" "127.0.0.1:26000"
  wait_http "short prefill" "http://127.0.0.1:${SHORT_PREFILL_PORT}/health"
  start_decode "short-decode" "$SHORT_DECODE_CUDA_VISIBLE_DEVICES" "$SHORT_DECODE_PORT" "$SHORT_BOOTSTRAP_PORT" "127.0.0.1:26100"
  wait_http "short decode" "http://127.0.0.1:${SHORT_DECODE_PORT}/health"

  start_prefill "long-prefill" "$LONG_PREFILL_CUDA_VISIBLE_DEVICES" "$LONG_PREFILL_PORT" "$LONG_BOOTSTRAP_PORT" "127.0.0.1:26200"
  wait_http "long prefill" "http://127.0.0.1:${LONG_PREFILL_PORT}/health"
  start_decode "long-decode" "$LONG_DECODE_CUDA_VISIBLE_DEVICES" "$LONG_DECODE_PORT" "$LONG_BOOTSTRAP_PORT" "127.0.0.1:26300"
  wait_http "long decode" "http://127.0.0.1:${LONG_DECODE_PORT}/health"

  start_router
  wait_http "router" "http://127.0.0.1:${ROUTER_PORT}/health"
  run_smoke_tests
  wait_for_stats_contract
  run_bench_serving
  run_accuracy_evals
}

main "$@"
