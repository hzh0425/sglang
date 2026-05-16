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
  local dist_init_addr="$4"
  local prefill_url="$5"
  local log_file="${LOG_DIR}/${name}.log"
  local common
  common="$(common_sglang_args)"
  CUDA_VISIBLE_DEVICES="$gpu" bash -lc "${common} --port ${port} --disaggregation-mode decode --dist-init-addr ${dist_init_addr} --prefill-server-url ${prefill_url}" >"$log_file" 2>&1 &
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
  curl -fsS "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H 'X-SGLang-Routing-Key: validation' \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}"
}

run_smoke_tests() {
  chat_request "short validation request" 16 >/dev/null
  chat_request "$(printf 'long %.0s' {1..700})" 32 >/dev/null
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
    --port "$ROUTER_PORT" \
    --num-examples 200 \
    --max-tokens 4096 \
    --repeat 4 >"${LOG_DIR}/mmlu.out" 2>&1
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

  start_prefill "short-prefill" "$SHORT_PREFILL_CUDA_VISIBLE_DEVICES" "$SHORT_PREFILL_PORT" "$SHORT_BOOTSTRAP_PORT" "127.0.0.1:26000"
  wait_http "short prefill" "http://127.0.0.1:${SHORT_PREFILL_PORT}/health"
  start_decode "short-decode" "$SHORT_DECODE_CUDA_VISIBLE_DEVICES" "$SHORT_DECODE_PORT" "127.0.0.1:26100" "http://127.0.0.1:${SHORT_PREFILL_PORT}"
  wait_http "short decode" "http://127.0.0.1:${SHORT_DECODE_PORT}/health"

  start_prefill "long-prefill" "$LONG_PREFILL_CUDA_VISIBLE_DEVICES" "$LONG_PREFILL_PORT" "$LONG_BOOTSTRAP_PORT" "127.0.0.1:26200"
  wait_http "long prefill" "http://127.0.0.1:${LONG_PREFILL_PORT}/health"
  start_decode "long-decode" "$LONG_DECODE_CUDA_VISIBLE_DEVICES" "$LONG_DECODE_PORT" "127.0.0.1:26300" "http://127.0.0.1:${LONG_PREFILL_PORT}"
  wait_http "long decode" "http://127.0.0.1:${LONG_DECODE_PORT}/health"

  start_router
  wait_http "router" "http://127.0.0.1:${ROUTER_PORT}/health"
  run_smoke_tests
  run_accuracy_evals
}

main "$@"
