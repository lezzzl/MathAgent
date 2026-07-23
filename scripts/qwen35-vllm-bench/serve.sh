#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
SELF_PID="${BASHPID}"
PID_FILE="${VLLM_PID_FILE:-${SCRIPT_DIR}/server.pid}"
LOG_FILE="${VLLM_LOG_FILE:-${SCRIPT_DIR}/server.log}"
STOP_TIMEOUT="${VLLM_STOP_TIMEOUT:-30}"

# Environment variables are intentionally used for the tuning knobs so that
# sweep.py can restart this server with reproducible configurations.
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-${MAX_NUM_SEQS}}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"

export CUDA_VISIBLE_DEVICES

if [[ -x "${SCRIPT_DIR}/.venv/bin/vllm" ]]; then
  VLLM_BIN="${VLLM_BIN:-${SCRIPT_DIR}/.venv/bin/vllm}"
else
  VLLM_BIN="${VLLM_BIN:-vllm}"
fi

args=(
  "${VLLM_BIN}" serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --dtype "${DTYPE}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  --enable-chunked-prefill
  --language-model-only
  --reasoning-parser qwen3
)

if [[ "${ENABLE_ASYNC_SCHEDULING}" == "1" ]]; then
  args+=(--async-scheduling)
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi

if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
  args+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

get_managed_pid() {
  local pid command_line
  [[ -r "${PID_FILE}" ]] || return 1
  read -r pid < "${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [[ "${command_line}" == *"serve.sh"* ]] || return 1
  printf '%s\n' "${pid}"
}

write_pid_file() {
  printf '%s\n' "${SELF_PID}" > "${PID_FILE}.tmp.${SELF_PID}"
  mv -f "${PID_FILE}.tmp.${SELF_PID}" "${PID_FILE}"
}

remove_own_pid_file() {
  local recorded_pid=""
  if [[ -r "${PID_FILE}" ]]; then
    read -r recorded_pid < "${PID_FILE}" || true
  fi
  if [[ "${recorded_pid}" == "${SELF_PID}" ]]; then
    rm -f -- "${PID_FILE}"
  fi
}

child_pid=""

shutdown_child() {
  [[ -n "${child_pid}" ]] || return 0
  kill -0 "${child_pid}" 2>/dev/null || return 0

  echo "Stopping vLLM process group ${child_pid}..." >&2
  kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true

  local deadline=$((SECONDS + STOP_TIMEOUT))
  while kill -0 "${child_pid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.2
  done

  if kill -0 "${child_pid}" 2>/dev/null; then
    echo "vLLM did not stop within ${STOP_TIMEOUT}s; sending SIGKILL." >&2
    kill -KILL -- "-${child_pid}" 2>/dev/null || kill -KILL "${child_pid}" 2>/dev/null || true
  fi
  wait "${child_pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  shutdown_child
  remove_own_pid_file
  exit "${status}"
}

run_supervisor() {
  local existing_pid
  existing_pid="$(get_managed_pid 2>/dev/null || true)"
  if [[ -n "${existing_pid}" && "${existing_pid}" != "${SELF_PID}" ]]; then
    echo "A managed vLLM server is already running (supervisor PID ${existing_pid})." >&2
    exit 1
  fi
  [[ -n "${existing_pid}" ]] || rm -f -- "${PID_FILE}"
  write_pid_file

  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap cleanup EXIT

  # A separate session makes the vLLM API server and all engine workers one
  # process group, allowing the supervisor to terminate the whole tree safely.
  setsid "${args[@]}" "$@" &
  child_pid=$!
  echo "vLLM child PID/process-group: ${child_pid}" >&2

  set +e
  wait "${child_pid}"
  local status=$?
  set -e
  exit "${status}"
}

start_background() {
  local existing_pid supervisor_pid
  existing_pid="$(get_managed_pid 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]]; then
    echo "vLLM is already managed by supervisor PID ${existing_pid}." >&2
    exit 1
  fi
  rm -f -- "${PID_FILE}"

  nohup "${SCRIPT_PATH}" --run "$@" >> "${LOG_FILE}" 2>&1 < /dev/null &
  supervisor_pid=$!

  for _ in {1..50}; do
    existing_pid="$(get_managed_pid 2>/dev/null || true)"
    if [[ "${existing_pid}" == "${supervisor_pid}" ]]; then
      echo "Started vLLM supervisor PID ${supervisor_pid}"
      echo "Log: ${LOG_FILE}"
      echo "Stop: ${SCRIPT_PATH} --stop"
      return 0
    fi
    if ! kill -0 "${supervisor_pid}" 2>/dev/null; then
      echo "vLLM failed to start; inspect ${LOG_FILE}" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "Timed out waiting for the vLLM supervisor; inspect ${LOG_FILE}" >&2
  exit 1
}

stop_background() {
  local pid deadline
  pid="$(get_managed_pid 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "No managed vLLM server is running."
    rm -f -- "${PID_FILE}"
    return 0
  fi

  echo "Requesting graceful shutdown of supervisor PID ${pid}..."
  kill -TERM "${pid}"
  deadline=$((SECONDS + STOP_TIMEOUT + 5))
  while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.2
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Supervisor is still stopping; inspect ${LOG_FILE}." >&2
    return 1
  fi
  echo "vLLM stopped."
}

show_status() {
  local pid
  pid="$(get_managed_pid 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "vLLM is not running under this supervisor."
    return 1
  fi
  echo "vLLM supervisor PID ${pid} is running."
  ps -o pid,ppid,pgid,etime,cmd --forest -g "${pid}" 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage:
  ./serve.sh [VLLM_ARGS...]              Run in foreground; Ctrl+C shuts down safely
  ./serve.sh --background [VLLM_ARGS...] Start detached and write to server.log
  ./serve.sh --stop                      Gracefully stop the managed background server
  ./serve.sh --status                    Show supervisor status
  ./serve.sh --logs                      Follow the background log

Environment:
  VLLM_LOG_FILE, VLLM_PID_FILE, VLLM_STOP_TIMEOUT (default: 30 seconds)
EOF
}

case "${1:-}" in
  --background)
    shift
    start_background "$@"
    ;;
  --stop)
    stop_background
    ;;
  --status)
    show_status
    ;;
  --logs)
    exec tail -F -- "${LOG_FILE}"
    ;;
  --help|-h)
    usage
    ;;
  --run)
    shift
    run_supervisor "$@"
    ;;
  *)
    run_supervisor "$@"
    ;;
esac
