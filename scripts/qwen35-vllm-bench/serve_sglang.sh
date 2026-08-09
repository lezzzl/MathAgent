#!/usr/bin/env bash
# SGLang-бэкенд с тем же интерфейсом переменных, что и serve_vllm.sh, чтобы
# run_benchmarks.sh мог переключаться между движками одной переменной.
#
# Зачем: в vLLM 0.26 сэмплер при ненулевом presence/frequency/repetition penalty
# каждый шаг декода пересобирает на CPU паддед-тензор ВСЕЙ истории токенов
# (v1/sample/ops/penalties.py::_convert_to_tensors), из-за чего шаг дорожает с
# ростом длины ответа. SGLang держит накопитель (batch, vocab) и на шаге пишет
# туда только новый токен (penaltylib/presence_penalty.py), то есть стоимость не
# зависит от длины. Для прогонов с presence_penalty=1.5 это принципиально.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
SELF_PID="${BASHPID}"
PID_FILE="${VLLM_PID_FILE:-${SCRIPT_DIR}/server.pid}"
LOG_FILE="${VLLM_LOG_FILE:-${SCRIPT_DIR}/server.log}"
STOP_TIMEOUT="${VLLM_STOP_TIMEOUT:-30}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
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

# SGLang живёт в ОТДЕЛЬНОМ окружении, а не в основном .venv проекта: он тянет
# outlines-core 0.1.26 (колёс под cp313 нет, нужен Rust) и понижает transformers,
# xgrammar, outlines-core и openai — а на openai через langchain-openai завязан
# клиент бенчмарков. Серверу и клиенту делить окружение незачем.
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  SGLANG_PY="${SGLANG_PY:-${SCRIPT_DIR}/.venv/bin/python}"
else
  SGLANG_PY="${SGLANG_PY:-python}"
fi

# find_spec, а НЕ `import sglang`: импорт тянет torch и занимает ~9 с, тогда как
# супервизор ждёт появления pid-файла всего 5 с — проверка успевала съесть окно
# и запуск падал с «Timed out waiting for the SGLang supervisor».
if ! "${SGLANG_PY}" -c "
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('sglang') else 1)" >/dev/null 2>&1; then
  cat >&2 <<EOF
В ${SGLANG_PY} нет пакета sglang.
Поставить в отдельное окружение (нужен именно python3.12 — см. комментарий выше):
  python3.12 -m venv ${SCRIPT_DIR}/.venv
  uv pip install --python ${SCRIPT_DIR}/.venv/bin/python --prerelease=allow 'sglang[all]==0.5.17'
Либо укажите свой интерпретатор через SGLANG_PY=/path/to/python.
EOF
  exit 1
fi

# vLLM-овский --kv-cache-dtype auto означает «как у модели»; в SGLang то же самое.
case "${KV_CACHE_DTYPE}" in
  auto|fp8_e4m3|fp8_e5m2|bf16|bfloat16) ;;
  *) echo "KV_CACHE_DTYPE='${KV_CACHE_DTYPE}' не поддержан SGLang-бэкендом" >&2; exit 1 ;;
esac

args=(
  "${SGLANG_PY}" -m sglang.launch_server
  --model-path "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size 1
  --dtype "${DTYPE}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --context-length "${MAX_MODEL_LEN}"
  --max-running-requests "${MAX_NUM_SEQS}"
  --chunked-prefill-size "${MAX_NUM_BATCHED_TOKENS}"
  --mem-fraction-static "${GPU_MEMORY_UTILIZATION}"
  --cuda-graph-max-bs "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  --reasoning-parser qwen3
)
# Аналога vLLM --language-model-only здесь нет: --enable-multimodal объявлен как
# store_true без парной --no-* формы, так что визуальную башню (~1.3 ГБ) SGLang
# грузит всегда. На фоне ~137 ГБ пула это терпимо, но память не бесплатная.

# Радиксное дерево у SGLang — это и есть префиксный кэш, и оно включено всегда.
if [[ "${ENABLE_PREFIX_CACHING}" != "1" ]]; then
  args+=(--disable-radix-cache)
fi

# SPECULATIVE_CONFIG приходит в формате vLLM; забираем из него число токенов.
# MTP в SGLang — это NEXTN: линейная цепочка драфта (eagle-topk = 1),
# num_steps = k, num_draft_tokens = k + 1.
if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
  spec_k="$(printf '%s' "${SPECULATIVE_CONFIG}" | grep -oE '"num_speculative_tokens"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' || true)"
  spec_k="${spec_k:-1}"
  args+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${spec_k}"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "$(( spec_k + 1 ))"
  )
fi

# ENABLE_ASYNC_SCHEDULING в SGLang соответствует overlap-планировщик, он включён
# по умолчанию; отдельного флага «включить» нет, поэтому переменную не мапим.

get_managed_pid() {
  local pid command_line
  [[ -r "${PID_FILE}" ]] || return 1
  read -r pid < "${PID_FILE}" || return 1
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  # Сверяемся с ИМЕНЕМ ЭТОГО файла, а не с литералом чужого: иначе
  # рукопожатие по pid-файлу для serve_sglang.sh не сходится никогда.
  [[ "${command_line}" == *"$(basename -- "${SCRIPT_PATH}")"* ]] || return 1
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

  echo "Stopping SGLang process group ${child_pid}..." >&2
  kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true

  local deadline=$((SECONDS + STOP_TIMEOUT))
  while kill -0 "${child_pid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.2
  done

  if kill -0 "${child_pid}" 2>/dev/null; then
    echo "SGLang did not stop within ${STOP_TIMEOUT}s; sending SIGKILL." >&2
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
    echo "A managed SGLang server is already running (supervisor PID ${existing_pid})." >&2
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
  echo "SGLang child PID/process-group: ${child_pid}" >&2

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
    echo "SGLang is already managed by supervisor PID ${existing_pid}." >&2
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
      echo "SGLang failed to start; inspect ${LOG_FILE}" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "Timed out waiting for the SGLang supervisor; inspect ${LOG_FILE}" >&2
  exit 1
}

stop_background() {
  local pid deadline
  pid="$(get_managed_pid 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "No managed SGLang server is running."
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
  echo "SGLang stopped."
}

show_status() {
  local pid
  pid="$(get_managed_pid 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "SGLang is not running under this supervisor."
    return 1
  fi
  echo "vLLM supervisor PID ${pid} is running."
  ps -o pid,ppid,pgid,etime,cmd --forest -g "${pid}" 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage:
  ./serve_sglang.sh [SGLANG_ARGS...]              Run in foreground; Ctrl+C shuts down safely
  ./serve_sglang.sh --background [SGLANG_ARGS...] Start detached and write to server.log
  ./serve_sglang.sh --stop                      Gracefully stop the managed background server
  ./serve_sglang.sh --status                    Show supervisor status
  ./serve_sglang.sh --logs                      Follow the background log

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
