#!/usr/bin/env bash
# Поднимает ДВА движка (обычно SGLang) — по одному на карту — и гоняет на них
# одну команду бенчмарков. Кем приходится второй модели первая — планировщиком,
# верификатором, судьёй — знает только сама команда, поэтому здесь они просто
# первая и вторая. Сервера гасятся на выходе в любом случае, включая Ctrl+C и
# падение.
#
# Одним сервером две модели не отдать: ни vLLM, ни SGLang не держат в процессе
# больше одной модели, поэтому «две модели» здесь — это всегда два процесса и
# два порта. Отсюда и требование двух карт.
#
# URL-ы серверов известны только после старта, а команда бенчмарков задаётся
# заранее (в спеке диспетчера — вообще без shell), поэтому она пишет плейсхолдеры
# в фигурных скобках, а скрипт подставляет в них настоящие значения:
#
#   {BASE_URL} {SECOND_BASE_URL}                     — http://127.0.0.1:<порт>/v1
#   {MODEL} {SECOND_MODEL}                           — путь модели
#   {SERVED_MODEL_NAME} {SECOND_SERVED_MODEL_NAME}   — имя, под которым отдаётся
#   {RUN_ID}                                          — идентификатор прогона
#
# Использование (флаги команды — её собственные; здесь второй моделью служит
# планировщик react-агента):
#   GPU=1,2 MODEL=Qwen/Qwen3.5-9B SECOND_MODEL=Qwen/Qwen3.6-35B-A3B \
#     scripts/run_benchmarks_two_models.sh -- \
#     .venv/bin/python scripts/benchmarks/pavel/run_all_benchmarks.py \
#     --benchmarks aime26 hmmt26 --pipeline react_agent \
#     --model '{MODEL}' --base-url '{BASE_URL}' --api-key EMPTY \
#     --planner-model '{SECOND_MODEL}' --planner-base-url '{SECOND_BASE_URL}' \
#     --planner-api-key EMPTY --run-id my-run-v1
#
# Настройки сервинга те же, что у run_benchmarks.sh (ENGINE, MODEL, PORT,
# MAX_MODEL_LEN, MAX_NUM_SEQS, …). Любую из них можно задать отдельно для
# второго сервера префиксом SECOND_ — без префикса значение общее для обоих.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/bench_common.sh
source "${ROOT}/scripts/bench_common.sh"

die() { echo "$*" >&2; exit 1; }

# Команда бенчмарков: всё после необязательного `--`.
[[ "${1:-}" == "--" ]] && shift
BENCH_CMD=("$@")
[[ ${#BENCH_CMD[@]} -gt 0 ]] \
  || die "Нечего запускать: команда бенчмарков задаётся после '--' (см. шапку скрипта)"

# --- карты -----------------------------------------------------------------
# GPU приходит списком через запятую: у диспетчера — номера арендованных карт,
# при ручном запуске его задаёт человек. Первая карта уходит первой модели,
# вторая — второй; порядок фиксирован, чтобы по логу было видно, кто где.
GPU="${GPU:-}"
[[ -n "${GPU}" ]] || die "GPU не задан: нужны две карты, например GPU=1,2"
IFS=',' read -r -a GPU_LIST <<< "${GPU}"
(( ${#GPU_LIST[@]} >= 2 )) \
  || die "GPU='${GPU}' — нужно две карты (в спеке диспетчера это gpus: 2)"
if (( ${#GPU_LIST[@]} > 2 )); then
  echo ">>> Карт выдано ${#GPU_LIST[@]}, использую первые две: ${GPU_LIST[0]}, ${GPU_LIST[1]}" >&2
fi
FIRST_GPU="${GPU_LIST[0]}"
SECOND_GPU="${GPU_LIST[1]}"

# --- настройки сервера --------------------------------------------------------
# Значение берётся из SECOND_<NAME>, если задано, иначе из <NAME>, иначе из
# умолчания. Так одна переменная (например MAX_MODEL_LEN) настраивает сразу оба
# сервера, а разойтись они могут только там, где это написано явно.
setting() {
  local prefix="$1" name="$2" fallback="${3-}" scoped="$1$2"
  if [[ -n "${prefix}" && -n "${!scoped:-}" ]]; then
    printf '%s' "${!scoped}"
    return 0
  fi
  printf '%s' "${!name:-${fallback}}"
}

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SECOND_MODEL="${SECOND_MODEL:-Qwen/Qwen3.6-35B-A3B}"
# Имя по умолчанию — сам путь модели: клиент бенчмарков передаёт в `--model`
# именно его, и разъезд имени с тем, что отдаёт сервер, стоит 404 на первом же
# запросе.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
SECOND_SERVED_MODEL_NAME="${SECOND_SERVED_MODEL_NAME:-${SECOND_MODEL}}"

PORT="${PORT:-8137}"
SECOND_PORT="${SECOND_PORT:-$(( PORT + 1 ))}"
[[ "${PORT}" != "${SECOND_PORT}" ]] || die "PORT и SECOND_PORT совпадают (${PORT})"

BASE_URL="http://127.0.0.1:${PORT}/v1"
SECOND_BASE_URL="http://127.0.0.1:${SECOND_PORT}/v1"

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"   # первая загрузка тянет десятки ГБ весов
KEEP_SERVER="${KEEP_SERVER:-0}"

# run_id нужен заранее: только зная его, можно закоммитить РОВНО каталог этого
# прогона. Берём из --run-id самой команды, как это делает run_benchmarks.sh.
RUN_ID="${RUN_ID:-$(bench_cmd_opt --run-id)}"
RUN_ID="${RUN_ID:-$(printf '%s' "${SERVED_MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '-')-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "RUN_ID='${RUN_ID}' не проходит RUN_ID_PATTERN — разрешены [A-Za-z0-9._-]"
RESULTS_DIR="results/runs/${RUN_ID}"

cd "${ROOT}"

# --- подстановка плейсхолдеров --------------------------------------------
raw_command="${BENCH_CMD[*]}"
# Адрес, названный в команде явно, ведёт мимо наших серверов: порты выдаёт
# диспетчер. Проверяем до подстановки — после в команде стоят наши же адреса.
require_no_hardcoded_address
substitute_placeholders \
  "BASE_URL=${BASE_URL}" \
  "SECOND_BASE_URL=${SECOND_BASE_URL}" \
  "MODEL=${MODEL}" \
  "SECOND_MODEL=${SECOND_MODEL}" \
  "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  "SECOND_SERVED_MODEL_NAME=${SECOND_SERVED_MODEL_NAME}" \
  "RUN_ID=${RUN_ID}"
# Команда без {BASE_URL} почти наверняка ходит в чужой (или несуществующий)
# сервер: два движка мы подняли, а сказать о них некому.
[[ "${raw_command}" == *"{BASE_URL}"* ]] \
  || echo ">>> ВНИМАНИЕ: в команде нет {BASE_URL} — она не узнает адрес поднятого сервера" >&2
[[ "${raw_command}" == *"{SECOND_BASE_URL}"* ]] \
  || echo ">>> ВНИМАНИЕ: в команде нет {SECOND_BASE_URL} — второй сервер никем не используется" >&2

# --- запуск серверов -------------------------------------------------------
# start_server <роль> <префикс переменных> <карта> <порт> <модель> <имя модели>
#
# Каждый сервер получает свои pid- и log-файлы: serve_*.sh опознают «свой»
# процесс именно по pid-файлу, и с общим они гасили бы друг друга.
start_server() {
  local role="$1" prefix="$2" gpu="$3" port="$4" model="$5" served="$6"
  local engine serve_basename serve_sh mtp speculative max_num_seqs decode_batch

  engine="$(setting "${prefix}" ENGINE sglang)"
  case "${engine}" in
    vllm)
      serve_basename="serve_vllm.sh"
      [[ -x "${ROOT}/.venv/bin/vllm" ]] || die "Нет ${ROOT}/.venv/bin/vllm"
      ;;
    sglang) serve_basename="serve_sglang.sh" ;;
    *) die "ENGINE='${engine}' — ожидалось vllm или sglang" ;;
  esac
  serve_sh="${ROOT}/scripts/qwen35-vllm-bench/${serve_basename}"
  [[ -x "${serve_sh}" ]] || die "Нет исполняемого ${serve_sh}"
  # Запоминаем до старта: гасить и опрашивать сервер нужно тем же скриптом,
  # и знать об этом должен даже тот случай, когда он умрёт через секунду.
  SERVE_SCRIPT_OF["${role}"]="${serve_sh}"

  max_num_seqs="$(setting "${prefix}" MAX_NUM_SEQS 32)"

  # MTP — число спекулятивных токенов (0 = выключить). Формат конфига общий с
  # vLLM: serve_sglang.sh достаёт из него num_speculative_tokens сам.
  mtp="$(setting "${prefix}" MTP 1)"
  [[ "${mtp}" =~ ^[0-9]+$ ]] || die "MTP='${mtp}' — ожидалось число (0 = выключить)"
  speculative="$(setting "${prefix}" SPECULATIVE_CONFIG "")"
  if (( mtp > 0 )) && [[ -z "${speculative}" ]]; then
    speculative="{\"method\":\"mtp\",\"num_speculative_tokens\":${mtp}}"
  fi

  # Шаг декода подаёт MAX_NUM_SEQS*(1+k) токенов; графы, снятые только до
  # MAX_NUM_SEQS, на таком батче не подойдут и декод уедет на медленный путь.
  decode_batch=$(( max_num_seqs * (1 + mtp) ))

  echo ">>> Стартую ${engine} (${role}): ${model} как '${served}' на GPU ${gpu}, порт ${port}"
  echo ">>> Лог: ${ROOT}/scripts/qwen35-vllm-bench/bench-${role}.log"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    VLLM_BIN="${ROOT}/.venv/bin/vllm" \
    SGLANG_PY="${SGLANG_PY:-${ROOT}/scripts/qwen35-vllm-bench/.venv/bin/python}" \
    VLLM_PID_FILE="${ROOT}/scripts/qwen35-vllm-bench/bench-${role}.pid" \
    VLLM_LOG_FILE="${ROOT}/scripts/qwen35-vllm-bench/bench-${role}.log" \
    MODEL="${model}" \
    SERVED_MODEL_NAME="${served}" \
    PORT="${port}" \
    MAX_MODEL_LEN="$(setting "${prefix}" MAX_MODEL_LEN 73728)" \
    MAX_NUM_SEQS="${max_num_seqs}" \
    MAX_NUM_BATCHED_TOKENS="$(setting "${prefix}" MAX_NUM_BATCHED_TOKENS 8192)" \
    MAX_CUDAGRAPH_CAPTURE_SIZE="$(setting "${prefix}" MAX_CUDAGRAPH_CAPTURE_SIZE "${decode_batch}")" \
    KV_CACHE_DTYPE="$(setting "${prefix}" KV_CACHE_DTYPE fp8_e4m3)" \
    GPU_MEMORY_UTILIZATION="$(setting "${prefix}" GPU_MEMORY_UTILIZATION 0.90)" \
    ENABLE_PREFIX_CACHING="$(setting "${prefix}" ENABLE_PREFIX_CACHING 1)" \
    ENABLE_ASYNC_SCHEDULING="$(setting "${prefix}" ENABLE_ASYNC_SCHEDULING 1)" \
    SPECULATIVE_CONFIG="${speculative}" \
    "${serve_sh}" --background
}

# Роль → как её погасить. Заполняется по мере старта: гасить нужно ровно то, что
# успело подняться, в том числе если второй сервер упал на старте.
declare -A SERVE_SCRIPT_OF=()

stop_servers() {
  local role serve_sh
  if [[ "${KEEP_SERVER}" == "1" ]]; then
    echo ">>> KEEP_SERVER=1 — сервера оставлены: ${BASE_URL} и ${SECOND_BASE_URL}"
    return 0
  fi
  for role in "${!SERVE_SCRIPT_OF[@]}"; do
    serve_sh="${SERVE_SCRIPT_OF[${role}]}"
    echo ">>> Останавливаю сервер (${role})..."
    env VLLM_PID_FILE="${ROOT}/scripts/qwen35-vllm-bench/bench-${role}.pid" \
        VLLM_LOG_FILE="${ROOT}/scripts/qwen35-vllm-bench/bench-${role}.log" \
        "${serve_sh}" --stop || true
  done
}
trap stop_servers EXIT INT TERM

# Прошлый лог сохраняем рядом как .prev: падение на старте иначе нечем разбирать,
# а serve_*.sh дописывают в файл (>>) и копили бы прогоны вперемешку.
rotate_log() {
  local log="$1"
  [[ -s "${log}" ]] && mv -f -- "${log}" "${log}.prev"
  : > "${log}"
}

FIRST_LOG="${ROOT}/scripts/qwen35-vllm-bench/bench-first.log"
SECOND_LOG="${ROOT}/scripts/qwen35-vllm-bench/bench-second.log"
rotate_log "${FIRST_LOG}"
rotate_log "${SECOND_LOG}"

# Оба порта проверяются до старта: движок на занятом порту упадёт на bind, а
# ожидание готовности примет за него чужой сервер, уже отвечающий по адресу.
require_free_port "${PORT}" first
require_free_port "${SECOND_PORT}" second

# Оба сервера стартуют сразу: загрузка весов занимает минуты, и делать её
# последовательно значит удвоить простой карт.
start_server first "" "${FIRST_GPU}" "${PORT}" "${MODEL}" "${SERVED_MODEL_NAME}"
start_server second SECOND_ "${SECOND_GPU}" "${SECOND_PORT}" \
  "${SECOND_MODEL}" "${SECOND_SERVED_MODEL_NAME}"

# --- ожидание готовности ---------------------------------------------------
wait_ready() {
  local role="$1" url="$2" log="$3" serve_sh="${SERVE_SCRIPT_OF[$1]}"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))

  echo ">>> Жду готовности ${role}: ${url} (лог: ${log})"
  until curl -fsS --max-time 5 "${url}/models" >/dev/null 2>&1; do
    if ! env VLLM_PID_FILE="${ROOT}/scripts/qwen35-vllm-bench/bench-${role}.pid" \
             "${serve_sh}" --status >/dev/null 2>&1; then
      echo "Сервер ${role} упал на старте. Хвост лога:" >&2
      tail -n 40 "${log}" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Сервер ${role} не поднялся за ${STARTUP_TIMEOUT}s. Хвост лога:" >&2
      tail -n 40 "${log}" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo ">>> Сервер ${role} готов"
}

wait_ready first "${BASE_URL}" "${FIRST_LOG}"
wait_ready second "${SECOND_BASE_URL}" "${SECOND_LOG}"

# Ответить мог и сосед: порт освобождается и занимается между проверкой и
# стартом. Сверка имени отличает наш сервер от чужого.
require_served_model "${BASE_URL}" "${SERVED_MODEL_NAME}" first
require_served_model "${SECOND_BASE_URL}" "${SECOND_SERVED_MODEL_NAME}" second

# --- прогон ----------------------------------------------------------------
echo ">>> Запускаю бенчмарки (run_id=${RUN_ID})"
printf '    %q' "${BENCH_CMD[@]}"; echo
status=0
"${BENCH_CMD[@]}" || status=$?

echo ">>> Команда завершилась с кодом ${status}"
publish_results "${RESULTS_DIR}" "${RUN_ID}" "${SERVED_MODEL_NAME} + ${SECOND_SERVED_MODEL_NAME}, status=${status}"
# Претензии на карты снимаются в trap stop_servers.
exit "${status}"
