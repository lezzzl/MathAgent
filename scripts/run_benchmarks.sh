#!/usr/bin/env bash
# Поднимает локальный движок (SGLang или vLLM — см. ENGINE) с Qwen3.5-9B, гоняет
# на нём вашу команду бенчмарков и гарантированно гасит сервер на выходе (в том
# числе по Ctrl+C или падению).
#
# Прогон всегда идёт на коде текущей ветки — переключением занимается сам
# пользователь до запуска.
#
# Команда обязательна и пишется после `--`: дефолта у неё нет, потому что что
# именно мерить и каким пайплайном, знает только она сама. Адрес поднятого
# сервера она получает плейсхолдером {BASE_URL} — порт заранее не известен.
#
# Использование:
#   scripts/run_benchmarks.sh -- \
#     .venv/bin/kedro run --pipeline benchmarks \
#     --params benchmarks.model.base_url={BASE_URL},benchmarks.run_id={RUN_ID}
#
#   SERVED_MODEL_NAME=Qwen/Qwen3.5-9B CONCURRENCY=30 PORT=8333 \
#     scripts/run_benchmarks.sh -- \
#     .venv/bin/python scripts/run_all_benchmarks.py --base-url {BASE_URL} ...
#
#   GPU=3 KEEP_SERVER=1 scripts/run_benchmarks.sh -- ...   # оставить сервер живым
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Чтение флагов команды, подстановка плейсхолдеров и публикация результатов —
# общее с run_benchmarks_two_models.sh.
# shellcheck source=scripts/bench_common.sh
source "${ROOT}/scripts/bench_common.sh"

die() { echo "$*" >&2; exit 1; }

# Падение на старте разбирают по хвосту лога, поэтому он идёт вместе с причиной.
die_log() { echo "$*" >&2; tail -n 40 "${VLLM_LOG_FILE}" >&2 || true; exit 1; }

# Все чтения лога идут через это: `|| true` обязателен. Присваивание вида
# x="$(grep ...)" возвращает код grep, и при set -e ненайденная строка молча
# убивает скрипт.
log_tail() { grep -aE "$1" "${VLLM_LOG_FILE}" | tail -n "${2:-1}" || true; }

# Команда бенчмарков: всё после `--`. Дефолта у неё нет — что именно мерить и
# каким пайплайном, знает только сама команда, а молчаливый дефолт означал бы
# прогон не того, что имели в виду (и час карты впустую).
[[ "${1:-}" == "--" ]] && shift
BENCH_CMD=("$@")
[[ ${#BENCH_CMD[@]} -gt 0 ]] \
  || die "Нет команды бенчмарков: напишите её после '--', а адрес сервера в ней — как {BASE_URL}"

# ENGINE=vllm|sglang — оба бэкенда принимают один и тот же набор переменных.
# SGLang нужен там, где важен ненулевой presence_penalty: в vLLM сэмплер на нём
# пересобирает историю токенов каждый шаг, в SGLang — нет.
ENGINE="${ENGINE:-sglang}"
case "${ENGINE}" in
  vllm)   SERVE_BASENAME="serve_vllm.sh" ;;
  sglang)
    SERVE_BASENAME="serve_sglang.sh"
    # SGLang живёт в своём окружении, и serve_sglang.sh ищет его рядом с собой.
    export SGLANG_PY="${SGLANG_PY:-${ROOT}/scripts/qwen35-vllm-bench/.venv/bin/python}"
    ;;
  *) die "ENGINE='${ENGINE}' — ожидалось vllm или sglang" ;;
esac

SERVE_SH="${ROOT}/scripts/qwen35-vllm-bench/${SERVE_BASENAME}"
[[ -x "${SERVE_SH}" ]] || die "Нет исполняемого ${SERVE_SH}"

# Своя команда сама несёт --concurrency и --run-id; вычитываем их оттуда
# (bench_cmd_opt из общей библиотеки), чтобы не дублировать те же числа ещё и в
# env — и не разъехаться с ними.

# --- что и на чём поднимаем ------------------------------------------------
GPU="${GPU:-1}"                       # GPU 0 занята чужими процессами
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"   # должно совпадать с benchmarks.model.name
PORT="${PORT:-8137}"                                  # должно совпадать с benchmarks.model.base_url
BASE_URL="http://127.0.0.1:${PORT}/v1"

# --- нагрузка --------------------------------------------------------------
# CONCURRENCY — сколько задач раннер шлёт параллельно (benchmarks.runtime.concurrency).
# MAX_NUM_SEQS сервера должен быть НЕ МЕНЬШЕ, иначе запросы просто ждут в очереди
# движка. Держим полуторный запас на ретраи, но без лишнего резерва mamba-состояний
# (49.5 MiB на слот) и времени захвата CUDA-графов.
# У своей команды значение берём из её же --concurrency.
CONCURRENCY="${CONCURRENCY:-$(bench_cmd_opt --concurrency)}"
CONCURRENCY="${CONCURRENCY:-32}"

# --- параметры сервинга ----------------------------------------------------
# Префиксный кэш включён: react-луп переотправляет растущий диалог.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-73728}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONCURRENCY * 3 / 2 ))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"

# MTP-спекуляция. На умеренном батче декод упирается в пропускную способность
# HBM, а не в вычисления, — MTP разменивает простаивающий компьют на меньшее
# число шагов. Спекулятивный декод не меняет распределение, так что качество
# перепроверять не нужно.
#
# MTP — это ЧИСЛО спекулятивных токенов (num_speculative_tokens): 0 = выключить,
# 1 = дефолт. У Qwen3.5 всего один MTP-слой (mtp_num_hidden_layers=1), поэтому
# при k>1 движок прогоняет его несколько раз подряд и acceptance падает.
MTP="${MTP:-1}"
[[ "${MTP}" =~ ^[0-9]+$ ]] || die "MTP='${MTP}' — ожидалось число (0 = выключить, 1 = дефолт)"
(( MTP > 1 )) && echo ">>> MTP=${MTP}: у Qwen3.5 один MTP-слой, acceptance при k>1 обычно ниже" >&2

SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG-}"
(( MTP > 0 )) && SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}\}}"

# Шаг декода подаёт MAX_NUM_SEQS*(1+k) токенов; графы, снятые только до
# MAX_NUM_SEQS, на таком батче не подойдут и декод уедет на медленный путь.
# При MTP=0 множитель вырождается в единицу.
DECODE_BATCH=$(( MAX_NUM_SEQS * (1 + MTP) ))
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-${DECODE_BATCH}}"
if (( MAX_CUDAGRAPH_CAPTURE_SIZE < DECODE_BATCH )); then
  echo ">>> ВНИМАНИЕ: MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE} < ${DECODE_BATCH}" >&2
  echo ">>> самые крупные декод-батчи пойдут мимо CUDA-графов" >&2
fi

# run_id нужен заранее: только зная его, можно закоммитить РОВНО каталог этого
# прогона и не задеть остальные правки. Допустимые символы — [A-Za-z0-9._-]
# (run_artifacts.py::RUN_ID_PATTERN), поэтому слэши из имени модели вычищаем.
RUN_ID="${RUN_ID:-$(bench_cmd_opt --run-id)}"
RUN_ID="${RUN_ID:-$(printf '%s' "${SERVED_MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '-')-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "RUN_ID='${RUN_ID}' не проходит RUN_ID_PATTERN — разрешены [A-Za-z0-9._-]"
RESULTS_DIR="results/runs/${RUN_ID}"

# Автопуш результатов в origin на текущую ветку. PUSH_RESULTS=0 — только локальный
# коммит; PUSH_RESULTS=none — не коммитить вообще.
PUSH_RESULTS="${PUSH_RESULTS:-1}"

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"   # первая загрузка тянет ~19 ГБ весов
KEEP_SERVER="${KEEP_SERVER:-0}"

# Свои pid/log файлы, чтобы не конфликтовать с вручную поднятым сервером.
export VLLM_PID_FILE="${VLLM_PID_FILE:-${ROOT}/scripts/qwen35-vllm-bench/bench-server.pid}"
export VLLM_LOG_FILE="${VLLM_LOG_FILE:-${ROOT}/scripts/qwen35-vllm-bench/bench-server.log}"

cd "${ROOT}"

# SGLang поднимается своим интерпретатором (SGLANG_PY) и vllm-бинарь не трогает.
[[ "${ENGINE}" != "vllm" || -x "${ROOT}/.venv/bin/vllm" ]] || die "Нет ${ROOT}/.venv/bin/vllm"
# Проверки на kedro тут больше нет: чем гонять бенчмарки, решает сама команда, а
# она может быть и обычным python-скриптом.

server_started=0

stop_server() {
  (( server_started )) || return 0
  if [[ "${KEEP_SERVER}" == "1" ]]; then
    echo ">>> KEEP_SERVER=1 — сервер оставлен: ${BASE_URL} (стоп: ${SERVE_SH} --stop)"
    return 0
  fi
  echo ">>> Останавливаю ${ENGINE}..."
  "${SERVE_SH}" --stop || true
}
trap stop_server EXIT INT TERM

# Чистим лог перед каждым запуском: оба serve_*.sh дописывают в него (>>), и без
# этого в файле копятся прогоны разных движков — а проверки конфига и строка про
# KV-кэш читают ЛОГ, то есть могли бы поймать чужой, давно умерший сервер.
# Прошлый лог сохраняем рядом как .prev — падение на старте иначе нечем разбирать.
[[ -s "${VLLM_LOG_FILE}" ]] && mv -f -- "${VLLM_LOG_FILE}" "${VLLM_LOG_FILE}.prev"
: > "${VLLM_LOG_FILE}"

# Порт проверяется до старта: иначе движок упадёт на bind, а цикл ожидания ниже
# примет за него чужой сервер, который на этом порту уже отвечает.
require_free_port "${PORT}" "${ENGINE}"

# Зашитый в команде адрес обесценил бы обе проверки порта: сервер мы поднимем
# свой, а запросы уйдут мимо него. Проверяем до подстановки плейсхолдеров —
# после неё в команде стоят наши же адреса.
require_no_hardcoded_address

echo ">>> Стартую ${ENGINE}: ${MODEL} как '${SERVED_MODEL_NAME}' на GPU ${GPU}, порт ${PORT}"
echo ">>> Лог: ${VLLM_LOG_FILE} (предыдущий — ${VLLM_LOG_FILE}.prev)"
export CUDA_VISIBLE_DEVICES="${GPU}" VLLM_BIN="${ROOT}/.venv/bin/vllm"
export MODEL SERVED_MODEL_NAME PORT MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
       MAX_CUDAGRAPH_CAPTURE_SIZE SPECULATIVE_CONFIG KV_CACHE_DTYPE \
       GPU_MEMORY_UTILIZATION ENABLE_PREFIX_CACHING ENABLE_ASYNC_SCHEDULING
"${SERVE_SH}" --background
server_started=1

echo ">>> Жду готовности ${BASE_URL} (лог: ${VLLM_LOG_FILE})"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; do
  "${SERVE_SH}" --status >/dev/null 2>&1 || die_log "${ENGINE} упал на старте. Хвост лога:"
  (( SECONDS < deadline )) || die_log "${ENGINE} не поднялся за ${STARTUP_TIMEOUT}s. Хвост лога:"
  sleep 5
done
# Ответил кто-то на нашем порту — но наш ли это движок: порт мог освободиться и
# достаться соседу между проверкой выше и стартом сервера.
require_served_model "${BASE_URL}" "${SERVED_MODEL_NAME}" "${ENGINE}"

echo ">>> Движок (${ENGINE}) готов. Размер KV-кэша:"
log_tail 'GPU KV cache size|[Mm]aximum concurrency|max_total_num_tokens' 2

# Сверяем, что движок принял то, что мы просили: молчаливо проигнорированный
# --speculative-config уже стоил одного прогона на 7% утилизации GPU.
if [[ "${ENGINE}" == "vllm" ]]; then
  engine_line="$(log_tail 'Initializing a V1 LLM engine')"
  if [[ -n "${engine_line}" ]]; then
    echo ">>> Движок поднялся с:"
    for field in speculative_config max_cudagraph_capture_size enable_prefix_caching kv_cache_dtype; do
      value="$(printf '%s' "${engine_line}" | grep -oE "'?${field}'?[:=] ?[^,}]+" | head -n 1 || true)"
      echo "      ${value:-${field}: <не найдено>}"
    done
    if (( MTP > 0 )) && printf '%s' "${engine_line}" | grep -q "speculative_config=None"; then
      echo ">>> ВНИМАНИЕ: просили MTP=${MTP}, а движок стартовал без спекуляции" >&2
      echo ">>> декод будет заметно медленнее ожидаемого" >&2
    fi
  fi
else
  spec_line="$(log_tail 'speculative')"
  echo ">>> Спекуляция: ${spec_line:-<в логе не видно>}"
  if (( MTP > 0 )) && ! grep -qa "NEXTN\|speculative_algorithm" "${VLLM_LOG_FILE}"; then
    echo ">>> ВНИМАНИЕ: просили MTP=${MTP}, а следов спекуляции в логе нет" >&2
  fi
fi

# Плейсхолдеры своей команды: адрес сервера и run_id она не может знать заранее
# (в спеке диспетчера run_id — это имя файла, а `$(...)` запрещён вовсе).
substitute_placeholders \
  "BASE_URL=${BASE_URL}" \
  "MODEL=${MODEL}" \
  "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  "RUN_ID=${RUN_ID}"

echo ">>> Запускаю бенчмарки (run_id=${RUN_ID})"
printf '    %q' "${BENCH_CMD[@]}"; echo
status=0
"${BENCH_CMD[@]}" || status=$?

echo ">>> Команда завершилась с кодом ${status}"
publish_results "${RESULTS_DIR}" "${RUN_ID}" "${SERVED_MODEL_NAME}, status=${status}"
# Претензии на GPU снимаются в trap stop_server.
exit "${status}"
