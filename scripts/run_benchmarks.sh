#!/usr/bin/env bash
# Поднимает локальный vLLM с Qwen3.5-9B, гоняет на нём пайплайн benchmarks
# и гарантированно гасит сервер на выходе (в том числе по Ctrl+C или падению).
#
# Прогон всегда идёт на коде текущей ветки — переключением занимается сам
# пользователь до запуска.
#
# Использование:
#   scripts/run_benchmarks.sh                 # дефолт: kedro-пайплайн benchmarks
#   GPU=3 scripts/run_benchmarks.sh
#   KEEP_SERVER=1 scripts/run_benchmarks.sh   # оставить сервер живым после прогона
#
# Своя команда — всё после `--`; сервер, ожидание готовности, остановка и пуш
# результатов работают так же. run_id берётся из `--run-id` этой команды:
#   SERVED_MODEL_NAME=Qwen/Qwen3.5-9B PORT=8333 CONCURRENCY=30 \
#     scripts/run_benchmarks.sh -- \
#     .venv/bin/python scripts/run_all_benchmarks.py --pipeline solver ...
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Команда бенчмарков: всё после необязательного `--`. Пусто → дефолтная kedro.
[[ "${1:-}" == "--" ]] && shift
BENCH_CMD=("$@")

# ENGINE=vllm|sglang — оба бэкенда принимают один и тот же набор переменных.
# SGLang нужен там, где важен ненулевой presence_penalty: в vLLM сэмплер на нём
# пересобирает историю токенов каждый шаг, в SGLang — нет.
ENGINE="${ENGINE:-vllm}"
case "${ENGINE}" in
  vllm)   SERVE_BASENAME="serve_vllm.sh" ;;
  sglang)
    SERVE_BASENAME="serve_sglang.sh"
    # SGLang живёт в своём окружении, и serve_sglang.sh ищет его рядом с собой.
    export SGLANG_PY="${SGLANG_PY:-${ROOT}/scripts/qwen35-vllm-bench/.venv/bin/python}"
    ;;
  *) echo "ENGINE='${ENGINE}' — ожидалось vllm или sglang" >&2; exit 1 ;;
esac

SERVE_SH="${ROOT}/scripts/qwen35-vllm-bench/${SERVE_BASENAME}"
[[ -x "${SERVE_SH}" ]] || { echo "Нет исполняемого ${SERVE_SH}" >&2; exit 1; }

# Своя команда сама несёт --concurrency и --run-id; вычитываем их оттуда, чтобы
# не дублировать те же числа ещё и в env (и не разъехаться с ними).
bench_cmd_opt() {
  local flag="$1" i
  for ((i = 0; i < ${#BENCH_CMD[@]}; i++)); do
    case "${BENCH_CMD[i]}" in
      "${flag}") printf '%s' "${BENCH_CMD[i + 1]:-}" ;;
      "${flag}="*) printf '%s' "${BENCH_CMD[i]#"${flag}"=}" ;;
    esac
  done
}

# --- что и на чём поднимаем ------------------------------------------------
GPU="${GPU:-1}"                       # GPU 0 занята чужими процессами
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"   # должно совпадать с benchmarks.model.name
PORT="${PORT:-8137}"                                  # должно совпадать с benchmarks.model.base_url
BASE_URL="http://127.0.0.1:${PORT}/v1"

# --- нагрузка --------------------------------------------------------------
# CONCURRENCY — сколько задач раннер шлёт параллельно (benchmarks.runtime.concurrency).
# MAX_NUM_SEQS сервера должен быть НЕ МЕНЬШЕ, иначе запросы просто ждут в очереди
# vLLM. Держим полуторный запас на ретраи, но без лишнего резерва mamba-состояний
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
# 1 = дефолт. Раньше здесь сравнивалось `== "1"` при дефолте 2, из-за чего ветка
# была мёртвой и спекуляция молча не включалась ни разу.
# У Qwen3.5 всего один MTP-слой (mtp_num_hidden_layers=1), поэтому при k>1 vLLM
# прогоняет его несколько раз подряд и acceptance падает — 1 почти всегда лучше.
MTP="${MTP:-1}"
if [[ ! "${MTP}" =~ ^[0-9]+$ ]]; then
  echo "MTP='${MTP}' — ожидалось число (0 = выключить, 1 = дефолт)" >&2
  exit 1
fi

if (( MTP > 0 )); then
  SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}\}}"
  # Шаг декода подаёт MAX_NUM_SEQS*(1+k) токенов; графы, снятые только до
  # MAX_NUM_SEQS, на таком батче не подойдут и декод уедет на медленный путь.
  DECODE_BATCH=$(( MAX_NUM_SEQS * (1 + MTP) ))
  (( MTP > 1 )) && echo ">>> MTP=${MTP}: у Qwen3.5 один MTP-слой, acceptance при k>1 обычно ниже" >&2
else
  SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
  DECODE_BATCH="${MAX_NUM_SEQS}"
fi
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-${DECODE_BATCH}}"

if (( MAX_CUDAGRAPH_CAPTURE_SIZE < DECODE_BATCH )); then
  echo ">>> ВНИМАНИЕ: MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE} < ${DECODE_BATCH}" >&2
  echo ">>> самые крупные декод-батчи пойдут мимо CUDA-графов" >&2
fi

# Разделитель `;`, а не `,`: kedro режет значение --params по запятым без учёта
# скобок (см. normalize_select в pipelines/benchmarks/nodes.py).
SELECT="${SELECT:-[aime26;hmmt26;imo_answerbench]}"

# run_id нужен заранее: только зная его, можно закоммитить РОВНО каталог этого
# прогона и не задеть остальные правки. Допустимые символы — [A-Za-z0-9._-]
# (run_artifacts.py::RUN_ID_PATTERN), поэтому слэши из имени модели вычищаем.
RUN_ID="${RUN_ID:-$(bench_cmd_opt --run-id)}"
RUN_ID="${RUN_ID:-$(printf '%s' "${SERVED_MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '-')-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID='${RUN_ID}' не проходит RUN_ID_PATTERN — разрешены [A-Za-z0-9._-]" >&2
  exit 1
fi
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

[[ -x "${ROOT}/.venv/bin/vllm" ]]  || { echo "Нет ${ROOT}/.venv/bin/vllm" >&2; exit 1; }
[[ -x "${ROOT}/.venv/bin/kedro" ]] || { echo "Нет ${ROOT}/.venv/bin/kedro" >&2; exit 1; }

# Коммитит РОВНО каталог этого прогона и пушит его в origin на текущую ветку.
# Пути указаны явно (никаких `git add -A`): незакоммиченные правки в рабочем
# дереве и уже проиндексированные чужие изменения не должны попасть в коммит,
# поэтому и `git commit` вызывается с pathspec.
publish_results() {
  local kedro_status="$1" branch head_before

  [[ "${PUSH_RESULTS}" == "none" ]] && { echo ">>> PUSH_RESULTS=none — результаты не коммичу"; return 0; }
  if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo ">>> ${RESULTS_DIR} не создан — коммитить нечего" >&2
    return 0
  fi

  branch="$(git -C "${ROOT}" symbolic-ref --quiet --short HEAD || true)"
  if [[ -z "${branch}" ]]; then
    echo ">>> HEAD отделён от ветки — результаты оставляю незакоммиченными" >&2
    return 0
  fi

  git -C "${ROOT}" add -- "${RESULTS_DIR}"
  if git -C "${ROOT}" diff --cached --quiet -- "${RESULTS_DIR}"; then
    echo ">>> В ${RESULTS_DIR} нет изменений — коммит не нужен"
    return 0
  fi

  git -C "${ROOT}" commit --quiet -m "bench: ${RUN_ID} (${SELECT}, kedro=${kedro_status})" \
    -- "${RESULTS_DIR}"
  head_before="$(git -C "${ROOT}" rev-parse --short HEAD)"
  echo ">>> Закоммитил ${head_before}: ${RESULTS_DIR}"

  [[ "${PUSH_RESULTS}" == "1" ]] || { echo ">>> PUSH_RESULTS=${PUSH_RESULTS} — пушить не буду"; return 0; }
  if ! git -C "${ROOT}" remote get-url origin >/dev/null 2>&1; then
    echo ">>> Нет remote origin — коммит остался локальным" >&2
    return 0
  fi

  echo ">>> Пушу в origin/${branch}"
  if git -C "${ROOT}" push origin "HEAD:${branch}"; then
    return 0
  fi
  # Ветка уехала вперёд: результаты лежат в каталоге с уникальным run_id,
  # так что ребейз поверх origin конфликтовать не должен. Пробуем ровно раз.
  echo ">>> Push отклонён, делаю rebase на origin/${branch} и пробую ещё раз" >&2
  if git -C "${ROOT}" pull --rebase origin "${branch}" && \
     git -C "${ROOT}" push origin "HEAD:${branch}"; then
    return 0
  fi
  echo ">>> Push не удался. Коммит ${head_before} остался локальным — запушь вручную:" >&2
  echo "    git push origin HEAD:${branch}" >&2
  return 0
}

server_started=0

stop_server() {
  (( server_started )) || return 0
  if [[ "${KEEP_SERVER}" == "1" ]]; then
    echo ">>> KEEP_SERVER=1 — сервер оставлен: ${BASE_URL} (стоп: ${SERVE_SH} --stop)"
    return 0
  fi
  echo ">>> Останавливаю vLLM..."
  VLLM_PID_FILE="${VLLM_PID_FILE}" VLLM_LOG_FILE="${VLLM_LOG_FILE}" \
    "${SERVE_SH}" --stop || true
}
trap stop_server EXIT INT TERM

# Чистим лог перед каждым запуском: serve_vllm.sh дописывает в него (>>), и без этого
# в файле копятся прогоны разных движков — а проверки конфига и строка про KV-кэш
# читают ЛОГ, то есть могли бы поймать чужой, давно умерший сервер.
# Прошлый лог сохраняем рядом как .prev — падение на старте иначе нечем разбирать.
if [[ -s "${VLLM_LOG_FILE}" ]]; then
  mv -f -- "${VLLM_LOG_FILE}" "${VLLM_LOG_FILE}.prev"
fi
: > "${VLLM_LOG_FILE}"

echo ">>> Стартую ${ENGINE}: ${MODEL} как '${SERVED_MODEL_NAME}' на GPU ${GPU}, порт ${PORT}"
echo ">>> Лог: ${VLLM_LOG_FILE} (предыдущий — ${VLLM_LOG_FILE}.prev)"
CUDA_VISIBLE_DEVICES="${GPU}" \
VLLM_BIN="${ROOT}/.venv/bin/vllm" \
MODEL="${MODEL}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
PORT="${PORT}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG}" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING}" \
  "${SERVE_SH}" --background
server_started=1

echo ">>> Жду готовности ${BASE_URL} (лог: ${VLLM_LOG_FILE})"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; do
  if ! VLLM_PID_FILE="${VLLM_PID_FILE}" "${SERVE_SH}" --status >/dev/null 2>&1; then
    echo "${ENGINE} упал на старте. Хвост лога:" >&2
    tail -n 40 "${VLLM_LOG_FILE}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "${ENGINE} не поднялся за ${STARTUP_TIMEOUT}s. Хвост лога:" >&2
    tail -n 40 "${VLLM_LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 5
done
echo ">>> Движок (${ENGINE}) готов. Размер KV-кэша:"
grep -iaE "GPU KV cache size|maximum concurrency|max_total_num_tokens" "${VLLM_LOG_FILE}" | tail -n 2 || true

# Сверяем, что движок принял то, что мы просили: молчаливо проигнорированный
# --speculative-config уже стоил одного прогона на 7% утилизации GPU.
# ВАЖНО: каждый grep здесь через `|| true`. Присваивание вида x="$(grep ...)"
# возвращает код grep, и при set -e ненайденная строка убивает скрипт молча —
# ровно на этом первый запуск SGLang «стартовал и сразу остановился».
if [[ "${ENGINE}" == "vllm" ]]; then
  engine_line="$(grep -a "Initializing a V1 LLM engine" "${VLLM_LOG_FILE}" | tail -n 1 || true)"
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
  spec_line="$(grep -a "speculative" "${VLLM_LOG_FILE}" | tail -n 1 || true)"
  echo ">>> Спекуляция: ${spec_line:-<в логе не видно>}"
  if (( MTP > 0 )) && ! grep -qa "NEXTN\|speculative_algorithm" "${VLLM_LOG_FILE}"; then
    echo ">>> ВНИМАНИЕ: просили MTP=${MTP}, а следов спекуляции в логе нет" >&2
  fi
fi

if [[ ${#BENCH_CMD[@]} -eq 0 ]]; then
  BENCH_CMD=(
    "${ROOT}/.venv/bin/kedro" run --pipeline benchmarks --params
    "benchmarks.pipeline=react,benchmarks.react.tester=true,benchmarks.prompt=conf/base/prompts/react-tools.yml,benchmarks.model.name=${SERVED_MODEL_NAME},benchmarks.model.base_url=${BASE_URL},benchmarks.select=${SELECT},benchmarks.runtime.concurrency=${CONCURRENCY},benchmarks.serving.vllm_max_num_seqs=${MAX_NUM_SEQS},benchmarks.serving.vllm_max_model_len=${MAX_MODEL_LEN},benchmarks.run_id=${RUN_ID}"
  )
fi

echo ">>> Запускаю бенчмарки (run_id=${RUN_ID})"
printf '    %q' "${BENCH_CMD[@]}"; echo
set +e
"${BENCH_CMD[@]}"
status=$?
set -e

echo ">>> kedro завершился с кодом ${status}"
publish_results "${status}"
# Претензии на GPU снимаются в trap stop_server.
exit "${status}"
