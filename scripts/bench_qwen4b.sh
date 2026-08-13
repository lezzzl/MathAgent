#!/usr/bin/env bash
# Прогон пошагового агента qwen4b на одном бенчмарке с одним зерном.
#
#   bash scripts/bench_qwen4b.sh aime26 42
#   MAX_NUM_SEQS=32 bash scripts/bench_qwen4b.sh hmmt_feb2025 43   # ручной прогон
#   bash scripts/bench_qwen4b.sh aime26 42 --limit 2               # дымовой
#   bash scripts/bench_qwen4b.sh aime26 42 --run-id другое-имя
#
# Всё после зерна уходит в раннер как есть (кроме нашего --run-id) — так дымовой
# прогон отличается от боевого одним аргументом в спеке, а не веткой в скрипте.
#
# В спеке диспетчера команда обязана начинаться с python|python3|bash|uv, поэтому
# там пишется `bash scripts/bench_qwen4b.sh ...`, а не путь к скрипту напрямую.
# Сам файл при этом должен быть закоммичен: диспетчер проверяет, что скрипт
# отслеживается git в том коммите, из которого запускается.
#
# Зачем отдельный скрипт, а не всё в спеке диспетчера. Диспетчер пропускает в
# задачу только переменные из своего белого списка — на этом уже споткнулся
# MAX_TOKENS у react-агента. Нам нужны KV_CACHE_DTYPE, MTP и TOOL_CALL_PARSER,
# которых там заведомо нет: последнюю мы вообще только что завели. Внутри своего
# скрипта окружение наше, и белый список ни при чём. Побочная польза — золотая
# конфигурация записана в одном месте, а не размножена по четырём спекам.
#
# Значения ниже воспроизводят прогон 2026-08-07, давший AIME26 28/30
# (12.39M токенов, 30.3 машино-часа). Любое из них переопределяется извне, но
# тогда это уже другая конфигурация — и сравнивать её с 28/30 нельзя.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# --- внутренний режим ------------------------------------------------------
# Команда бенчмарков исполняется уже при поднятом сервере, и проверка живого
# сервера (probe_thinking) возможна только там. Поэтому скрипт зовёт сам себя:
# сначала как обвязку, потом как саму команду. Отдельный файл ради двух функций
# заводить не хочется, а `bash -c` внутри argv читается хуже этого флага.
INNER=0
if [[ "${1:-}" == "--inner" ]]; then
  INNER=1
  shift
fi

BENCH="${1:?первый аргумент — бенчмарк: aime24 | aime25 | aime26 | hmmt_feb2025}"
SEED="${2:?второй аргумент — зерно, например 42}"
shift 2   # остаток — доп. флаги раннера

# --run-id вынимаем из остатка: это НАШ флаг, раннер его не знает. Нужен, когда
# имя спеки разошлось с шаблоном ниже — например после переименования из-за
# занятого run_id. Ошибиться тут дорого: каталог результатов должен называться
# ровно так же, как yml-файл, иначе диспетчер их не найдёт.
EXPLICIT_RUN_ID=""
PASSTHROUGH=()
while (( $# )); do
  case "$1" in
    --run-id) EXPLICIT_RUN_ID="${2:?--run-id без значения}"; shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
set -- "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

RUNNER="${ROOT}/scripts/benchmarks/run_${BENCH}.py"
[[ -f "${RUNNER}" ]] || { echo "Нет ${RUNNER}" >&2; exit 1; }

# --- конфигурация прогона --------------------------------------------------
PORT="${PORT:-8137}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
# Имя = путь модели: клиент передаёт в --model именно его, и разъезд с тем, что
# отдаёт сервер, стоит 404 на первом же запросе.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# run_id — это имя yml-спеки, и по нему диспетчер забирает каталог результатов.
# Он кладёт его в окружение, но под каким именем — снаружи не видно (в его тестах
# переменная скрыта за константой RUN_ID_ENV), поэтому смотрим оба правдоподобных
# имени. Явный --run-id старше всего: он не зависит от догадок.
RUN_ID="${EXPLICIT_RUN_ID:-${RUN_ID:-${MATHAGENT_RUN_ID:-nikita-qwen4b-v11-${BENCH}-s${SEED}}}}"
OUTPUT="${ROOT}/results/runs/${RUN_ID}/${BENCH}.jsonl"

PROMPT="${PROMPT:-conf/base/prompts/agent-step-qwen4b-v11.yml}"
WORKERS="${WORKERS:-16}"

if (( ! INNER )); then
  # --- обвязка: поднимаем сервер и уходим в run_benchmarks.sh --------------
  # SGLang, а не vLLM: на нём прогоны идут быстрее, и на нём считает вся группа.
  export ENGINE="${ENGINE:-sglang}"
  export MODEL SERVED_MODEL_NAME PORT RUN_ID

  # 65536 — как в докере того прогона. Ниже 52000 лимит генератора (40000) не
  # влезает, раннер его урежет и шаги начнут обрываться на середине.
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

  # Дефолт run_benchmarks.sh — fp8_e4m3. Квантованный KV-кэш меняет численность,
  # а мы воспроизводим замер, а не ускоряем его.
  export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

  # Спекулятивный декод (в SGLang — NEXTN) у 4B никто не проверял, а на модели
  # без MTP-слоя сервер просто не поднимется. Ускорение того не стоит.
  export MTP="${MTP:-0}"

  # §2.18 памяти: на hermes сервер не разобрал НИ ОДНОГО вызова инструмента из
  # 228 — всё уходило в клиентский фоллбек; на qwen3_coder разобрал все.
  export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

  echo ">>> ${BENCH}, зерно ${SEED}, run_id=${RUN_ID}"
  echo ">>> результаты: ${OUTPUT}"
  # Путь абсолютный, а не $0: run_benchmarks.sh перед запуском команды делает
  # cd в корень, и относительный путь зависел бы от того, откуда позвали нас.
  # run_id передаём явно: во внутреннем вызове он должен быть тем же, что здесь,
  # независимо от того, что там окажется в окружении.
  exec "${ROOT}/scripts/run_benchmarks.sh" -- \
    /usr/bin/env bash "${ROOT}/scripts/bench_qwen4b.sh" --inner "${BENCH}" "${SEED}" \
    --run-id "${RUN_ID}" "$@"
fi

# --- внутренний режим: сервер уже готов ------------------------------------

# Вся конфигурация v11 держится на per-role enable_thinking, а он едет на сервер
# в chat_template_kwargs. Если движок это поле игнорирует, размышления включатся
# у всех ролей разом — прогон отработает молча и даст числа от другой
# конфигурации. Ровно так мы уже потеряли две недели (§2.9а), когда --thinking on
# незаметно перекрывал enable_thinking: false у оценщика. Две секунды проверки.
probe_thinking() {
  "${ROOT}/.venv/bin/python" - "${BASE_URL}/chat/completions" "${SERVED_MODEL_NAME}" <<'PY'
import json, sys, urllib.request

url, model = sys.argv[1], sys.argv[2]


def reasoning_len(flag):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with the number only."}],
        "max_tokens": 200,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": flag},
    }
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        message = json.load(response)["choices"][0]["message"]
    return len((message.get("reasoning_content") or message.get("reasoning") or "").strip())


off, on = reasoning_len(False), reasoning_len(True)
if off == 0 < on:
    print(f"[probe] enable_thinking УЧТЁН: reasoning_content off=0, on={on} симв.")
else:
    print(f"[probe] ⚠️  enable_thinking ПРОИГНОРИРОВАН: off={off}, on={on} симв.")
    print("[probe] ⚠️  per-role настройки из yml не доехали — это НЕ та конфигурация, "
          "на которой мерили 28/30. Числа прогона сравнивать с ней нельзя.")
PY
}
probe_thinking || echo "[probe] проверка не удалась — прогон продолжаю"

mkdir -p -- "$(dirname -- "${OUTPUT}")"

# Золотой набор (§ changelog 2026-08-06): v11, single, samples 3 с порогом
# добора 250k, без верификатора, бюджет 800k. --thinking не указан намеренно: на
# v11 дефолтный auto даёт нужное, а попытка сэкономить на размышлениях оценщика
# стоила 25/30 → 22/30 при том же расходе (§2.9б).
exec "${ROOT}/.venv/bin/python" -u "${RUNNER}" \
  --model "${SERVED_MODEL_NAME}" \
  --base-url "${BASE_URL}" \
  --api-key EMPTY \
  --pipeline qwen4b \
  --prompt "${PROMPT}" \
  --branch-mode single \
  --seed "${SEED}" \
  --samples 3 \
  --resample-token-threshold 250000 \
  --no-verify-step \
  --token-budget 800000 \
  --timeout 1200 \
  --workers "${WORKERS}" \
  --output "${OUTPUT}" \
  "$@"
