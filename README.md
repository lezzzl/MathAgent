# MathAgent

MathAgent for summer MLA — агент для решения математических задач.

Модель вызывается через **OpenAI-совместимый API** (`ChatOpenAI` + LangGraph),
что позволяет одинаково работать и с локальной ollama, и с Yandex AI Studio.
В репозитории сейчас две части:

- **Агент (Kedro-пайплайн)** — конфиг-driven запуск и эксперименты (`conf/`, `uv`).
- **Бенчмарки** — раннеры AIME24/25 и MATH500, результаты в `results/*.jsonl`.

> ⚠️ Пока сосуществуют две системы зависимостей: `pyproject.toml` (uv, для агента)
> и `requirements.txt` (venv, для бенчмарков). Их предстоит свести в одну — см. TODO внизу.

## Требования

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (для Kedro-пайплайна)
- Доступ к OpenAI-совместимому API: ollama локально **или** Yandex AI Studio (ключ + `folder_id`)

## Установка

Для агента (Kedro + uv):

```bash
./scripts/setup.sh          # ставит uv, uv sync, создаёт .env
# или просто:
uv sync
```

Для бенчмарк-скриптов (venv + pip):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Доступ к модели

### Yandex AI Studio (OpenAI-совместимый эндпоинт)

Секреты **не хранятся в репозитории** — конфиг ссылается только на имена
переменных окружения (`agent.api_key_env`, `agent.folder_id_env` в
`parameters.yml`, по умолчанию `YC_API_KEY` и `YC_FOLDER_ID`):

```bash
echo 'YC_API_KEY=...'    >> .env
echo 'YC_FOLDER_ID=...'  >> .env
```

Модель задаётся как `agent.model` (`yandexgpt/latest`, `yandexgpt-lite/latest`)
и разворачивается в `gpt://<folder_id>/<model>`. Base URL —
`https://llm.api.cloud.yandex.net/v1`.

### ollama (локально)

```bash
ollama pull qwen3.5:4b
ollama serve
```

### vLLM на GPU-карте через SSH-тоннель

Модель разворачивается на карте кластера (карты прописаны в `~/.ssh/config`,
напр. `mla-yac-a100-3`), а её порт пробрасывается на твой localhost по SSH.
Затем Kedro/раннер обращаются к `http://127.0.0.1:<порт>/v1` как к обычному
OpenAI-совместимому эндпоинту.

**1. Поднять модель на карте** (в сессии на сервере). Выбери свободную GPU по
`nvidia-smi` и подставь её номер в `--gpus`:

```bash
ssh mla-yac-a100-3
docker run -d --name qwen35-9b --gpus '"device=3"' --ipc=host \
  -p 8137:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-9B \
  --served-model-name qwen35-9b \
  --max-model-len 73728 \
  --reasoning-parser qwen3
# логи: docker logs -f qwen35-9b   (ждём "Application startup complete")
```

> `--served-model-name` задаёт имя, которое надо будет передать в
> `benchmarks.model.name`. Если native-контекст модели меньше нужного —
> добавь YaRN: `--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'`.

**2. Построить тоннель** (в отдельном терминале на ноуте, оставить висеть):

```bash
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 8137:localhost:8137 mla-yac-a100-3
```

Порт слева (`8137:`) — локальный, справа (`:8137`) — на сервере (тот, что в
`-p 8137:8000` у docker). Их удобно держать одинаковыми.

**3. Проверить, что эндпоинт виден и узнать served-name:**

```bash
curl -s http://127.0.0.1:8137/v1/models        # в ответе "id": "qwen35-9b"
```

Если `curl` пустой — тоннель не поднят или модель ещё грузится. Дальше
`benchmarks.model.base_url = http://127.0.0.1:8137/v1` и
`benchmarks.model.name = qwen35-9b` (см. раздел «Бенчмарки»).

**4. Убить модель**, когда закончил:

```bash
ssh mla-yac-a100-3 docker rm -f qwen35-9b     # карта освобождается
```

## Запуск

### Агент (Kedro-эксперименты)

Эксперименты — это Kedro-окружения в `conf/experiments/<name>`, каждое
переопределяет только нужные параметры поверх `conf/base`:

```bash
uv run kedro run --env experiments/baseline               # без RAG
uv run kedro run --env experiments/rag_similar_conditions # RAG до решения
uv run kedro run --env experiments/rag_agent_triggered    # RAG как инструмент агента
```

Разовые оверрайды — через `--params rag.top_k=8`.

### Бенчмарки (Kedro-оркестратор)

Все гиперпараметры прогона — модель, sampling, runtime, список бенчей — лежат в
`conf/base/parameters.yml` в секции `benchmarks`. Kedro читает их и гоняет
выбранные бенчмарки под одним `run_id` в `results/runs/<run_id>/` (JSONL +
общий `manifest.json`). Сам раннер (`scripts/benchmarks/*`) переиспользуется.

> **Против модели на карте:** сначала подними vLLM и тоннель (см. «vLLM на
> GPU-карте через SSH-тоннель»), затем передай порт тоннеля и `served-name`.
> `base_url` и `model.name` **обязаны совпасть** с тем, что показал
> `curl http://127.0.0.1:8137/v1/models`, иначе каждая задача даст 404.

```bash
# смоук на 2 задачах через тоннель (порт 8137, served-name qwen35-9b):
kedro run --pipeline benchmarks \
  --params 'benchmarks.model.name=qwen35-9b,benchmarks.model.base_url=http://127.0.0.1:8137/v1,benchmarks.limit=2,benchmarks.select=[aime24]'

# полный прогон всех бенчей через тоннель:
kedro run --pipeline benchmarks \
  --params benchmarks.model.name=qwen35-9b,benchmarks.model.base_url=http://127.0.0.1:8137/v1
# то же короче, через env:
RUN_MODEL=qwen35-9b OPENAI_BASE_URL=http://127.0.0.1:8137/v1 kedro run --pipeline benchmarks

# прогнать всё с дефолтами из конфига (только если порт=8000 и served-name совпадает):
kedro run --pipeline benchmarks

# сменить сэмплинг разово, не трогая конфиг:
kedro run --pipeline benchmarks \
  --params benchmarks.model.name=qwen35-9b,benchmarks.generation.temperature=0.6

# только пара бенчей + быстрый прогон на 2 задачах:
kedro run --pipeline benchmarks \
  --params 'benchmarks.select=[aime24,math500],benchmarks.limit=2'

# продолжить прерванный прогон (тот же run_id):
kedro run --pipeline benchmarks --params benchmarks.run_id=<run_id>,benchmarks.resume=true
```

Переопределять параметры можно тремя способами (по возрастанию приоритета):
`conf/base/parameters.yml` → env-переменные (`RUN_MODEL`, `OPENAI_BASE_URL`) →
флаг `--params`. **Ключ API в конфиге не хранится** — только имя env-переменной
(`benchmarks.model.api_key_env`, по умолчанию `OPENAI_API_KEY`).

Доступные бенчи для `benchmarks.select`: `aime24 aime25 aime26 hmmt25 hmmt26
imo_answerbench math500`. Сводка прогона пишется в
`data/output/benchmark_run_summary.json`.

#### Наблюдение за прогоном

**Живьём.** Прогресс по каждой задаче (submit → done с latency, output-токенами,
finish_reason; ошибки; circuit breaker) сыпется в терминал во время `kedro run`
и копится в `logs/<run_id>/runner.log`. Отдельным окном:

```bash
tail -f logs/<run_id>/runner.log
```

**Пост-фактум в LangSmith.** Готовый прогон можно залить в LangSmith
(решения, reasoning и оценка каждой задачи в UI) — это инструмент из
`src/langsmith`. Чтобы Kedro публиковал автоматически после прогона, включи флаг
и задай ключ через env-переменную:

```bash
export LANGSMITH_API_KEY=...
kedro run --pipeline benchmarks \
  --params benchmarks.model.name=qwen35-9b,benchmarks.model.base_url=http://127.0.0.1:8137/v1,benchmarks.observability.publish_langsmith=true
```

Публикация не роняет прогон: если ключа нет или заливка упала — бенчмарки уже
сохранены на диск, узел просто вернёт `published: false`. Вручную тот же прогон
публикуется через `python src/langsmith/compare_runs.py publish --run-id <run_id>`
(см. раздел «LangSmith benchmark comparison»).

<details>
<summary>Запуск отдельных бенчмарков напрямую (без Kedro)</summary>

```bash
python scripts/benchmarks/run_aime24.py --model <served-name> --base-url http://127.0.0.1:8000/v1
python scripts/run_all_benchmarks.py --limit 2
```
</details>

## Results dashboard

The local Streamlit dashboard discovers run manifests under `results/runs` and
compares correctness, latency, and token usage. Generate compact correctness
sidecars first, then launch the app:

```bash
uv run python -m dashboard.evaluate
uv run streamlit run src/dashboard/app.py
```

Evaluation runs in an isolated worker with a 15-second hard timeout per task.
Progress is checkpointed every 10 tasks, so rerunning the same command resumes
an interrupted benchmark. Use `--task-timeout <seconds>` to change the limit.
The app uses PyArrow's system allocator and serializes dataframe conversion so
multiple browser tabs can safely share one Streamlit server on macOS.

Use `--results-dir` and `--evaluations-dir` with the evaluator, or
`MATHAGENT_RESULTS_DIR` and `MATHAGENT_EVALUATIONS_DIR` for both commands, to
read artifacts from different locations. Dashboard code is isolated under
`src/dashboard` and does not import the agent or benchmark runner.

## LangSmith benchmark comparison

The independent tooling under `src/langsmith` publishes existing benchmark
artifacts without rerunning the model and never imports `src/dashboard`. Set
`LANGSMITH_API_KEY` and, for a non-default installation, `LANGSMITH_ENDPOINT`.

```bash
uv run python src/langsmith/compare_runs.py publish \
  --run-id baseline-qwen35-9b-all-v1

uv run python src/langsmith/compare_runs.py compare \
  --left mathagent::baseline-qwen35-4b-all-v1 \
  --right mathagent::baseline-qwen35-9b-all-v1
```

Publishing automatically compares the new run (Run-2) against every compatible
existing source experiment (Run-1), so a positive difference means the new run
improved. Comparisons continue independently after failures, all successful
report URLs are printed, and the command exits nonzero if any comparison fails.
Deterministic source and report names make retries idempotent.

Each pair produces two complementary LangSmith artifacts: a native comparative
experiment with per-task `pairwise_accuracy` preferences, and the materialized
bootstrap/Holm report used by the five-column custom renderer. Resolved ties
receive `0.5` for both runs; tasks with an unresolved grade receive no pairwise
feedback. The native comparison URL is printed by the LangSmith SDK and stored
in the report experiment metadata.

Both commands accept `--resamples` (default `10000`), `--seed` (default `42`),
`--source-dataset`, and `--report-dataset`; `publish` also accepts
`--results-dir`. Reports use a paired centered-null bootstrap test with Holm
correction. Only shared tasks with resolved grades enter the accuracies and
test, while missing and unresolved counts remain in the report payload.

To show the five-column table inside LangSmith, host
`src/langsmith/renderer/index.html` over HTTPS and configure that URL as the
custom output renderer for the `mathagent-comparison-reports-v1` dataset. For a
self-hosted UI, append its allowed origin, for example
`?origins=https://smith.internal.example`. The renderer performs no network
requests and accepts report messages only from configured origins.

## Структура

```
conf/base/          общие конфиги: catalog, parameters, globals, sources, prompts/
conf/experiments/   по папке на эксперимент — оверрайды parameters.yml
conf/local/         локальные секреты (в .gitignore); ключ — в .env
data/               слои данных (raw / processed / output)
results/            результаты бенчмарков (jsonl)
src/mathagent/
  agent/            граф LangGraph (graph.py) + резолвинг ключей (keys.py)
  pipelines/        Kedro-пайплайны (agent_eval)
scripts/
  benchmarks/       раннеры AIME24/25, MATH500
  run_all_benchmarks.py
  setup.sh          установка окружения (uv)
```

## TODO

- Свести зависимости в одну систему (`pyproject.toml`/uv или `requirements.txt`).
- Объединить наборы промптов (`conf/base/prompts/solver.yml` и `solver-v0.yml`).
- Подключить бенчмарк-раннеры к Kedro-пайплайну (или наоборот).
