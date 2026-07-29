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

### Полный benchmark workflow

Kedro также оркестрирует agentic benchmark loop, проверку ответов и
инкрементальное обновление dashboard comparison table:

```bash
export MODEL=Qwen/Qwen3.5-9B
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=token-abc123

# 1) agent -> 2) evaluate -> 3) compare
uv run python scripts/run_experiment.py --config experiment.yml

# Только 2) evaluate -> 3) compare для уже готового запуска
uv run python scripts/evaluate_experiment.py \
  --run-id my-agent-run

# Только evaluate для запуска из другого каталога результатов
uv run python scripts/evaluate_experiment.py \
  --run-id my-agent-run \
  --runs-dir /mnt/storage-1/MathAgent/results/runs \
  --score-only
```

`run_experiment.py` всегда выполняет все три этапа. `--config` указывает на YAML
с параметрами конкретного эксперимента; корень файла — сама секция experiment,
поэтому повторять ключ `experiment` не нужно:

```yaml
run_id: qwen4b-v1
agent:
  args:
    - --pipeline
    - qwen4b
    - --workers
    - "4"
```

Этот YAML частично и рекурсивно объединяется с общими настройками
`experiment` из `conf/base/parameters.yml`. Поле `run_id` обязательно и
определяет каталог `results/runs/<run_id>`.

`evaluate_experiment.py` требует `--run-id`, потому что должен выбрать уже
существующие результаты.

Настройки benchmark-скриптов и evaluators находятся в `experiment` внутри
`conf/base/parameters.yml`. Для частичных Kedro environment overrides включён
soft merge: файл эксперимента может указывать только изменяемые поля, не
повторяя всю секцию `experiment`.

Сырые JSONL сохраняются рядом с manifest. Этап evaluation пишет отдельные
`*_verified.jsonl`, добавляет в них `is_correct` и переключает manifest на
проверенный файл; исходные ответы не перезаписываются.

### Бенчмарки

```bash
python scripts/benchmarks/run_aime24.py
python scripts/benchmarks/run_aime25.py
python scripts/benchmarks/run_math500.py
python scripts/run_all_benchmarks.py

# тестовый прогон с ограничением числа задач:
python scripts/run_all_benchmarks.py --limit 2
```

Результаты сохраняются в `results/<benchmark>/<timestamp>.jsonl`.

## Results dashboard

The local Streamlit dashboard discovers run manifests under `results/runs` and
compares pre-scored benchmark runs. Each task record must contain a numeric
`score` from 0 to 1 or the Boolean `is_correct` field written by the standalone
verification scripts. The comparison step never grades solutions; the
dashboard's **Score runs** action invokes the configured verification scripts
before comparison.

```bash
uv run python -m dashboard.compare
uv run streamlit run src/dashboard/app.py
```

The comparison command writes `results/comparisons.parquet`. On later runs it
checks every discovered pre-scored run pair and adds any missing shared-benchmark
rows. Existing pair statistics are not recomputed; Holm-adjusted p-values are
refreshed when the multiple-testing family grows.

Use **Score runs** to run the configured evaluators for every discovered run
that still has unscored records. Then use **Update comparison table** to create
the table when needed and complete all missing pairwise comparisons. The runs
field defaults to `/mnt/storage-1/MathAgent/results/runs` and still honors
`MATHAGENT_RESULTS_DIR`. Streamlit always stores comparisons at
`/mnt/storage-1/MathAgent/results/comparison/table.parquet`; that path is no
longer editable in the dashboard. The two dashboard views compare one run with
another or several runs against a selected baseline. The app polls the
comparison table, so updates made by another process appear automatically.

The app uses PyArrow's system allocator and serializes dataframe conversion so
multiple browser tabs can safely share one Streamlit server on macOS.

Set `MATHAGENT_RESULTS_DIR` to read dashboard runs from another directory.
`MATHAGENT_COMPARISONS_PATH` still controls the standalone comparison command,
but Streamlit always uses its fixed comparison path. Dashboard code is isolated
under `src/dashboard` and does not import the agent or benchmark runner.

Explicit paths can also be passed to the comparison command:

```bash
uv run python -m dashboard.compare \
  --runs-dir /path/to/runs \
  --output /path/to/comparisons.parquet
```

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
