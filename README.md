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
