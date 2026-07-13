# MathAgent

MathAgent for summer MLA — агентский луп для решения математических задач.

Агент построен на **LangGraph** (сам цикл «рассуждение → вызов инструмента →
проверка результата»), а **Kedro** отвечает за конфигурацию, данные и
воспроизводимый eval-пайплайн.

## Требования

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Доступ к Yandex AI Studio: API-ключ и `folder_id` каталога

## Установка

```bash
./scripts/setup.sh
```

Скрипт:
1. ставит `uv`, если его нет;
2. выполняет `uv sync` — создаёт `.venv` и ставит зависимости из `pyproject.toml`
   (версии фиксируются в `uv.lock`);
3. создаёт `.env` для ключа, если его ещё нет.

Если `uv` уже установлен, установку можно свести к одной команде:

```bash
uv sync
```

## Доступ к Yandex AI Studio

Модель вызывается через OpenAI-совместимый эндпоинт Yandex AI Studio
(`https://llm.api.cloud.yandex.net/v1`). Секреты **не хранятся в репозитории** —
конфиг ссылается только на имена переменных окружения (`agent.api_key_env` и
`agent.folder_id_env` в `parameters.yml`, по умолчанию `YC_API_KEY` и
`YC_FOLDER_ID`). Задай их одним из способов:

```bash
# вариант 1: .env (создаётся скриптом, лежит в .gitignore)
echo 'YC_API_KEY=...'    >> .env
echo 'YC_FOLDER_ID=...'  >> .env

# вариант 2: переменные окружения
export YC_API_KEY=...
export YC_FOLDER_ID=...
```

Либо укажи путь к файлу с ключом через `agent.api_key_file` в `parameters.yml`.
Модель задаётся как `agent.model` (например `yandexgpt/latest`, `yandexgpt-lite/latest`)
и разворачивается в `gpt://<folder_id>/<model>`.

## Запуск

Эксперименты — это Kedro-окружения в `conf/experiments/<name>`. Каждое
переопределяет только нужные параметры поверх `conf/base`:

```bash
uv run kedro run --env experiments/baseline               # без RAG
uv run kedro run --env experiments/rag_similar_conditions # RAG до решения
uv run kedro run --env experiments/rag_agent_triggered    # RAG как инструмент агента
```

Разовые оверрайды без нового конфига — через `--params`:

```bash
uv run kedro run --env experiments/rag_similar_conditions --params rag.top_k=8
```

`uv run` автоматически синхронизирует окружение перед запуском, так что
активировать `.venv` вручную не нужно.

## Структура

```
conf/base/          общие конфиги: catalog, parameters, globals, sources, prompts/
conf/experiments/   по папке на эксперимент — оверрайды parameters.yml
conf/local/         локальные секреты (в .gitignore); ключ — в .env
data/               слои данных (raw / processed / output)
src/mathagent/
  agent/            агент на LangGraph (переносимый, не зависит от Kedro)
  pipelines/        Kedro-пайплайны (agent_eval)
scripts/setup.sh    установка окружения
```
