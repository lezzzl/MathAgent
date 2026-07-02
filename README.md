# MathAgent

MathAgent for summer MLA — агентский луп для решения математических задач.

Агент построен на **LangGraph** (сам цикл «рассуждение → вызов инструмента →
проверка результата»), а **Kedro** отвечает за конфигурацию, данные и
воспроизводимый eval-пайплайн.

## Требования

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- API-ключ Anthropic

## Установка

```bash
./scripts/setup.sh
```

Скрипт:
1. ставит `uv`, если его нет;
2. выполняет `uv sync` — создаёт `.venv` и ставит зависимости из `pyproject.toml`
   (версии фиксируются в `uv.lock`);
3. создаёт шаблон `conf/local/credentials.yml`, если его ещё нет.

Если `uv` уже установлен, установку можно свести к одной команде:

```bash
uv sync
```

## Настройка

После установки впиши ключ в `conf/local/credentials.yml`:

```yaml
anthropic:
  api_key: sk-ant-...
```

## Запуск

```bash
uv run kedro run --pipeline agent_eval
```

`uv run` автоматически синхронизирует окружение перед запуском, так что
активировать `.venv` вручную не нужно.

## Структура

```
conf/base/          конфиги: catalog, parameters, globals, sources, промпты
conf/local/         секреты (credentials.yml, не коммитится)
data/               слои данных (raw / processed / output)
src/mathagent/
  agent/            агент на LangGraph (переносимый, не зависит от Kedro)
  pipelines/        Kedro-пайплайны (agent_eval)
scripts/setup.sh    установка окружения
```
