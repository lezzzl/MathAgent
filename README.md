# MathAgent

Скрипты для прогона математических бенчмарков через OpenAI-совместимый API пока на ollama.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama

```bash
ollama pull qwen3.5:4b
ollama serve
```

## Запуск

```bash
python scripts/benchmarks/run_aime24.py
python scripts/benchmarks/run_aime25.py
python scripts/benchmarks/run_math500.py
python scripts/run_all_benchmarks.py
```

Для тестового прогона можно ограничить количество задач:

```bash
python scripts/benchmarks/run_math500.py --limit 2
python scripts/run_all_benchmarks.py --limit 2
```

Результаты сохраняются в `results/<benchmark>/<timestamp>.jsonl`.
