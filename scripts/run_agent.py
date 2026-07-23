"""Запуск базового агентского лупа против OpenAI-совместимой модели.

Модель может быть развёрнута где угодно, лишь бы был OpenAI-совместимый эндпоинт:
Yandex AI Studio, локальная ollama или vLLM на GPU-карте (см. README «Запуск на GPU»).

Пример (порт модели с карты проброшен на localhost:8000):
    python scripts/run_agent.py \\
        --base-url http://127.0.0.1:8000/v1 \\
        --model Qwen/Qwen2.5-Math-7B-Instruct \\
        --api-key EMPTY \\
        "Найди все корни уравнения x^2 - 5x + 6 = 0"
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_openai import ChatOpenAI

from mathagent.agent.loop import build_agent_loop

DEFAULT_SYSTEM = (
    "Ты — математический помощник. Решай задачу пошагово "
    "и в конце выведи финальный ответ."
)


def parse_args() -> argparse.Namespace:
    """Параметры модели и вопрос из командной строки."""
    parser = argparse.ArgumentParser(description="Базовый агентский луп (без тулов)")
    parser.add_argument("question", help="Условие задачи / вопрос модели")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-совместимый эндпоинт модели",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--api-key", default="EMPTY", help="для vLLM/ollama подойдёт любой")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    app = build_agent_loop(model)

    result = app.invoke(
        {"messages": [("system", args.system), ("human", args.question)]}
    )
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
