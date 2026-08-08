"""Запуск ReAct-агента с инструментами против OpenAI-совместимой модели.

Модель явно знает свой инструментарий из систем-промпта и может проверять
утверждения на примерах через run_python (тестирование как с кодом).

Примеры (модель на карте проброшена на localhost:8137):

    # solver: решает задачу, проверяя свои утверждения кодом
    python scripts/run_react_agent.py --base-url http://127.0.0.1:8137/v1 \\
        --model qwen35-9b --api-key EMPTY \\
        "Найди наименьшее n>1, при котором n^2+n+41 составное."

    # tester: проверяет конкретное утверждение на примерах
    python scripts/run_react_agent.py --role tester --model qwen35-9b \\
        --base-url http://127.0.0.1:8137/v1 \\
        "Для всех целых n>=0 число n^2+n+41 простое."
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathagent.agent.react import (
    build_react_loop,
    build_system_prompt,
    make_test_claim_tool,
)
from mathagent.agent.tools import TOOLS
from mathagent.agent.vllm_chat import ChatVLLM

DEFAULT_PROMPT = ROOT / "conf/base/prompts/react-tools.yml"


def parse_args() -> argparse.Namespace:
    """Параметры модели, роли и вопроса."""
    parser = argparse.ArgumentParser(description="ReAct-агент с инструментами")
    parser.add_argument("question", help="Задача (solver) или утверждение (tester)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--role", default="solver", choices=("solver", "tester"))
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=6, help="лимит вызовов инструментов")
    parser.add_argument(
        "--no-tester",
        action="store_true",
        help="у солвера убрать инструмент test_claim (оставить только run_python)",
    )
    parser.add_argument(
        "--tester-steps", type=int, default=4, help="лимит шагов вложенного тестировщика"
    )
    return parser.parse_args()


def load_persona(prompt_path: Path, role_name: str) -> str:
    """Достать `system` выбранной роли из YAML-промпта."""
    config = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    try:
        return config["roles"][role_name]["system"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Нет роли '{role_name}' в {prompt_path}") from exc


def main() -> int:
    args = parse_args()
    persona = load_persona(args.prompt, args.role)

    model = ChatVLLM(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        extra_body={"max_tokens": args.max_tokens},
    )

    # Набор инструментов зависит от роли: солверу добавляем test_claim (за ним —
    # вложенный тестировщик), у самого тестировщика только run_python.
    tools = dict(TOOLS)
    if args.role == "solver" and not args.no_tester:
        tester_persona = load_persona(args.prompt, "tester")
        test_claim = make_test_claim_tool(model, tester_persona, max_steps=args.tester_steps)
        tools[test_claim.name] = test_claim

    system_prompt = build_system_prompt(persona, tools)
    app = build_react_loop(model, tools, max_steps=args.max_steps)

    result = app.invoke(
        {
            "messages": [("system", system_prompt), ("human", args.question)],
            "steps": 0,
        }
    )

    # Печатаем всю трассу: рассуждения, вызовы инструментов, Observation, финал
    print("=" * 70)
    for message in result["messages"]:
        role = getattr(message, "type", "?")
        content = getattr(message, "content", message)
        print(f"\n[{role}]\n{content}")
    print("=" * 70)
    print(f"\nШагов с инструментами: {result.get('steps', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
