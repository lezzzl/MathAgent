"""Режим self-consistency: N независимых решений + мажоритарное голосование.

Мотивация. Пошаговый пайплайн с оценщиком и верификатором на 7B тратит бюджет
на самооценку, которая почти не несёт сигнала (на прогоне aime26 верификатор
подтвердил 16 решений, из них 13 неверных). Self-consistency тратит тот же
бюджет на разнообразие сэмплов, и на моделях этого размера это исторически
самый надёжный способ поднять точность на олимпиадных задачах.

Модуль намеренно не зависит от langgraph_math_solver — это отдельный режим,
который сравнивается с пошаговым на тех же задачах.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from answer_utils import extract_answer, majority_vote, normalize_answer
from tool_generator_subgraph import (
    count_chain_tokens,
    generator_with_tools,
    log as _log,
    make_llm,
)
from tools import reset_calculator_state

DEFAULT_SYSTEM_PROMPT = """You are an expert competition mathematician solving an AIME-style problem.

You have a `python_exec` tool that runs multi-line Python with sympy preloaded
(solve, Eq, Rational, symbols, factorint, isprime, divisors, binomial, simplify,
Matrix, primerange), plus numpy as np and itertools.

YOU MUST CALL `python_exec` AT LEAST ONCE BEFORE GIVING YOUR FINAL ANSWER.
Doing arithmetic in your head is the single most common way to fail this task.
Use the tool to:
  - carry out every multi-digit computation, factorisation, or modular reduction;
  - enumerate the cases of a counting problem instead of reasoning about them;
  - solve equations symbolically with sympy rather than by hand;
  - re-check the final number before you box it.

Prefer exact types: Rational(1,3) and sqrt(2), never 0.333 or 1.414. print() what
you need; the last expression is printed automatically. Execution is capped at
10 seconds, so narrow the search space instead of brute-forcing millions of cases.

Reason step by step, but keep the prose short — put the work in the tool.

End your response with the final answer on its own line in exactly this form:
Final answer: \\boxed{ANSWER}

The answer must be a single integer between 0 and 999 inclusive."""

DEFAULT_USER_TEMPLATE = "Problem:\n{problem}"

@dataclass
class SelfConsistencyConfig:
    n_samples: int = 8
    temperature: float = 0.8
    max_tokens: int = 2048
    max_hops: int = 6
    max_total_tool_calls: int = 10
    sample_workers: int = 4
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE
    model_name: str = ""
    base_url: str = ""
    api_key: str = ""
    # Температура первого сэмпла: один прогон около-жадный обычно полезен как
    # якорь, остальные сэмплируются горячее ради разнообразия.
    first_sample_temperature: float = 0.3


@dataclass
class SampleResult:
    index: int
    answer: Optional[str]
    normalized: Optional[str]
    text: str
    tokens: int
    tool_calls: int
    error: Optional[str] = None
    truncated: bool = False


@dataclass
class SolveResult:
    final_answer: Optional[str]
    votes: List[Any] = field(default_factory=list)
    samples: List[SampleResult] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def agreement(self) -> float:
        """Доля сэмплов, проголосовавших за победивший ответ.

        Полезная прокси-метрика уверенности: на практике задачи с agreement
        ниже ~0.4 почти всегда решены неверно, и по ней можно отбирать, куда
        имеет смысл доливать компьют.
        """
        valid = [s for s in self.samples if s.normalized]
        if not valid or not self.votes:
            return 0.0
        return self.votes[0][1] / len(valid)


def _run_one_sample(index: int, problem: str, config: SelfConsistencyConfig) -> SampleResult:
    temperature = config.first_sample_temperature if index == 0 else config.temperature
    try:
        llm = make_llm(
            temperature,
            model_name=config.model_name,
            base_url=config.base_url,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
        )
        result = generator_with_tools.invoke(
            {
                "messages": [
                    SystemMessage(content=config.system_prompt),
                    HumanMessage(content=config.user_template.format(problem=problem)),
                ],
                "max_hops": config.max_hops,
                "max_total_tool_calls": config.max_total_tool_calls,
            },
            config={"configurable": {"llm": llm}},
        )
    except Exception as exc:  # noqa: BLE001 — один сэмпл не должен ронять задачу
        _log(f"    - sample {index + 1}: ОШИБКА {type(exc).__name__}: {exc}")
        return SampleResult(index, None, None, "", 0, 0, error=f"{type(exc).__name__}: {exc}")

    messages = result["messages"]
    last = messages[-1]
    text = last.content or ""
    answer = extract_answer(text)
    normalized = normalize_answer(answer)
    tokens = count_chain_tokens(messages)
    tool_calls = sum(1 for m in messages if isinstance(m, ToolMessage))

    # Сэмпл, упёршийся в max_tokens, обычно не успевает написать \boxed и
    # молча теряет голос. Считаем такие отдельно: если их много, лимит на
    # ответ занижен, и голосование идёт по половине сэмплов.
    finish_reason = (getattr(last, "response_metadata", None) or {}).get("finish_reason")
    truncated = finish_reason == "length" or (normalized is None and bool(text.strip()))

    note = " [ОБРЕЗАН по лимиту токенов]" if truncated and normalized is None else ""
    _log(
        f"    - sample {index + 1}: answer={normalized!r} "
        f"(temp {temperature:.2f}, {tokens} токенов, {tool_calls} tool-вызовов){note}"
    )
    return SampleResult(index, answer, normalized, text, tokens, tool_calls, truncated=truncated)


def solve_with_self_consistency(problem: str, config: SelfConsistencyConfig) -> SolveResult:
    """Сэмплит N решений параллельно и возвращает победителя голосования."""
    reset_calculator_state()
    _log(f"  [self-consistency] {config.n_samples} сэмплов, до {config.sample_workers} параллельно")

    workers = max(1, min(config.sample_workers, config.n_samples))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = list(
            pool.map(lambda i: _run_one_sample(i, problem, config), range(config.n_samples))
        )

    winner, votes = majority_vote(s.normalized for s in samples)
    tokens_used = sum(s.tokens for s in samples)
    failed = [s for s in samples if s.error]

    result = SolveResult(
        final_answer=winner,
        votes=votes,
        samples=samples,
        tokens_used=tokens_used,
        error=f"{len(failed)}/{len(samples)} сэмплов упали" if failed else None,
    )
    vote_summary = ", ".join(f"{ans}×{cnt}" for ans, cnt in votes[:5]) or "нет валидных ответов"
    lost = sum(1 for s in samples if s.normalized is None and not s.error)
    lost_note = f" | потеряно голосов: {lost}/{len(samples)}" if lost else ""
    _log(
        f"  [self-consistency] голоса: {vote_summary} | победитель: {winner!r} "
        f"| согласие: {result.agreement:.0%} | токенов: {tokens_used}{lost_note}"
    )
    return result


def build_metrics(result: SolveResult) -> Dict[str, Any]:
    """Метрики в metadata записи бенчмарка."""
    return {
        "mode": "self_consistency",
        "tokens_used": result.tokens_used,
        "n_samples": len(result.samples),
        "n_valid_samples": sum(1 for s in result.samples if s.normalized),
        "n_failed_samples": sum(1 for s in result.samples if s.error),
        "n_truncated_samples": sum(1 for s in result.samples if s.truncated),
        "votes": [{"answer": a, "count": c} for a, c in result.votes],
        "agreement": round(result.agreement, 3),
        "mean_tool_calls": (
            round(sum(s.tool_calls for s in result.samples) / len(result.samples), 2)
            if result.samples else 0.0
        ),
        "error": result.error,
    }
