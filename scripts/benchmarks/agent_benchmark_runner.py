import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import langgraph_math_solver
import langgraph_math_solver_qwen4b
from tools import reset_calculator_state, shutdown_workers
from trajectory import RECORDER

from dotenv import load_dotenv
load_dotenv()

# v2 = v1 с обеззараженными примерами. v1 содержит DATA LEAKAGE (постановка
# AIME26 task 1 у генератора, палиндромы AIME26 task 2 и «180+24=204» из
# AIME24 task 60 у оценщика) и оставлен только для воспроизведения старых
# прогонов: --prompt conf/base/prompts/agent-step-v1.yml
DEFAULT_PROMPT = ROOT / "conf/base/prompts/agent-step-v2.yml"

# Пошаговые пайплайны, переключаемые флагом --pipeline.
#   default — оригинал под Qwen 7B/9B (langgraph_math_solver);
#   qwen4b  — тот же граф + стадия сегментации одного шага, без тулов.
PIPELINES = {
    "default": langgraph_math_solver,
    "qwen4b": langgraph_math_solver_qwen4b,
}
# Дефолтный yaml промптов на каждый пайплайн (если --prompt не задан явно).
DEFAULT_PROMPTS = {
    "default": DEFAULT_PROMPT,
    # Линейка промптов пошагового пайплайна (каждый следующий = предыдущий плюс
    # одно изменение, чтобы А/Б был чистым):
    #   v1 — базовый;
    #   v2 = v1 + секция TOOL у оценщика;
    #   v3 = v2 + проверка «ответ отвечает на заданный вопрос» (generator+evaluator);
    #   v4 = v3 с обеззараженными примерами генератора и оценщика;
    #   v5 = v4 + обеззараженный пример сегментатора (в v4 его пропустили: там
    #        оставалась постановка AIME26 task 1) и исправленный Example 3
    #        оценщика (боксил 137 при выводе 209 — пример показывал оценку 1.0
    #        за ответ, не следующий из вывода);
    #   v6 = v5 + чинёный формат вывода у верификатора, оценщика и сегментатора:
    #        описания «(here: ...)» стояли на месте значения, и модель их
    #        пропускала или копировала — вердикт верификатора не разбирался в
    #        40–72% случаев, оценщика в 17–26%.
    #
    # Дефолт — v6. Версии v1–v3 содержат DATA LEAKAGE: их примеры взяты из
    # реальных задач бенчмарков (постановка AIME26 task 1, палиндромы AIME26
    # task 2, «180+24=204» из AIME24 task 60, а v3 добавил cos(theta)=29/36 и
    # цель m+n из AIME26 task 5). Замер на них завышен, поэтому по умолчанию их
    # брать нельзя. v4 чист для aime24/25, но не для aime26 (см. выше).
    # Подключать старые версии только осознанно и только для воспроизведения
    # прошлых прогонов: --prompt conf/base/prompts/agent-step-qwen4b-v4.yml
    # Проверка: python scripts/check_prompt_leakage.py
    "qwen4b": ROOT / "conf/base/prompts/agent-step-qwen4b-v6.yml",
}

# Активный модуль пайплайна. Переустанавливается в run_benchmark по --pipeline;
# дефолт сохраняет прежнее поведение (оригинальный солвер), чтобы существующие
# команды запуска работали без изменений.
solver_mod = langgraph_math_solver


@dataclass(frozen=True)
class BenchmarkConfig:
    """Хранит параметры, которые различаются у бенчмарков."""
    name: str
    dataset_name: str
    split: str
    task_id_field: str
    output_directory: str
    metadata_fields: tuple[str, ...] = ()
    # Имена полей условия и эталона в датасете. Дефолты подходят AIME-датасетам
    # (problem/answer); IMO-AnswerBench, например, использует Problem/Short Answer.
    problem_field: str = "problem"
    ground_truth_field: str = "answer"
    # Необязательный предфильтр задач по строке эталона. Нужен датасетам со
    # смешанным форматом ответа (IMO-AnswerBench: часть символьная/множественная),
    # где числовая сверка math-verify применима лишь к подмножеству. Принимает
    # строку Short Answer, возвращает True — оставить задачу. compare=False,
    # чтобы Callable не участвовал в hash/eq frozen-датакласса.
    answer_filter: Optional[Callable[[str], bool]] = field(default=None, compare=False)
    answer_filter_name: str = ""
    # Чем сверять ответы сразу после прогона:
    #   "math" — verify_answers.py (math-verify), годится для AIME/HMMT/MATH500,
    #            где эталон числовой;
    #   "imo"  — verify_imo_answers.py (math-verify + LLM-судья на символьных
    #            ответах), нужен для IMO-AnswerBench и подобных смешанных наборов.
    verifier: str = "math"


def parse_benchmark_args(
    description: str,
    *,
    include_output: bool = True,
    extra_flags: Optional[list[tuple[list[str], dict[str, Any]]]] = None,
) -> argparse.Namespace:
    """Считывает общие параметры модели и запуска из командной строки.

    Имя модели берётся из --model или переменной MODEL; дефолта нет —
    бенчмарки гоняются на моделях с сервера, поэтому модель задаётся явно.

    extra_flags — доп. аргументы конкретного бенчмарка в виде
    [(["--flag"], {"action": ...}), ...]; их значения попадают в тот же
    Namespace. Нужно, чтобы бенчмарк-специфичные опции (например, --all-answers
    у IMO) не приходилось объявлять здесь, в общем парсере.
    """
    parser = argparse.ArgumentParser(description=description)
    for names, opts in (extra_flags or []):
        parser.add_argument(*names, **opts)
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL"),
        required=os.getenv("MODEL") is None,
        help="Имя модели на сервере. Обязателен, если не задана переменная MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override для температуры генератора (по умолчанию "
             "берётся temperature роли 'generator' из --prompt yaml). Для "
             "reasoning-моделей (Qwen3.5 и т.п.) не ставьте ниже ~0.5: низкая "
             "температура в режиме размышлений — известный триггер вырождения "
             "в бесконечный повтор без выхода из <think>.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "2048")),
        help="Лимит токенов на один ответ модели. Должен быть заметно меньше "
             "контекста сервера, иначе длинные диалоги упираются в 400.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--prompt", type=Path, default=None,
        help="YAML с промптами ролей. Если не задан — берётся дефолт под выбранный "
             "--pipeline (default: agent-step-v1.yml).",
    )
    parser.add_argument(
        "--pipeline", default="default", choices=tuple(PIPELINES),
        help="Какой пошаговый пайплайн использовать: "
             "'default' — оригинал под Qwen 7B/9B (по умолчанию, поведение не "
             "меняется); 'qwen4b' — тот же граф со стадией сегментации одного "
             "шага.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip", type=int, default=0, help="Пропустить N первых задач")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Сколько задач бенчмарка решать параллельно. Одновременных запросов "
             "к серверу будет workers * sample-workers — именно это число определяет "
             "пиковый размер KV-кэша. Дефолт рассчитан на выделенный сервер с A100; "
             "на слабой машине снижайте",
    )
    parser.add_argument("--no-tools", action="store_true", help="Отключить использование калькулятора")
    parser.add_argument(
        "--thinking", choices=["auto", "on", "off"], default="auto",
        help="Блок размышлений у thinking-моделей (Qwen3.x). 'auto' — как задано "
             "per-role в yaml (по умолчанию); 'on'/'off' — принудительно для всех "
             "ролей, для A/B-замеров без правки yaml. Учтите: с размышлениями "
             "оценщику нужно ~5000 токенов на кандидата вместо ~400.",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Дописать существующий JSONL, пропустив уже решённые task_id",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Не сверять ответы после прогона. По умолчанию сразу запускается "
             "verify_answers.py (для IMO — verify_imo_answers.py), отдельным "
             "процессом; JSONL прогона сохраняется в любом случае.",
    )
    parser.add_argument(
        "--verify-timeout", type=float, default=3600.0,
        help="Потолок времени на сверку, секунд (по умолчанию час).",
    )
    parser.add_argument(
        "--trajectory", nargs="?", const="auto", default=None, metavar="PATH",
        help="Записать полные траектории решения (промпты, сырые генерации, "
             "оценки, вердикты, токены, время) в JSON для просмотрщика. Без "
             "значения путь берётся рядом с --output. Затем: "
             "python scripts/make_viewer.py <файл.json>. Работает для --role solver.",
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Локальный файл датасета (jsonl/parquet/csv) вместо загрузки с HF Hub. "
             "Нужно при недоступной сети: скачайте данные вручную (scp) и укажите "
             "путь. Поля должны совпадать с ожидаемыми (problem/answer и т.п.).",
    )

    group = parser.add_argument_group("пошаговый солвер")
    group.add_argument("--k-branches", type=int, default=3)
    group.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Минимальная оценка шага для коммита (иначе recovery). Оценщик теперь "
             "градуированный (0/0.25/0.5/0.75/1.0): 0.5 = 'корректно и есть "
             "прогресс'. Прежние 0.8 при бинарной шкале означали 'ровно 1.0'.",
    )
    group.add_argument("--branch-mode", default="multi", choices=["single", "multi"])
    group.add_argument(
        "--token-budget", type=int, default=800000,
        help="Лимит токенов на задачу. Дефолт поднят с 250k: с включёнными "
             "инструментами один вызов генератора стоит до ~220k токенов (цикл "
             "тулов умножает num_predict=40000 на число витков), и на 600k "
             "по-прежнему умирали задачи, которые были в шаге от ответа. "
             "Снижайте, если нужен более дешёвый и быстрый прогон.",
    )
    group.add_argument("--max-stuck-steps", type=int, default=2)
    group.add_argument("--max-unreliable-evals", type=int, default=3)
    group.add_argument("--max-recoveries", type=int, default=5)
    group.add_argument(
        "--min-steps-before-answer", type=int, default=0,
        help="Сколько шагов должно быть принято, прежде чем \\boxed{} засчитывается "
             "как финальный ответ (только --pipeline qwen4b). 0 (по умолчанию) — "
             "ответ принимается сразу: если задача решается за один шаг, незачем "
             "тратить ещё один раунд генерации. 1 — запрещает ответ на глубине 0 и "
             "заставляет пайплайн сделать хотя бы один промежуточный шаг; нужно, "
             "если важно измерять именно пошаговость, а не итоговую точность.",
    )

    if include_output:
        parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_verification(config: BenchmarkConfig, output_path: Path,
                     args: argparse.Namespace) -> Optional[Path]:
    """Сверяет ответы сразу после прогона, отдельным процессом.

    Подпроцесс, а не импорт: оба скрипта сверки — самостоятельные CLI со своим
    разбором аргументов, а math_verify умеет зависать и падать на отдельных
    выражениях. Изоляция гарантирует, что уже записанный JSONL прогона (часы
    работы) не пострадает, что бы ни случилось со сверкой.

    Возвращает путь к файлу со сверенными ответами либо None.
    """
    script = "verify_imo_answers.py" if config.verifier == "imo" else "verify_answers.py"
    suffix = "_imoverified.jsonl" if config.verifier == "imo" else "_verified.jsonl"
    verified_path = output_path.with_name(output_path.stem + suffix)

    cmd = [sys.executable, str(ROOT / script), str(output_path), "-o", str(verified_path)]
    if config.verifier == "imo":
        # Судьёй берём ту же модель, против которой шёл прогон: её сервер точно
        # поднят, а дефолты скрипта смотрят на другой порт.
        cmd += ["--judge-base-url", args.base_url,
                "--judge-model", args.model,
                "--judge-api-key", args.api_key]
    elif sys.platform == "win32":
        # На Windows таймаут math_verify реализован через сигналы и не работает.
        cmd.append("--no-timeout")

    print(f"\n[verify] {script} -> {verified_path.name}")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), timeout=args.verify_timeout)
    except subprocess.TimeoutExpired:
        print(f"[verify] ОШИБКА: сверка не уложилась в {args.verify_timeout:.0f} с. "
              f"Прогон сохранён: {output_path}\n"
              f"          Запустите вручную: python {script} {output_path}")
        return None
    except Exception as exc:  # noqa: BLE001 — сверка не должна ронять прогон
        print(f"[verify] ОШИБКА: {type(exc).__name__}: {exc}\n"
              f"          Прогон сохранён: {output_path}\n"
              f"          Запустите вручную: python {script} {output_path}")
        return None

    if proc.returncode != 0 or not verified_path.exists():
        print(f"[verify] Сверка завершилась с кодом {proc.returncode}. "
              f"Прогон сохранён: {output_path}\n"
              f"          Запустите вручную: python {script} {output_path}")
        return None

    # Короткая сводка, чтобы не лезть в файл.
    try:
        total = correct = 0
        with verified_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                correct += bool(rec.get("is_correct"))
        if total:
            print(f"[verify] ИТОГ {config.name}: {correct}/{total} = {correct/total:.1%}")
    except Exception:  # noqa: BLE001 — сводка не критична
        pass
    return verified_path


def build_record(
    config: BenchmarkConfig,
    task_id: str,
    solution: str | None,
    ground_truth: str,
    model_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Формирует одну запись итогового JSONL-файла."""
    return {
        "benchmark_name": config.name,
        "task_id": task_id,
        "solution": solution,
        "ground_truth": ground_truth,
        "model_name": model_name,
        "metadata": metadata,
    }


def resolve_output_path(config: BenchmarkConfig, output: Path | None) -> Path:
    """Выбирает переданный путь или генерирует имя JSONL по времени запуска."""
    if output is not None:
        return output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "results" / config.output_directory / f"agent_{timestamp}.jsonl"


_LOCAL_LOADERS = {".jsonl": "json", ".json": "json", ".parquet": "parquet", ".csv": "csv"}


def load_dataset_offline_safe(config: BenchmarkConfig, data_file: "str | None" = None):
    """Грузит датасет, устойчиво к отсутствию сети к HuggingFace Hub.

    Порядок:
      1. --data-file: локальный jsonl/parquet/csv, скачанный вручную (например
         через scp). Hub не опрашивается вообще — самый надёжный путь при
         недоступной сети.
      2. обычный load_dataset по имени с Hub/кэша.
      3. фоллбек на сырые parquet из hub-кэша: `hf download` кладёт файлы в
         ~/.cache/huggingface/hub/, а load_dataset в offline-режиме ищет свой
         обработанный кэш в ~/.cache/huggingface/datasets/ и падает с
         OfflineModeIsEnabled, хотя данные уже на диске.
    """
    from datasets import load_dataset

    if data_file:
        path = os.path.expanduser(data_file)
        loader = _LOCAL_LOADERS.get(os.path.splitext(path)[1].lower())
        if loader is None:
            raise ValueError(
                f"--data-file: неподдерживаемое расширение {path!r}. "
                f"Ожидается один из: {', '.join(_LOCAL_LOADERS)}"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"--data-file: файл не найден: {path}")
        print(f"[dataset] локальный файл {path} (loader={loader}), Hub не опрашивается")
        return load_dataset(loader, data_files=[path], split="train")

    try:
        return load_dataset(config.dataset_name, split=config.split)
    except Exception as exc:
        cache_root = os.path.expanduser(
            os.getenv("HF_HUB_CACHE")
            or os.path.join(os.getenv("HF_HOME", "~/.cache/huggingface"), "hub")
        )
        slug = "datasets--" + config.dataset_name.replace("/", "--")
        pattern = os.path.join(cache_root, slug, "snapshots", "*", "**", "*.parquet")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise
        print(f"[dataset] {type(exc).__name__} при обычной загрузке — беру сырые "
              f"parquet из hub-кэша ({len(files)} файл(ов))")
        # data_files без разбивки по сплитам кладёт всё в 'train'; наши бенчмарки
        # используют одиночный сплит, поэтому этого достаточно.
        return load_dataset("parquet", data_files=files, split="train")


class ServerUnavailable(RuntimeError):
    """Сервер модели недоступен или отдаёт не то, чего мы ждём."""


# Замер на Qwen3.5-9B: оценка одного шага с размышлениями заняла 5277 токенов.
# Лимит ниже этого приводит к обрыву на середине размышлений и пустому ответу —
# роль молча перестаёт работать, что выглядит как "модель выдаёт мусор".
_THINKING_MIN_TOKENS = 6000

# Сколько токенов оставить под промпт, урезая num_predict под контекст сервера.
# Типичный промпт роли — условие задачи плюс принятые шаги, это заметно меньше;
# запас берём с двойным перекрытием, чтобы не ловить 400 на глубоких шагах.
_PROMPT_MARGIN = int(os.getenv("PROMPT_MARGIN", "4096"))


def apply_thinking_override(mode: str) -> None:
    """Принудительно включает или выключает размышления для всех ролей.

    Нужно для честного A/B: иначе каждый замер требует правки yaml, и легко
    сравнить два прогона с разными настройками, думая, что менялась только одна.
    """
    if mode == "auto":
        return
    enabled = mode == "on"
    # segmenter — чисто механическая экстракция шага; размышления ей не помогают,
    # а при малом num_predict (4000) обрывают ответ на середине think-блока, и
    # маркеры ###STEP### не доезжают. Именно так --thinking on убил прогон на
    # Qwen3.5-4B (41 ненадёжная сегментация). Поэтому флаг её не трогает —
    # segmenter всегда работает по своему yaml-значению (без размышлений).
    skip = {"segmenter"}
    for role_name, role in solver_mod.ROLES.items():
        if role_name in skip:
            continue
        solver_mod.ROLES[role_name] = replace(role, enable_thinking=enabled)
    kept = [n for n in skip if n in solver_mod.ROLES]
    note = f" (кроме {', '.join(kept)} — всегда без размышлений)" if kept else ""
    print(f"[config] --thinking={mode}: размышления {'включены' if enabled else 'выключены'} "
          f"для ролей{note}, per-role настройки yaml проигнорированы")


def warn_on_context_fit(context_length: int | None) -> None:
    """Подгоняет num_predict ролей под фактический контекст сервера.

    Не просто предупреждает, а УРЕЗАЕТ лимит, если он не влезает. Причина: в
    yaml лимиты подняты под большой контекст (генератору нужно 40000), и на
    сервере с --max-model-len 32768 каждый вызов возвращал бы HTTP 400 с пустым
    ответом — ровно тот сбой, который однажды выглядел как «модель не слушается
    инструкций». Урезание даёт деградацию (ответы будут обрываться, это видно
    в метрике обрывов), а не молчаливую поломку всего прогона.
    """
    if not isinstance(context_length, int):
        return
    # vLLM отклоняет запрос, когда prompt + max_tokens > max_model_len, то есть
    # проверка идёт НА КАЖДЫЙ запрос, а не по худшему случаю. Поэтому урезаем
    # только до значения, при котором остаётся место под типичный промпт:
    # так лимит режется лишь когда 400 практически неизбежен, а не «на всякий
    # случай» (иначе мы бы сами занижали бюджет генератора и плодили обрывы).
    hard_cap = context_length - _PROMPT_MARGIN
    for role_name, role in solver_mod.ROLES.items():
        limit = role.num_predict or 0
        if limit > hard_cap:
            if hard_cap < 1000:
                print(f"[config] ОШИБКА: контекст сервера {context_length} слишком мал "
                      f"для роли '{role_name}'. Поднимите --max-model-len.")
                continue
            solver_mod.ROLES[role_name] = replace(role, num_predict=hard_cap)
            print(
                f"[config] num_predict роли '{role_name}' урезан {limit} -> {hard_cap} "
                f"под контекст сервера {context_length} (иначе HTTP 400 на каждом "
                f"вызове). Для полного лимита поднимите --max-model-len до "
                f"{limit + _PROMPT_MARGIN} (генератору комфортно 65536)."
            )
            limit = hard_cap
        # Мягкое предупреждение: запрос пройдёт, но на промпт остаётся мало.
        soft_reserve = solver_mod.MAX_CONTEXT_CHARS // 4 + 1500
        if limit + soft_reserve > context_length:
            print(
                f"[config] ВНИМАНИЕ: у роли '{role_name}' num_predict={limit} при "
                f"контексте {context_length} — на промпт и накопленные шаги остаётся "
                f"{context_length - limit} токенов. На глубоких шагах возможны обрывы."
            )


def warn_on_thinking_budget() -> None:
    """Предупреждает, если у роли включены размышления при малом num_predict."""
    for role_name, role in solver_mod.ROLES.items():
        if role.enable_thinking and (role.num_predict or 0) < _THINKING_MIN_TOKENS:
            print(
                f"[config] ВНИМАНИЕ: у роли '{role_name}' включены размышления, но "
                f"num_predict={role.num_predict} < {_THINKING_MIN_TOKENS}. Ответ будет "
                f"обрываться на середине размышлений и приходить пустым. Поднимите "
                f"num_predict в yaml или используйте --thinking off."
            )


def report_resource_budget(args: argparse.Namespace, context_length: int | None) -> None:
    """Печатает потолки по памяти до старта прогона.

    Смысл в том, чтобы столкновение с пределами машины было видно заранее, а не
    через полчаса свопа. Считаем только клиентскую часть; сервер модели, если он
    на той же машине, занимает свою память сверх этого.
    """
    from tools import MAX_SANDBOX_WORKERS, MEMORY_LIMIT_MB

    concurrent = max(1, args.workers)
    sandbox_ceiling_gb = MAX_SANDBOX_WORKERS * MEMORY_LIMIT_MB / 1024

    print(
        f"[budget] одновременных запросов к серверу: {concurrent} (= workers)"
    )
    print(
        f"[budget] песочница: до {MAX_SANDBOX_WORKERS} процессов, "
        f"потолок {sandbox_ceiling_gb:.1f} ГБ (SANDBOX_WORKERS x SANDBOX_MEMORY_MB)"
    )

    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1024 ** 3
        print(f"[budget] свободно RAM сейчас: {available_gb:.1f} ГБ")
        if sandbox_ceiling_gb > available_gb * 0.5:
            print(
                f"[budget] ВНИМАНИЕ: песочница одна может занять {sandbox_ceiling_gb:.1f} ГБ "
                f"при {available_gb:.1f} ГБ свободных. Уменьшите SANDBOX_WORKERS "
                f"или SANDBOX_MEMORY_MB."
            )
    except ImportError:
        pass

    if isinstance(context_length, int):
        # Оценка KV-кэша сверху для 7B-класса с GQA (~56 KiB на токен).
        kv_gb = concurrent * context_length * 56 / 1024 ** 2
        print(
            f"[budget] KV-кэш сервера в пике: ~{kv_gb:.1f} ГБ "
            f"({concurrent} посл. x {context_length} токенов, худший случай). "
            f"Должно помещаться в пул vLLM за вычетом весов; "
            f"на A100-80GB это ~57 ГБ."
        )


def preflight_check(base_url: str, api_key: str, model: str) -> None:
    """Проверяет сервер ДО старта прогона.

    Без этой проверки недоступный сервер не останавливает бенчмарк: каждый
    сэмпл ловит APIConnectionError по отдельности, прогон честно доходит до
    конца и записывает файл, полный null-ответов. Дешевле упасть сразу.
    """
    import requests

    url = f"{base_url.rstrip('/')}/models"
    try:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.ConnectionError as exc:
        raise ServerUnavailable(
            f"Сервер модели не отвечает на {url}.\n"
            f"  Похоже, vLLM не запущен. Поднимите его, например:\n"
            f"    vllm serve {model} --max-model-len 32768 \\\n"
            f"      --enable-auto-tool-choice --tool-call-parser hermes\n"
            f"  Исходная ошибка: {type(exc).__name__}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ServerUnavailable(f"Не удалось опросить {url}: {type(exc).__name__}: {exc}") from exc

    served = {entry.get("id"): entry for entry in payload.get("data", [])}
    if model not in served:
        raise ServerUnavailable(
            f"Сервер на {base_url} не отдаёт модель '{model}'.\n"
            f"  Доступны: {', '.join(served) or '(пусто)'}\n"
            f"  Передайте правильное имя через --model."
        )

    context_length = served[model].get("max_model_len")
    print(f"[preflight] {model} доступна, max_model_len={context_length}")
    if isinstance(context_length, int) and context_length < 6000:
        print(
            f"[preflight] ВНИМАНИЕ: контекст {context_length} мал даже для одного "
            f"решения — сэмплы будут обрезаться и терять голос."
        )
    return context_length


def load_completed_task_ids(path: Path) -> set[str]:
    """task_id из уже записанных строк — чтобы не пересчитывать их при --resume.

    Битые строки в конце файла (обрыв прогона на середине записи) молча
    пропускаются: перерешать такую задачу дешевле, чем падать на старте.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id is None:
                continue
            metadata = record.get("metadata") or {}
            # Перерешиваем всё, что не дало ответа: и явные исключения, и
            # случай, когда упали все сэмплы (тогда task-level error пуст, а
            # solution = null — такую запись нельзя считать решённой).
            failed = (
                metadata.get("error")
                or (metadata.get("agent_metrics") or {}).get("error")
                or not str(record.get("solution") or "").strip()
            )
            if not failed:
                done.add(str(task_id))
    return done


def _solve_with_graph(graph, problem: str, args: argparse.Namespace) -> tuple[str | None, dict]:
    """Пошаговый режим: возвращает (ответ, метрики агента)."""
    reset_calculator_state()
    # Каждый пайплайн знает форму своего состояния. qwen4b/-full предоставляют
    # make_initial_state (у них есть свои поля вроде candidate_raw/segmented);
    # оригинал такой функции не имеет — для него собираем состояние по-старому.
    if hasattr(solver_mod, "make_initial_state"):
        initial_state = solver_mod.make_initial_state(problem, args)
    else:
        initial_state = {
            "problem": problem,
            "steps": [],
            "candidate_steps": [],
            "candidate_scores": [],
            "k_branches": args.k_branches,
            "score_threshold": args.score_threshold,
            "branch_mode": args.branch_mode,
            "base_temperature": args.temperature,
            "tokens_used": 0,
            "token_budget": args.token_budget,
            "in_recovery": False,
            "max_recoveries": args.max_recoveries,
            "total_recovery_events": 0,
            "stuck_streak": 0,
            "max_stuck_steps": args.max_stuck_steps,
            "unreliable_eval_streak": 0,
            "max_unreliable_evals": args.max_unreliable_evals,
            "eval_history": [],
            "eval_advice": "",
            "thinking_overruns": 0,
            "final_answer": None,
            "is_valid": False,
            "verifier_rationale": "",
            "gave_up": False,
            "gave_up_reason": "",
            "step_recovery_attempts": 0,
            "max_step_attempts": 3,
            "use_tools": not getattr(args, "no_tools", False),
        }
    state = graph.invoke(initial_state)
    history = state.get("eval_history") or []
    all_scores = [s for round_ in history for s in round_.get("scores", [])]
    metrics = {
        "mode": "solver",
        "tokens_used": state.get("tokens_used", 0),
        "is_valid": state.get("is_valid", False),
        "verifier_rationale": state.get("verifier_rationale"),
        "gave_up": state.get("gave_up", False),
        "gave_up_reason": state.get("gave_up_reason"),
        "steps_count": len(state.get("steps", [])),
        # Поведение оценщика: сколько раундов, сколько кандидатов он завернул,
        # и были ли раунды, где ни один ответ не распарсился.
        "eval_rounds": len(history),
        "eval_candidates": len(all_scores),
        "eval_rejected": sum(1 for s in all_scores if s < args.score_threshold),
        "eval_unreliable_rounds": sum(1 for r in history if not r.get("reliable")),
        "recovery_rounds": sum(1 for r in history if r.get("in_recovery")),
        "eval_history": history,
        # Сколько генераций пришлось повторить без размышлений: ненулевое
        # значение означает, что замер "с ризонингом" неоднородный.
        "thinking_overruns": state.get("thinking_overruns", 0),
    }
    # Метрики, специфичные для пайплайнов с сегментацией. Без них из JSONL не
    # видно ни как часто звался сегментатор, ни (главное) на какой глубине принят
    # ответ — а именно это показывает, не выродился ли пошаговый режим в CoT.
    for extra in ("segmenter_calls", "segmenter_unreliable",
                  "premature_answers", "answer_depth", "api_errors",
                  "tool_calls", "tool_salvaged"):
        if extra in state:
            metrics[extra] = state.get(extra)
    return state.get("final_answer"), metrics


def run_benchmark(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    """Решает задачи бенчмарка выбранным режимом и пишет результаты в JSONL."""
    from datasets import load_dataset

    # Выбираем активный пайплайн и его дефолтный yaml промптов.
    global solver_mod
    solver_mod = PIPELINES[getattr(args, "pipeline", "default")]
    if getattr(args, "prompt", None) is None:
        args.prompt = DEFAULT_PROMPTS[getattr(args, "pipeline", "default")]
    print(f"[config] pipeline={args.pipeline} (модуль {solver_mod.__name__}), "
          f"промпты: {args.prompt}")

    try:
        context_length = preflight_check(args.base_url, args.api_key, args.model)
    except ServerUnavailable as exc:
        print(f"\n[preflight] {exc}\n")
        return 2
    report_resource_budget(args, context_length)

    solver_mod.MODEL_NAME = args.model
    solver_mod.BASE_URL = args.base_url
    solver_mod.API_KEY = args.api_key
    solver_mod.DEFAULT_MAX_TOKENS = args.max_tokens
    solver_mod.REQUEST_TIMEOUT = args.timeout

    dataset = load_dataset_offline_safe(config, getattr(args, "data_file", None))

    # Предфильтр по формату эталона (для датасетов со смешанными ответами).
    # Применяется ДО skip/limit, чтобы --limit N отсчитывался от отобранных
    # задач, а не от исходных с дырами.
    if config.answer_filter is not None:
        before = len(dataset)
        gt = config.ground_truth_field
        dataset = dataset.filter(lambda item: config.answer_filter(str(item.get(gt, ""))))
        print(f"[filter] {config.answer_filter_name or 'answer_filter'}: "
              f"оставлено {len(dataset)}/{before} задач")
        if len(dataset) == 0:
            raise ValueError("answer_filter отсеял все задачи — проверьте фильтр и поле эталона")

    start = max(0, args.skip)
    end = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if start or end != len(dataset):
        dataset = dataset.select(range(start, end))

    output_path = args.resume or resolve_output_path(config, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_path: Path | None = None
    if args.trajectory:
        trajectory_path = (
            output_path.with_name(output_path.stem + "_trajectory.json")
            if args.trajectory == "auto" else Path(args.trajectory)
        )
        RECORDER.enabled = True
        RECORDER.run_meta = {
            "benchmark": config.name,
            "dataset": config.dataset_name,
            "split": config.split,
            "model": args.model,
            "pipeline": getattr(args, "pipeline", "default"),
            "prompt": str(args.prompt),
            "thinking": args.thinking,
            "use_tools": not args.no_tools,
            "branch_mode": args.branch_mode,
            "k_branches": args.k_branches,
            "score_threshold": args.score_threshold,
            "token_budget": args.token_budget,
            # Нужны для manifest.json при выгрузке в общий формат сравнения.
            "temperature": (args.temperature
                            if args.temperature is not None
                            else solver_mod.ROLES["generator"].temperature),
            "workers": args.workers,
            "timeout": args.timeout,
            "max_tokens": args.max_tokens,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(f"[trajectory] запись траекторий включена -> {trajectory_path}")
    completed = load_completed_task_ids(output_path) if args.resume else set()
    if completed:
        print(f"[resume] {len(completed)} задач уже решено в {output_path}, пропускаю их")

    pending = [
        item for item in dataset
        if str(item[config.task_id_field]) not in completed
    ]
    if not pending:
        print("Все задачи уже решены — нечего запускать.")
        return 0

    if args.prompt:
        solver_mod.load_prompts_from_yaml(args.prompt)
    apply_thinking_override(args.thinking)
    warn_on_thinking_budget()
    warn_on_context_fit(context_length)

    # Эффективная температура генератора для логов: явный --temperature, иначе
    # то, что реально возьмёт generate_step — temperature роли из yaml.
    effective_solver_temperature = (
        args.temperature
        if args.temperature is not None
        else solver_mod.ROLES["generator"].temperature
    )
    if args.temperature is not None:
        print(f"[config] --temperature={args.temperature} переопределяет "
              f"generator.temperature из yaml")

    graph = solver_mod.build_solver_graph()

    write_lock = threading.Lock()
    counters = {"done": 0, "errors": 0, "consecutive_dead": 0}
    total = len(pending)
    width = len(str(total))
    mode = "a" if args.resume else "w"
    # Предохранитель на случай, если сервер умрёт посреди прогона: нет смысла
    # молотить оставшиеся задачи, записывая null-ответы.
    abort = threading.Event()

    def solve_one(item: dict[str, Any]) -> dict[str, Any]:
        task_id = str(item[config.task_id_field])
        problem = item[config.problem_field]
        ground_truth = str(item.get(config.ground_truth_field, ""))
        started = time.perf_counter()
        solution: str | None = None
        error: str | None = None
        agent_metrics: dict[str, Any] = {}

        if abort.is_set():
            return build_record(
                config, task_id, None, ground_truth, args.model,
                {
                    "dataset": config.dataset_name,
                    "error": "skipped: сервер модели недоступен",
                    "agent_metrics": {},
                },
            )

        RECORDER.start_task(task_id, problem, ground_truth=ground_truth,
                            benchmark=config.name)
        try:
            solution, agent_metrics = _solve_with_graph(graph, problem, args)
        except Exception as exc:  # noqa: BLE001 — одна задача не валит прогон
            error = f"{type(exc).__name__}: {exc}"
            with write_lock:
                counters["errors"] += 1
            print(f"  ❌ task {task_id}: {error}")
        finally:
            RECORDER.finish_task(
                final_answer=solution, ground_truth=ground_truth, error=error,
                latency_seconds=round(time.perf_counter() - started, 3),
                metrics=agent_metrics,
            )

        # Задача, не давшая ни одного ответа, — сигнал что сервер отвалился.
        # Две подряд считаем достаточным поводом остановиться. НО: честный
        # give_up (модель не смогла, сеть цела) — не улика падения сервера.
        # Пайплайны с полем api_errors (qwen4b) позволяют это различить: без
        # прямой улики сетевого сбоя give_up не должен абортить прогон. Иначе,
        # как на Qwen3.5-4B, два give_up подряд ложно пометили 13 живых задач
        # как "сервер недоступен" (api_errors=0). SC и оригинальный солвер такого
        # поля не отдают — для них поведение прежнее.
        if agent_metrics.get("api_errors") is not None:
            produced_nothing = solution is None and bool(agent_metrics.get("api_errors") or error)
        else:
            produced_nothing = solution is None and not agent_metrics.get("n_valid_samples")
        with write_lock:
            if produced_nothing:
                counters["consecutive_dead"] += 1
                if counters["consecutive_dead"] >= 2 and not abort.is_set():
                    abort.set()
                    print(
                        "\n[ABORT] Две задачи подряд не дали ни одного ответа — "
                        "похоже, сервер модели отвалился. Останавливаю прогон.\n"
                        f"        Продолжить после починки: --resume {output_path}"
                    )
            else:
                counters["consecutive_dead"] = 0

        return build_record(
            config,
            task_id,
            solution,
            ground_truth,
            args.model,
            {
                "dataset": config.dataset_name,
                **{field: item.get(field) for field in config.metadata_fields},
                "prompt_version": args.pipeline,
                "temperature": effective_solver_temperature,
                "max_tokens": args.max_tokens,
                # Без этого прогоны с тулами и без них неразличимы в результатах.
                "use_tools": not args.no_tools,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": error,
                "agent_metrics": agent_metrics,
            },
        )

    print(f"Пайплайн: {args.pipeline} | задач: {total} | параллельно: {args.workers} | вывод: {output_path}")
    try:
        with output_path.open(mode, encoding="utf-8") as output:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for record in pool.map(solve_one, pending):
                    with write_lock:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        counters["done"] += 1
                        print(
                            f"[{counters['done']:0{width}d}/{total:0{width}d}] "
                            f"{config.name} task {record['task_id']} -> {record['solution']!r} "
                            f"(эталон {record['ground_truth']})"
                        )
    finally:
        # Процессы-песочницы переиспользуются между задачами, поэтому гасим их
        # один раз в конце — в том числе при Ctrl+C, чтобы не оставлять сирот.
        shutdown_workers()
        # Дамп даже при обрыве: половина траекторий полезнее, чем ничего.
        if trajectory_path is not None:
            saved = RECORDER.dump(trajectory_path)
            if saved:
                print(f"[trajectory] сохранено: {saved}")
                print(f"[trajectory] просмотр:  python scripts/make_viewer.py {saved}")

    print(f"Saved {counters['done']} records to {output_path}")
    if counters["errors"]:
        print(f"Задач с ошибками: {counters['errors']}")

    # Сверка идёт ПОСЛЕ записи JSONL и не влияет на код возврата прогона:
    # результат многочасовой работы не должен зависеть от того, отработал ли
    # math_verify.
    if not args.no_verify and counters["done"]:
        run_verification(config, output_path, args)
    elif args.no_verify:
        print(f"[verify] пропущено (--no-verify). Вручную: "
              f"python {'verify_imo_answers.py' if config.verifier == 'imo' else 'verify_answers.py'} "
              f"{output_path}")

    return 1 if counters["errors"] else 0
