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
import run_console

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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Зерно выборки. Без него каждый прогон — независимая выборка, и "
             "32%% задач плавают от прогона к прогону: одиночным прогоном "
             "эффект правки промпта измерить нельзя. Зерно каждого вызова "
             "выводится из (seed, task_id, номер вызова), поэтому прогоны "
             "воспроизводятся, а ветки внутри прогона остаются разными.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip", type=int, default=0, help="Пропустить N первых задач")
    parser.add_argument(
        "--task-ids", default=None,
        help="Гонять только эти задачи: список id через запятую либо путь к файлу "
             "с id (по одному в строке, '#' — комментарий). Нужен для замеров на "
             "фиксированном подмножестве: --skip/--limit режут только непрерывный "
             "кусок, а подмножество «15 решаемых + 15 нерешаемых» непрерывным не "
             "бывает. Применяется ДО --skip/--limit.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Сколько задач бенчмарка решать параллельно. Одновременных запросов "
             "к серверу будет workers * sample-workers — именно это число определяет "
             "пиковый размер KV-кэша. Дефолт рассчитан на выделенный сервер с A100; "
             "на слабой машине снижайте",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Печатать подробности каждого узла (прежнее поведение). По "
             "умолчанию в терминал идёт одна строка на задачу, а подробности "
             "живут в траектории и JSONL.",
    )
    parser.add_argument(
        "--color", choices=["auto", "always", "never"], default="auto",
        help="Цвет в терминале. auto — только когда stdout это терминал: при "
             "перенаправлении в файл escape-последовательности только мешают.",
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
    # Дефолт single: замер §2.5 показал, что ветвление на каждом шаге не даёт
    # ничего (+18% токенов, 0 задач), а верного ответа не было ни в одной
    # отвергнутой ветке. Все реальные прогоны и так шли с single — дефолт
    # приведён в соответствие. В recovery multi включается независимо от флага.
    group.add_argument("--branch-mode", default="single", choices=["single", "multi"])
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
    # 5 был жёстким стопом на всю задачу: замер v9 показал, что 7 сдач из 8 —
    # именно «recovery budget exhausted», причём один трудный depth съедал
    # бюджет и задача умирала, не дойдя до более лёгких шагов. Теперь это мягкий
    # потолок, а жёсткий лимит на попытки живёт per-depth (--max-step-attempts).
    group.add_argument("--max-recoveries", type=int, default=15)
    group.add_argument(
        "--max-step-attempts", type=int, default=3,
        help="Сколько раз пробовать ОДИН и тот же шаг, прежде чем признать его "
             "безнадёжным. Именно этот лимит теперь жёсткий, а не глобальный.",
    )
    group.add_argument(
        "--finish-at", type=float, default=0.85,
        help="Доля бюджета, после которой агент перестаёт генерировать новые шаги "
             "и один раз пробует собрать ответ из уже принятых. Ждать полного "
             "исчерпания нельзя: на сам сбор ответа бюджета уже не останется.",
    )
    group.add_argument(
        "--samples", type=int, default=1,
        help="Сколько независимых решений задачи брать перед голосованием. "
             "1 — как раньше. Смысл: 32%% задач решаются то верно, то нет от "
             "прогона к прогону; голосование забирает эту полосу. Второй и "
             "третий сэмплы запускаются только для ненадёжных задач "
             "(см. --resample-token-threshold), поэтому цена далеко не кратная.",
    )
    group.add_argument(
        "--resample-token-threshold", type=int, default=400000,
        help="Порог расхода, выше которого первый сэмпл считается ненадёжным и "
             "запускается добор. 0 — добирать всегда при --samples > 1.",
    )
    group.add_argument(
        "--resample-min-steps", type=int, default=0,
        help="Добирать сэмплы, если решение получено за это число принятых шагов "
             "или меньше. Общая эвристика: очень короткое решение олимпиадной "
             "задачи подозрительно, а порог по расходу такие случаи не ловит — "
             "они дешёвые. 0 (по умолчанию) — правило выключено.",
    )
    group.add_argument(
        "--vote-tiebreak-samples", type=int, default=1,
        help="Сколько дополнительных сэмплов запускать, если большинства нет "
             "(все варианты набрали поровну). При ничьей победитель иначе "
             "определяется порядком сэмплов, а не голосами. 0 — не добирать.",
    )
    group.add_argument(
        "--split-sample-budget", action="store_true",
        help="Делить --token-budget между сэмплами, чтобы суммарный расход "
             "задачи не рос кратно числу сэмплов. По умолчанию выключено: "
             "включение меняет условия замера, поэтому его стоит вводить "
             "отдельным экспериментом.",
    )
    group.add_argument(
        "--no-verify-step", action="store_true",
        help="Не звать верификатора в конце задачи. Его вердикт ни на что не "
             "влияет, ошибается на верных задачах в 41% случаев и стоит ~4% "
             "бюджета — на замерах его стоит отключать.",
    )
    group.add_argument(
        "--evaluator-mode", default="llm", choices=["llm", "random"],
        help="Чем оценивать шаги. 'llm' — роль evaluator из промпта (как всегда). "
             "'random' — контрольный режим: модель не зовётся, вердикт бросается "
             "монетой по той же бинарной шкале 0.0/1.0 с той же долей нулей "
             "(RANDOM_EVAL_REJECT_RATE, по умолчанию 0.6 — как у живого оценщика). "
             "Нужен, чтобы измерить вклад оценщика: если счёт не изменился, "
             "вердикт не нёс информации. Работает только у --pipeline qwen4b.",
    )
    group.add_argument(
        "--sampling", default="",
        help="Явные параметры сэмплирования для ВСЕХ ролей, через запятую: "
             "'top_p=1.0,top_k=-1,min_p=0'. Пусто (по умолчанию) — не слать их "
             "вовсе, как было всегда; тогда значения подставляет движок, и они "
             "у vLLM и SGLang разные. Это единственное известное различие между "
             "прогоном на 28/30 и нынешними 23/30 (§5 п.00 памяти). "
             "Только --pipeline qwen4b.",
    )
    group.add_argument(
        "--evaluator-min-depth", type=int, default=0,
        help="Глубина, начиная с которой оценщик включается. 0 (по умолчанию) — "
             "оценивать все шаги. 2 — шаги на глубинах 0 и 1 принимать без вызова "
             "модели. Замер по базовым прогонам: там 83%% всех оценок и 87%% всех "
             "отвержений, причём строгость выше (69%%/66%% против 49-50%% глубже). "
             "Флаг проверяет, работа это или шум. Только --pipeline qwen4b.",
    )
    group.add_argument(
        "--min-steps-before-answer", type=int, default=0,
        help="Сколько шагов должно быть принято, прежде чем \\boxed{} засчитывается "
             "как финальный ответ. Работает для ОБОИХ пайплайнов. 0 (по умолчанию) "
             "— ответ принимается сразу: если задача решается за один шаг, незачем "
             "тратить ещё один раунд генерации. 1 — запрещает ответ на глубине 0 и "
             "заставляет пайплайн сделать хотя бы один промежуточный шаг. "
             "ВАЖНО для 9B: замер 0810/0811 показал, что 96-100% первых шагов уже "
             "содержат финальный \\boxed, то есть пайплайн вырождается в CoT. При "
             "0 сегментатор всё равно приклеит найденный ответ к первому шагу, и "
             "вырождение сохранится — чтобы проверить пошаговый режим, нужен 1.",
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


def parse_task_ids(value: "str | None") -> "set[str] | None":
    """Разбирает --task-ids: либо путь к файлу, либо список через запятую.

    Файл удобнее для длинных подмножеств и тем, что его можно закоммитить рядом
    с замером: подмножество — часть методики, а не разовый аргумент командной
    строки.
    """
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        raw = candidate.read_text(encoding="utf-8").splitlines()
    else:
        raw = value.split(",")
    ids = {
        line.split("#", 1)[0].strip()
        for line in raw
    }
    ids.discard("")
    if not ids:
        raise ValueError(f"--task-ids не содержит ни одного id: {value!r}")
    return ids


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


def _vote(answers: list[str]) -> tuple[str | None, dict]:
    """Голосование по финальным ответам нескольких независимых решений.

    Сравнение по нормализованной форме (иначе `\\frac{1}{2}` и `\\frac12`
    считались бы разными ответами), а возвращается исходная запись — её потом
    разбирает math-verify.
    """
    from answer_utils import normalize_answer

    buckets: dict[str, list[str]] = {}
    for a in answers:
        a = (a or "").strip()
        if not a:
            continue
        buckets.setdefault(normalize_answer(a) or a, []).append(a)
    if not buckets:
        return None, {"votes": {}, "agreement": 0.0, "quorum": False, "tied": 0}
    ranked = sorted(buckets.values(), key=len, reverse=True)
    best = ranked[0]
    top = len(best)
    # Сколько вариантов делят первое место. Если больше одного, победитель
    # выбран порядком сэмплов, а не голосами: замер на hmmt показал, что в двух
    # задачах из пяти таких верный ответ ПРИСУТСТВОВАЛ и проиграл ничью
    # (№24 ['', '4', '20'] при эталоне 20; №10 то же с парой корней).
    tied = sum(1 for b in ranked if len(b) == top)
    return best[0], {
        "votes": {k: len(v) for k, v in buckets.items()},
        "agreement": top / max(len(answers), 1),
        # quorum=False означает «настоящего большинства нет»: либо все варианты
        # по одному голосу, либо несколько делят максимум.
        "quorum": top >= 2 and tied == 1,
        "tied": tied,
    }


# Счётчики: описывают РАСХОД и ПОВЕДЕНИЕ всей задачи, поэтому складываются по
# сэмплам. До 2026-08-12 суммировался только tokens_used, а остальное молча
# бралось из первого сэмпла: в прогоне v4 это показало segmenter_calls=5 при
# фактических 15, то есть диагностика занижалась во столько раз, сколько было
# сэмплов.
_ADDITIVE_METRICS = (
    "tokens_used", "eval_rounds", "eval_candidates", "eval_rejected",
    "eval_unreliable_rounds", "recovery_rounds", "thinking_overruns",
    "segmenter_calls", "segmenter_unreliable", "premature_answers",
    "answers_not_in_step_text", "api_errors", "tool_calls", "tool_salvaged",
)


def _merge_sample_metrics(metrics_all: list[dict], answers: list[str],
                          final: "str | None") -> dict:
    """Сводит метрики сэмплов в одну запись.

    Счётчики складываются, а описательные поля (steps_count, is_valid,
    verifier_rationale, answer_depth, eval_history) берутся у ТОГО сэмпла, чей
    ответ победил в голосовании: они характеризуют возвращённый ответ, и брать
    их у первого сэмпла неверно, когда победил третий.
    """
    if not metrics_all:
        return {}
    from answer_utils import normalize_answer

    winner = 0
    if final:
        key = normalize_answer(final) or final
        for idx, a in enumerate(answers):
            if a and (normalize_answer(a) or a) == key:
                winner = idx
                break
    merged = dict(metrics_all[winner])
    for name in _ADDITIVE_METRICS:
        vals = [m.get(name) for m in metrics_all
                if isinstance(m.get(name), (int, float)) and not isinstance(m.get(name), bool)]
        if vals:
            merged[name] = sum(vals)
    # Раскладка по сэмплам: без неё из суммы не видно, был ли расход ровным или
    # его создал один тяжёлый сэмпл.
    merged["winning_sample"] = winner
    merged["steps_per_sample"] = [m.get("steps_count") for m in metrics_all]
    merged["tokens_per_sample"] = [m.get("tokens_used") for m in metrics_all]
    return merged


def _solve_once(graph, problem: str, args: argparse.Namespace,
                task_id: Any = None) -> tuple[str | None, dict]:
    """Одно независимое решение задачи."""
    reset_calculator_state()
    if getattr(args, "seed", None) is not None and hasattr(solver_mod, "begin_task_seed"):
        # Зерно привязано к task_id, а не к порядку решения: при --workers N
        # задачи заканчиваются в произвольном порядке, и счётчик «по порядку»
        # давал бы разные зёрна одной задаче в разных прогонах.
        solver_mod.begin_task_seed(task_id)
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
            "api_errors": 0,
            "segmenter_calls": 0,
            "segmenter_unreliable": 0,
            "min_steps_before_answer": getattr(args, "min_steps_before_answer", 1),
            "final_answer": None,
            "is_valid": False,
            "verifier_rationale": "",
            "gave_up": False,
            "gave_up_reason": "",
            "step_recovery_attempts": 0,
            "max_step_attempts": getattr(args, "max_step_attempts", 3),
            "use_tools": not getattr(args, "no_tools", False),
            "skip_verifier": bool(getattr(args, "no_verify_step", False)),
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
                  "premature_answers", "answers_not_in_step_text",
                  "answer_depth", "api_errors",
                  "tool_calls", "tool_salvaged"):
        if extra in state:
            metrics[extra] = state.get(extra)
    return state.get("final_answer"), metrics


def _solve_with_graph(graph, problem: str, args: argparse.Namespace,
                      task_id: Any = None) -> tuple[str | None, dict]:
    """Решает задачу; при --samples > 1 берёт ответ большинством голосов.

    Зачем. Разбор всех чистых прогонов: 54 задачи из 90 решаются всегда, 7 не
    решаются никогда, а 29 (32%) плавают от прогона к прогону. Плавающие — это
    задачи, где верный ответ находится, но не каждый раз; одиночное решение
    выбрасывает эту информацию. Голосование по нескольким независимым решениям
    бьёт ровно в эту полосу.

    Добор адаптивный: второй и третий сэмплы запускаются только там, где первый
    выглядит ненадёжным (сдался, не дал ответа или сжёг больше
    --resample-token-threshold токенов). На 54 «всегда решаемых» задачах это не
    стоит ничего, а дорогие задачи и так дорогие.
    """
    samples = max(1, int(getattr(args, "samples", 1) or 1))
    if samples == 1:
        return _solve_once(graph, problem, args, task_id)

    if getattr(args, "split_sample_budget", False):
        # Бюджет задачи делится между сэмплами: иначе каждый сэмпл имеет право
        # израсходовать полный лимит, и трудная задача стоит кратно числу
        # сэмплов (замер hmmt: 2.3 млн токенов на задачу при лимите 800k).
        args = argparse.Namespace(**vars(args))
        args.token_budget = max(1, args.token_budget // samples)

    threshold = int(getattr(args, "resample_token_threshold", 400_000) or 0)
    min_steps = int(getattr(args, "resample_min_steps", 0) or 0)
    extra = max(0, int(getattr(args, "vote_tiebreak_samples", 1) or 0))
    answers: list[str] = []
    metrics_all: list[dict] = []
    planned = samples
    i = 0
    while i < planned:
        if getattr(args, "seed", None) is not None and hasattr(solver_mod, "begin_task_seed"):
            # Разные сэмплы одной задачи обязаны идти разными путями, иначе
            # голосование выродится в один и тот же ответ. Зерно остаётся
            # воспроизводимым: оно детерминировано номером сэмпла.
            #
            # Нулевой сэмпл намеренно берёт ГОЛЫЙ task_id — тот же, что и при
            # --samples 1. Тогда прогон с голосованием повторяет одиночный шаг в
            # шаг на первой траектории, и разница между ними — ровно вклад
            # добора, а не смена пути сэмплирования.
            solver_mod.begin_task_seed(task_id if i == 0 else f"{task_id}#{i}")
        answer, m = _solve_once(graph, problem, args, task_id)
        answers.append(answer or "")
        metrics_all.append(m)
        i += 1
        if i == 1:
            steps = m.get("steps_count") or 0
            # Порог по расходу не задевает «уверенно неверные» задачи: они дешёвые
            # и с ответом. На hmmt так были потеряны две задачи, где один сэмпл
            # дал неверный ответ и второго мнения не спросили. Признак общий, без
            # знания бенчмарка: решение из одного-двух шагов на олимпиадной задаче
            # подозрительно коротко.
            too_short = bool(min_steps) and steps <= min_steps
            # threshold == 0 означает «добирать всегда» (так написано в справке
            # флага). Раньше здесь стояло `threshold and ...`, и ноль как ложное
            # значение ВЫКЛЮЧАЛ проверку расхода целиком — то есть 0 давал прямо
            # противоположное: добор только у сдавшихся и безответных.
            over_budget = m.get("tokens_used", 0) > threshold if threshold else True
            shaky = (m.get("gave_up") or not (answer or "").strip()
                     or over_budget or too_short)
            if not shaky:
                print(f"  [samples] первый сэмпл уверенный — добор не нужен.")
                break
            why = ("сдался" if m.get("gave_up") else
                   "нет ответа" if not (answer or "").strip() else
                   f"шагов {steps} ≤ {min_steps}" if too_short else
                   "порог 0 — добираем всегда" if not threshold else
                   f"токенов {m.get('tokens_used', 0):,} > {threshold:,}")
            print(f"  [samples] первый сэмпл ненадёжен ({why}) — добираю.")

    final, vote_info = _vote(answers)

    # Ничья: победитель определился порядком сэмплов, а не голосами. Один
    # дополнительный сэмпл — единственный способ получить настоящее большинство;
    # «вернуть пусто» здесь строго хуже, потому что в части таких задач верный
    # ответ уже присутствует среди кандидатов и просто проиграл ничью.
    while (extra and not vote_info["quorum"] and any(a.strip() for a in answers)
           and len(answers) < planned + extra):
        print(f"  [samples] большинства нет (голоса {vote_info['votes']}) — "
              f"добираю сэмпл для разрешения ничьей.")
        if getattr(args, "seed", None) is not None and hasattr(solver_mod, "begin_task_seed"):
            solver_mod.begin_task_seed(f"{task_id}#{len(answers)}")
        answer, m = _solve_once(graph, problem, args, task_id)
        answers.append(answer or "")
        metrics_all.append(m)
        final, vote_info = _vote(answers)
    merged = _merge_sample_metrics(metrics_all, answers, final)
    merged["gave_up"] = final is None
    merged["samples"] = len(metrics_all)
    merged["sample_answers"] = answers
    merged.update(vote_info)
    if len(metrics_all) > 1:
        print(f"  [samples] {len(metrics_all)} сэмплов, голоса {vote_info['votes']}, "
              f"согласие {vote_info['agreement']:.0%}, кворум={vote_info['quorum']} "
              f"-> {final!r}")
    return final, merged


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
    mode = getattr(args, "evaluator_mode", "llm")
    if mode != "llm":
        if hasattr(solver_mod, "EVALUATOR_MODE"):
            solver_mod.EVALUATOR_MODE = mode
            print(f"[config] ⚠️  ОЦЕНЩИК В КОНТРОЛЬНОМ РЕЖИМЕ '{mode}': вердикты "
                  f"не от модели, доля отвержений "
                  f"{solver_mod.RANDOM_EVAL_REJECT_RATE:.0%}. Это не замер "
                  f"качества пайплайна, а проверка вклада оценщика.")
        else:
            print(f"[config] ⚠️  --evaluator-mode={mode} игнорируется: пайплайн "
                  f"'{args.pipeline}' его не поддерживает")

    sampling = (getattr(args, "sampling", "") or "").strip()
    if sampling:
        if hasattr(solver_mod, "parse_sampling_overrides"):
            try:
                overrides = solver_mod.parse_sampling_overrides(sampling)
            except ValueError as exc:
                print(f"\n[config] ❌ --sampling: {exc}\n")
                return 2
            solver_mod.SAMPLING_OVERRIDES = overrides
            print(f"[config] сэмплирование задано явно: "
                  + ", ".join(f"{k}={v}" for k, v in overrides.items())
                  + " — эти значения идут во ВСЕ вызовы вместо дефолтов движка")
        else:
            print(f"[config] ⚠️  --sampling={sampling} игнорируется: пайплайн "
                  f"'{args.pipeline}' его не поддерживает")

    min_depth = int(getattr(args, "evaluator_min_depth", 0) or 0)
    if min_depth:
        if hasattr(solver_mod, "EVALUATOR_MIN_DEPTH"):
            solver_mod.EVALUATOR_MIN_DEPTH = min_depth
            print(f"[config] ⚠️  ОЦЕНЩИК ВКЛЮЧАЕТСЯ С ГЛУБИНЫ {min_depth}: шаги "
                  f"на глубинах 0..{min_depth - 1} принимаются без вызова модели.")
        else:
            print(f"[config] ⚠️  --evaluator-min-depth={min_depth} игнорируется: "
                  f"пайплайн '{args.pipeline}' его не поддерживает")

    if getattr(args, "seed", None) is not None:
        if hasattr(solver_mod, "SEED"):
            solver_mod.SEED = args.seed
            print(f"[config] seed={args.seed} — прогон воспроизводим "
                  f"(зерно вызова выводится из seed + task_id + номер вызова)")
        else:
            print(f"[config] ⚠️  --seed={args.seed} игнорируется: пайплайн "
                  f"'{args.pipeline}' не поддерживает зерно")

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

    # Фиксированное подмножество задач. Идёт после answer_filter и до skip/limit:
    # так «--task-ids файл --limit 5» означает «первые пять из подмножества».
    wanted = parse_task_ids(getattr(args, "task_ids", None))
    if wanted is not None:
        before = len(dataset)
        tid = config.task_id_field
        dataset = dataset.filter(lambda item: str(item[tid]) in wanted)
        found = {str(item[tid]) for item in dataset}
        missing = wanted - found
        print(f"[task-ids] отобрано {len(dataset)}/{before} задач по списку из {len(wanted)} id")
        if missing:
            print(f"[task-ids] ⚠️  не найдены в датасете: {sorted(missing)[:20]}"
                  f"{' …' if len(missing) > 20 else ''}")
        if len(dataset) == 0:
            raise ValueError(
                f"--task-ids не совпал ни с одной задачей. Поле id этого бенчмарка — "
                f"'{tid}'; проверьте, что в списке значения именно из него.")

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
            # temperature и roles дописываются ниже, после загрузки промптов:
            # здесь ROLES ещё держит аварийные значения по умолчанию.
            "workers": args.workers,
            "timeout": args.timeout,
            "max_tokens": args.max_tokens,
            # Всё, что делает прогоны сопоставимыми или несопоставимыми, обязано
            # лежать в шапке траектории. Без seed нельзя отличить сеянный прогон
            # от несеянного, а значит нельзя сказать, законно ли сравнение двух
            # прогонов вообще — именно на этом мы уже один раз обожглись.
            "seed": getattr(args, "seed", None),
            "samples": getattr(args, "samples", 1),
            "resample_token_threshold": getattr(args, "resample_token_threshold", None),
            "finish_at": getattr(args, "finish_at", None),
            "max_recoveries": args.max_recoveries,
            "max_step_attempts": getattr(args, "max_step_attempts", None),
            "verifier_enabled": not getattr(args, "no_verify_step", False),
            # Параметры ролей сюда НЕ кладём: промпты грузятся ниже, и на этот
            # момент ROLES ещё держит аварийные значения по умолчанию. Так поле
            # generator_num_predict показывало 24000 (дефолт из кода) вместо
            # реальных 40000 из yaml и увело внешний разбор прогона в сторону.
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

    # Параметры ролей дописываем в шапку ТОЛЬКО здесь — после load_prompts_from_yaml,
    # apply_thinking_override и warn_on_context_fit. Иначе в траекторию попадают
    # не те значения, с которыми прогон реально пойдёт.
    if RECORDER.enabled:
        RECORDER.run_meta.update({
            "temperature": effective_solver_temperature,
            "roles": {
                name: {"num_predict": role.num_predict,
                       "temperature": role.temperature,
                       "enable_thinking": role.enable_thinking}
                for name, role in solver_mod.ROLES.items()
            },
        })
        gen = solver_mod.ROLES.get("generator")
        if gen is not None:
            RECORDER.run_meta["generator_num_predict"] = gen.num_predict
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
        run_console.begin_task()
        try:
            solution, agent_metrics = _solve_with_graph(graph, problem, args, task_id)
        except Exception as exc:  # noqa: BLE001 — одна задача не валит прогон
            error = f"{type(exc).__name__}: {exc}"
            with write_lock:
                counters["errors"] += 1
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

        # Тревожные строки забираем здесь, пока буфер принадлежит этому потоку;
        # печатает их поток-писатель вместе с итоговой строкой задачи.
        alerts = run_console.take_alerts()
        return build_record(
            config,
            task_id,
            solution,
            ground_truth,
            args.model,
            {
                "dataset": config.dataset_name,
                "_console": {"alerts": alerts, "error": error,
                             "elapsed": round(time.perf_counter() - started, 3)},
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

    run_console.configure_color(args.color)
    # Подробный вывод узлов глушим ЗДЕСЬ, после всех сообщений конфигурации:
    # предупреждения про контекст, зерно и промпты должны остаться видимыми.
    # Глушить надо КАЖДЫЙ модуль, который печатает: подмена работает через
    # globals модуля, и незаявленный модуль продолжает писать в терминал мимо
    # тихого режима. У qwen4b цикл инструментов написан внутри самого солвера,
    # поэтому хватало solver_mod; у 9B он вынесен в tool_generator_subgraph —
    # без него в терминал шли ровно и только логи tool-вызовов.
    run_console.install(
        [
            solver_mod,
            sys.modules.get("tools"),
            sys.modules.get("tool_generator_subgraph"),
        ],
        quiet=not args.verbose,
    )
    if not args.verbose and trajectory_path is None:
        print("[config] ⚠️  Тихий вывод без --trajectory: сырые генерации и вердикты "
              "никуда не сохранятся. Для разбора прогона добавьте --trajectory "
              "или --verbose.")
    print(f"Пайплайн: {args.pipeline} | задач: {total} | параллельно: {args.workers} | вывод: {output_path}")
    try:
        with output_path.open(mode, encoding="utf-8") as output:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for record in pool.map(solve_one, pending):
                    # Служебный блок для консоли в JSONL не пишем: он нужен
                    # только чтобы донести данные из рабочего потока сюда.
                    console = record["metadata"].pop("_console", {})
                    with write_lock:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        counters["done"] += 1
                        metrics = record["metadata"].get("agent_metrics") or {}
                        status = run_console.status_of(
                            record["solution"], record["ground_truth"],
                            console.get("error"))
                        counters[status] = counters.get(status, 0) + 1
                        # Ноль токенов — генератор не получил ответа ни разу.
                        # Это всегда сбой связи, а не свойство задачи.
                        if not metrics.get("tokens_used"):
                            counters["dead"] = counters.get("dead", 0) + 1
                        extra = f"task {record['task_id']}"
                        if metrics.get("samples", 1) > 1:
                            extra += f" | sm {metrics['samples']}"
                        if metrics.get("gave_up"):
                            extra += " | gave up"
                        run_console.emit(
                            run_console.task_line(
                                counters["done"], total, status,
                                record["solution"], record["ground_truth"],
                                metrics.get("tokens_used", 0),
                                console.get("elapsed", 0.0), extra),
                            console.get("alerts", []), console.get("error"))
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

    tally = " | ".join(f"{name} {counters[name]}"
                       for name in ("PASS", "FAIL", "EMPTY", "ERROR", "?")
                       if counters.get(name))
    print(f"Saved {counters['done']} records to {output_path}"
          + (f"   ({tally})" if tally else ""))
    if tally:
        print("Статусы в строках задач — быстрая сверка для глаз; итоговая "
              "точность считается ниже отдельным процессом.")
    if counters["errors"]:
        print(f"Задач с ошибками: {counters['errors']}")

    # Прогон, где задачи не потратили ни одного токена, — это не замер, а
    # недоступный сервер. Без этой проверки он завершается кодом 0, диспетчер
    # публикует его как состоявшийся, и в results/ ложится правдоподобное с виду
    # «0/30». Ровно так и вышло 2026-08-14: на машине кончилось место, SGLang
    # умер через шесть минут после старта, все 30 задач получили ConnectionError,
    # а прогон записался как done — с нулём, который легко принять за результат
    # эксперимента.
    dead = counters.get("dead", 0)
    if dead:
        print(f"\n⚠️  Задач без единого токена: {dead} из {counters['done']} — "
              f"генератор ни разу не получил ответа от сервера.")
    if counters["done"] and dead >= max(1, int(counters["done"] * 0.3)):
        print("❌ ПРОГОН НЕДОСТОВЕРЕН: сервер был недоступен большую часть "
              "времени. Сверку не запускаю и возвращаю ненулевой код, чтобы "
              f"результат не ушёл как состоявшийся. JSONL сохранён: {output_path}")
        return 1

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
