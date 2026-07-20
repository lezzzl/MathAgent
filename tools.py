"""Инструмент исполнения Python-кода для агента.

Отличия от прежнего calculator():
  * sympy вместо голого math — для AIME нужна точная арифметика (Rational,
    solve, factorint), а не float;
  * exec многострочного кода с захватом stdout вместо eval одного выражения —
    половина олимпиадных задач решается коротким перебором со скриптом;
  * персистентный воркер-процесс вместо multiprocessing.Process на каждый
    вызов: на Windows spawn с реимпортом numpy/sympy стоил ~1 с на вызов;
  * состояние (анти-зацикливание, сам воркер) хранится per-thread, чтобы
    параллельные задачи бенчмарка не мешали друг другу.
"""

import ast
import io
import multiprocessing
import os
import queue
import threading
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.tools import tool

EXEC_TIMEOUT_SECONDS = 10.0
MAX_OUTPUT_CHARS = 2000
REPEAT_LIMIT = 3

# Сколько процессов-песочниц держать одновременно. Каждый несёт загруженные
# sympy+numpy (~200 МБ), поэтому число должно быть небольшим и НЕ зависеть от
# числа потоков: воркеры переиспользуются между задачами и сэмплами.
MAX_SANDBOX_WORKERS = int(os.getenv("SANDBOX_WORKERS", "4"))

# Потолок памяти на ОДИН воркер. Таймаут ограничивает время, но не память:
# перебор вида list(range(10**9)) успевает съесть десятки гигабайт за 10 с.
#
# Внимание: лимит поштучный, поэтому потолок всей песочницы —
# SANDBOX_WORKERS * SANDBOX_MEMORY_MB (по умолчанию 4 * 1024 = 4 ГБ).
# Держите это число с запасом ниже свободной RAM: сервер модели, если он
# крутится на той же машине, уже занимает её основную часть.
MEMORY_LIMIT_MB = int(os.getenv("SANDBOX_MEMORY_MB", "1024"))

try:
    import psutil  # опционально: без него работает только ограничение по времени
except ImportError:
    psutil = None

_thread_state = threading.local()


# ---------------------------------------------------------------------------
# Песочница
# ---------------------------------------------------------------------------
def _build_globals() -> Dict[str, Any]:
    """Пространство имён для исполняемого кода.

    Это НЕ песочница безопасности: sympy и numpy тянут за собой достаточно,
    чтобы ограничение __builtins__ обходилось. Изоляция здесь — от случайных
    ошибок и зависаний модели, а не от злонамеренного кода. Настоящую границу
    даёт отдельный процесс с таймаутом.
    """
    import math
    import cmath
    import fractions
    import itertools
    import random
    import re as _re
    import statistics

    import numpy as np
    import sympy as sp
    from sympy import (
        Eq, Matrix, Rational, Symbol, binomial, ceiling, cos, diff, divisors,
        expand, factor, factorial, factorint, floor, gcd, integrate, isprime,
        lcm, limit, log, mod_inverse, nsimplify, oo, pi, primerange, prime,
        nextprime, simplify, sin, solve, sqrt, symbols, tan, totient, I, E,
    )

    # Модели почти всегда начинают код со `from sympy import ...`, хотя всё уже
    # предзагружено. Без __import__ такой вызов падал целиком с
    # "ImportError: __import__ not found", и шаг терял вычисление.
    _real_import = __import__
    _allowed_modules = {
        "sympy", "math", "cmath", "numpy", "np", "itertools", "fractions",
        "decimal", "random", "re", "statistics", "collections", "functools",
        "operator", "heapq", "bisect", "string",
    }

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in _allowed_modules:
            raise ImportError(
                f"module '{name}' is not available in this sandbox. "
                f"sympy, numpy (as np), itertools, math and fractions are already imported."
            )
        return _real_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        "__import__": _safe_import,
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "complex": complex, "dict": dict, "divmod": divmod,
        "enumerate": enumerate, "filter": filter, "float": float,
        "format": format, "frozenset": frozenset, "hex": hex, "int": int,
        "isinstance": isinstance, "iter": iter, "len": len, "list": list,
        "map": map, "max": max, "min": min, "next": next, "oct": oct,
        "ord": ord, "pow": pow, "print": print, "range": range, "repr": repr,
        "reversed": reversed, "round": round, "set": set, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "True": True, "False": False, "None": None,
        "ValueError": ValueError, "TypeError": TypeError,
        "ZeroDivisionError": ZeroDivisionError, "Exception": Exception,
        "StopIteration": StopIteration, "KeyError": KeyError,
        "IndexError": IndexError, "ArithmeticError": ArithmeticError,
    }

    namespace: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math, "cmath": cmath, "np": np, "numpy": np,
        "sp": sp, "sympy": sp,
        "itertools": itertools, "fractions": fractions,
        "Fraction": fractions.Fraction, "statistics": statistics,
        "random": random, "re": _re,
        "Eq": Eq, "Matrix": Matrix, "Rational": Rational, "Symbol": Symbol,
        "binomial": binomial, "ceiling": ceiling, "cos": cos, "diff": diff,
        "divisors": divisors, "expand": expand, "factor": factor,
        "factorial": factorial, "factorint": factorint, "floor": floor,
        "gcd": gcd, "integrate": integrate, "isprime": isprime, "lcm": lcm,
        "limit": limit, "log": log, "mod_inverse": mod_inverse,
        "nsimplify": nsimplify, "oo": oo, "pi": pi, "prime": prime,
        "primerange": primerange, "nextprime": nextprime, "simplify": simplify,
        "sin": sin, "solve": solve, "sqrt": sqrt, "symbols": symbols,
        "tan": tan, "totient": totient, "I": I, "E": E,
    }
    return namespace


def _run_code(code: str, namespace: Dict[str, Any]) -> Tuple[str, str]:
    """Исполняет код, возвращает (status, output).

    Значение последнего выражения печатается автоматически — модели постоянно
    пишут последней строкой просто `answer` и ждут, что увидят результат.
    """
    buffer = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return "ERROR", f"SyntaxError: {exc}"

    last_expr: Optional[ast.Expression] = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(body=tree.body.pop().value)
        ast.fix_missing_locations(last_expr)

    try:
        with redirect_stdout(buffer):
            if tree.body:
                exec(compile(tree, "<agent>", "exec"), namespace)
            if last_expr is not None:
                value = eval(compile(last_expr, "<agent>", "eval"), namespace)
                if value is not None:
                    print(repr(value))
    except Exception as exc:
        printed = buffer.getvalue()
        detail = f"{type(exc).__name__}: {exc}"
        return "ERROR", (printed + detail) if printed else detail

    return "SUCCESS", buffer.getvalue()


def _worker_loop(request_q: multiprocessing.Queue, response_q: multiprocessing.Queue) -> None:
    """Цикл воркера: живёт между вызовами, импортирует sympy/numpy один раз."""
    try:
        namespace = _build_globals()
    except Exception as exc:  # noqa: BLE001 — сообщаем родителю и выходим
        response_q.put(("ERROR", f"worker init failed: {type(exc).__name__}: {exc}"))
        return
    response_q.put(("READY", ""))

    while True:
        message = request_q.get()
        if message is None:
            return
        # Каждый вызов получает свежую копию пространства имён: иначе
        # переменные одного шага незаметно протекают в следующий и модель
        # получает результат, которого не ожидает.
        response_q.put(_run_code(message, dict(namespace)))


# ---------------------------------------------------------------------------
# Управление воркером (по одному на поток бенчмарка)
# ---------------------------------------------------------------------------
class _Worker:
    def __init__(self) -> None:
        self.process: Optional[multiprocessing.Process] = None
        self.request_q: Optional[multiprocessing.Queue] = None
        self.response_q: Optional[multiprocessing.Queue] = None

    def start(self) -> None:
        self.request_q = multiprocessing.Queue()
        self.response_q = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=_worker_loop,
            args=(self.request_q, self.response_q),
            daemon=True,
        )
        self.process.start()
        try:
            status, detail = self.response_q.get(timeout=120.0)
        except queue.Empty:
            self.kill()
            raise RuntimeError("worker did not become ready in 120s")
        if status != "READY":
            self.kill()
            raise RuntimeError(detail)

    def kill(self) -> None:
        process = self.process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                # terminate не сработал — раньше здесь терялся handle и процесс
                # оставался жить со всей своей памятью до конца прогона.
                process.kill()
                process.join(timeout=5.0)
        # Очереди держат feeder-поток и пару дескрипторов на каждую;
        # без явного закрытия они накапливались вместе с воркерами.
        for q in (self.request_q, self.response_q):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self.process = None
        self.request_q = None
        self.response_q = None

    def _wait_for_result(self, timeout: float) -> Tuple[str, str]:
        """Ждёт ответ, попутно следя за памятью воркера."""
        if psutil is None or self.process is None:
            assert self.response_q is not None
            return self.response_q.get(timeout=timeout)

        assert self.response_q is not None
        try:
            monitor = psutil.Process(self.process.pid)
        except psutil.Error:
            return self.response_q.get(timeout=timeout)

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                # Частый опрос: между двумя проверками numpy успевает выделить
                # очень много, а psutil-вызов стоит микросекунды.
                return self.response_q.get(timeout=min(0.1, remaining))
            except queue.Empty:
                pass
            try:
                used_mb = monitor.memory_info().rss / 1024 ** 2
            except psutil.Error:
                continue
            if used_mb > MEMORY_LIMIT_MB:
                self.kill()
                return (
                    "ERROR",
                    f"execution exceeded the {MEMORY_LIMIT_MB} MB memory limit - you are "
                    "materialising too large a collection. Iterate lazily or narrow the "
                    "search space instead of building a huge list.",
                )

    def ensure_alive(self) -> None:
        if self.process is None or not self.process.is_alive():
            self.kill()
            self.start()

    def execute(self, code: str, timeout: float) -> Tuple[str, str]:
        self.ensure_alive()
        assert self.request_q is not None
        self.request_q.put(code)
        try:
            return self._wait_for_result(timeout)
        except queue.Empty:
            self.kill()
            return (
                "ERROR",
                f"execution timed out after {timeout:.0f}s - likely brute force over "
                "too large a range. Narrow the search space or derive a formula.",
            )


# ---------------------------------------------------------------------------
# Пул воркеров
#
# Раньше воркер жил в threading.local. Сэмплы self-consistency бегут во
# внутреннем ThreadPoolExecutor, который пересоздаётся на каждую задачу, —
# значит каждая задача плодила новые потоки, каждый поток порождал свой
# процесс с sympy+numpy, и никто их не закрывал: +4 процесса и ~400 МБ на
# задачу, линейный рост до исчерпания памяти.
#
# Теперь воркеры переиспользуются между задачами и потоками, а их число
# ограничено сверху независимо от степени параллелизма.
# ---------------------------------------------------------------------------
_pool_lock = threading.Lock()
_idle_workers: List[_Worker] = []
_all_workers: Set[_Worker] = set()
_worker_slots = threading.Semaphore(MAX_SANDBOX_WORKERS)


def _acquire_worker() -> _Worker:
    """Берёт свободный воркер, при необходимости ждёт освобождения слота."""
    _worker_slots.acquire()
    try:
        with _pool_lock:
            if _idle_workers:
                return _idle_workers.pop()
            worker = _Worker()
            _all_workers.add(worker)
            return worker
    except BaseException:
        _worker_slots.release()
        raise


def _release_worker(worker: _Worker) -> None:
    """Возвращает воркер в пул. Мёртвый воркер поднимется лениво в ensure_alive."""
    with _pool_lock:
        _idle_workers.append(worker)
    _worker_slots.release()


def _get_repeat_state() -> Dict[str, Any]:
    state = getattr(_thread_state, "repeat", None)
    if state is None:
        state = {"last_code": None, "streak": 0}
        _thread_state.repeat = state
    return state


def reset_calculator_state() -> None:
    """Сбрасывает анти-зацикливание перед новой задачей.

    Состояние теперь per-thread, поэтому параллельные задачи бенчмарка больше
    не сбрасывают счётчики друг другу и не ловят чужие ложные срабатывания.
    """
    state = _get_repeat_state()
    state["last_code"] = None
    state["streak"] = 0


def shutdown_workers() -> None:
    """Останавливает ВСЕ процессы-песочницы. Вызывать в конце прогона.

    Раньше эта функция убивала только воркер вызывающего потока, а раннер
    вызывал её из внешнего потока — реальные воркеры сэмплов не закрывались
    никогда. Теперь она чистит весь пул.
    """
    with _pool_lock:
        workers = list(_all_workers)
        _all_workers.clear()
        _idle_workers.clear()
    for worker in workers:
        worker.kill()


# ---------------------------------------------------------------------------
# Сам инструмент
# ---------------------------------------------------------------------------
@tool
def python_exec(code: str) -> str:
    """Execute Python code for exact mathematics and print the result.

    Preloaded: sympy (solve, Eq, Rational, symbols, factorint, isprime,
    divisors, binomial, simplify, Matrix, ...), numpy as np, itertools,
    math, Fraction. Sympy names shadow math, so sqrt(8) stays exact.

    Write multi-line code and print() what you need; the value of the final
    expression is printed automatically. Prefer exact types (Rational, sqrt)
    over floats. Execution is capped at 10 seconds, so derive a formula or
    narrow the range instead of brute-forcing millions of cases.

    Example:
        n = symbols('n', integer=True, positive=True)
        sols = solve(Eq(n**2 - 5*n + 6, 0), n)
        print(sols)
    """
    clean_code = (code or "").strip()
    if not clean_code:
        return "ERROR: empty code."

    state = _get_repeat_state()
    if state["last_code"] == clean_code:
        state["streak"] += 1
    else:
        state["last_code"] = clean_code
        state["streak"] = 1

    if state["streak"] >= REPEAT_LIMIT:
        return (
            "ERROR: you have run this exact code several times in a row. "
            "Stop calling the tool and use the result you already have."
        )

    try:
        worker = _acquire_worker()
    except Exception as exc:  # noqa: BLE001 — тул не должен ронять граф
        return f"ERROR: sandbox unavailable ({type(exc).__name__}: {exc})"

    try:
        status, output = worker.execute(clean_code, EXEC_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: sandbox failure ({type(exc).__name__}: {exc})"
    finally:
        _release_worker(worker)

    text = (output or "").strip()
    if status == "ERROR":
        return f"ERROR: {text}"
    if not text:
        return "(no output — nothing was printed and the last line was not an expression)"
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [output truncated at {MAX_OUTPUT_CHARS} chars]"
    return text


# Прежнее имя инструмента: старые промпты и логи ссылаются на `calculator`.
calculator = python_exec
