#!/usr/bin/env bash
# Сквозная проверка очереди на временном репозитории.
#
# Поднимает bare-remote, ветку студента, спеку в main и прогоняет два тика
# диспетчера: первый запускает задание, второй подбирает его и публикует
# результаты. Ничего в рабочем репозитории не меняет.
#
#   scripts/jobs/e2e_check.sh
#
# Требует git, python с PyYAML и пустой каталог под $TMPDIR.

set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${MATHAGENT_PYTHON:-$SOURCE_ROOT/.venv/bin/python}"
SANDBOX="${TMPDIR:-/tmp}/mathagent-e2e"

rm -rf "$SANDBOX"
mkdir -p "$SANDBOX"/{src,home/gpu-slots}

export MATHAGENT_HOME="$SANDBOX/home"
# Песочница требует bwrap, которого может не быть на машине разработчика.
export MATHAGENT_SANDBOX=0
export GIT_AUTHOR_NAME=e2e GIT_AUTHOR_EMAIL=e2e@example.com
export GIT_COMMITTER_NAME=e2e GIT_COMMITTER_EMAIL=e2e@example.com

# --- исходное дерево тестового репозитория -----------------------------------

mkdir -p "$SANDBOX/src/scripts/jobs" "$SANDBOX/src/scripts/benchmarks" "$SANDBOX/src/jobs/queue"
cp "$SOURCE_ROOT"/scripts/jobs/*.py "$SANDBOX/src/scripts/jobs/"
cp "$SOURCE_ROOT"/scripts/benchmarks/run_artifacts.py "$SANDBOX/src/scripts/benchmarks/"
printf 'results/**/*.jsonl\n' > "$SANDBOX/src/.gitignore"

cat > "$SANDBOX/src/scripts/benchmarks/fake_bench.py" <<'PY'
"""Учебный бенчмарк: пишет пару файлов результатов и завершается."""
import json, os, sys, time
from pathlib import Path

run_id = os.environ.get("MATHAGENT_RUN_ID", "unknown")
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 1
sleep_for = float(sys.argv[sys.argv.index("--sleep") + 1]) if "--sleep" in sys.argv else 0.0

print(f"run_id={run_id} devices={os.environ.get('CUDA_VISIBLE_DEVICES')!r} limit={limit}")
print(f"HOME={os.environ['HOME']} keys={sorted(os.environ)}")
time.sleep(sleep_for)

out = Path("results") / "runs" / run_id
out.mkdir(parents=True, exist_ok=True)
(out / "fake.jsonl").write_text("".join(json.dumps({"task_id": i}) + "\n" for i in range(limit)))
(out / "manifest.json").write_text(json.dumps({"run_id": run_id, "status": "done"}))
print("done")
PY

cat > "$SANDBOX/src/scripts/benchmarks/fake_daemon.py" <<'PY'
"""Учебный «сервер»: уходит в собственную сессию и не завершается сам.

Нужен, чтобы проверить уборку процессов после таймаута — именно так ведёт себя
забытый vLLM, продолжающий держать карту.
"""
import os, sys, time

if os.fork() > 0:
    sys.exit(0)
os.setsid()           # намеренно покидаем группу процессов запуска
while True:
    time.sleep(1)
PY

git -C "$SANDBOX/src" init --quiet --initial-branch=main
git -C "$SANDBOX/src" add -A
git -C "$SANDBOX/src" commit --quiet -m "initial"

git init --quiet --bare "$SANDBOX/origin.git"
git -C "$SANDBOX/src" remote add origin "$SANDBOX/origin.git"
git -C "$SANDBOX/src" push --quiet origin main

# --- ветка студента ----------------------------------------------------------

git -C "$SANDBOX/src" checkout --quiet -b Ilya/demo
printf '# ветка студента\n' >> "$SANDBOX/src/scripts/benchmarks/fake_bench.py"
git -C "$SANDBOX/src" commit --quiet -am "student tweak"
git -C "$SANDBOX/src" push --quiet origin Ilya/demo
git -C "$SANDBOX/src" checkout --quiet main

# --- спеки в очереди ---------------------------------------------------------

write_spec() {
    cat > "$SANDBOX/src/jobs/queue/$1.yml" <<YAML
branch: Ilya/demo
command: $2
gpus: $3
timeout_minutes: $4
YAML
}

write_spec ilya-first  "python scripts/benchmarks/fake_bench.py --limit 3" 1 5
write_spec ilya-second "python scripts/benchmarks/fake_bench.py --limit 2" 1 5
write_spec ilya-third  "python scripts/benchmarks/fake_bench.py --limit 1" 2 5
git -C "$SANDBOX/src" add -A && git -C "$SANDBOX/src" commit --quiet -m "queue: three specs"
git -C "$SANDBOX/src" push --quiet origin main

# --- рабочая копия диспетчера ------------------------------------------------

git clone --quiet "$SANDBOX/origin.git" "$MATHAGENT_HOME/repo"
touch "$MATHAGENT_HOME/gpu-slots/gpu0.lock" "$MATHAGENT_HOME/gpu-slots/gpu1.lock"

dispatcher() { "$PYTHON" "$SOURCE_ROOT/scripts/jobs/dispatcher.py" "$@"; }

echo "=== тик 1: раздача карт ==="
dispatcher

echo
echo "=== состояние сразу после запуска ==="
dispatcher --status

echo
echo "=== ждём завершения запусков ==="
sleep 6

echo
echo "=== тик 2: подбор и публикация ==="
dispatcher

echo
echo "=== итоговое состояние ==="
dispatcher --status

echo
echo "=== ветка студента после публикации ==="
git -C "$MATHAGENT_HOME/repo" fetch --quiet origin
git -C "$MATHAGENT_HOME/repo" log --oneline origin/Ilya/demo | head -5
echo "--- опубликованные файлы:"
git -C "$MATHAGENT_HOME/repo" ls-tree -r --name-only origin/Ilya/demo -- results | head -20

echo
echo "=== свободны ли карты ==="
dispatcher --status | head -4

echo
echo "SANDBOX=$SANDBOX"
