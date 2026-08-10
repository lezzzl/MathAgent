#!/usr/bin/env bash
# Обёртка тика диспетчера для cron.
#
# Cron даёт почти пустое окружение, поэтому интерпретатор и MATHAGENT_HOME
# задаются здесь явно. Сам тик короткий: он отсоединяет запуски и выходит.
#
# Строка crontab (см. jobs/RUNNER.md):
#   * * * * * /usr/bin/flock -n $HOME/mathagent/dispatcher.lock \
#       $HOME/mathagent/repo/scripts/jobs/dispatch.sh >> $HOME/mathagent/logs/dispatcher.log 2>&1

set -euo pipefail

MATHAGENT_HOME="${MATHAGENT_HOME:-$HOME/mathagent}"
export MATHAGENT_HOME

REPO="${MATHAGENT_REPO:-$MATHAGENT_HOME/repo}"
PYTHON="${MATHAGENT_PYTHON:-$MATHAGENT_HOME/venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "python interpreter not found at $PYTHON — see jobs/RUNNER.md" >&2
    exit 1
fi

exec "$PYTHON" "$REPO/scripts/jobs/dispatcher.py" "$@"
