#!/usr/bin/env bash
#
# Установка окружения MathAgent через uv.
# Запуск из корня репозитория:  ./scripts/setup.sh
#
set -euo pipefail

# Перейти в корень проекта (на уровень выше scripts/), чтобы скрипт можно было
# запускать из любого места.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Проверка uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "    uv не найден — устанавливаю"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Инсталлятор кладёт uv в ~/.local/bin — добавим в PATH для текущей сессии.
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "==> Синхронизация зависимостей (uv sync)"
# Создаёт .venv и ставит всё из pyproject.toml, фиксируя версии в uv.lock.
uv sync

echo "==> Проверка credentials"
CRED="conf/local/credentials.yml"
if [ ! -s "$CRED" ]; then
  mkdir -p "$(dirname "$CRED")"
  cat > "$CRED" <<'YAML'
# Секреты (этот файл не коммитится). Впиши ключ Anthropic:
anthropic:
  api_key: sk-ant-...
YAML
  echo "    создан шаблон $CRED — не забудь вписать api_key"
else
  echo "    $CRED уже заполнен"
fi

echo ""
echo "Готово. Дальше:"
echo "  # вписать ключ в $CRED"
echo "  uv run kedro run --pipeline agent_eval    # запустить пайплайн"
