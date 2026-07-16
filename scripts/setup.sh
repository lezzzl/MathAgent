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

echo "==> Проверка доступа к Yandex AI Studio"
# Секреты НЕ хранятся в репозитории. Держим их в .env (в .gitignore),
# конфиг ссылается лишь на имена переменных YC_API_KEY / YC_FOLDER_ID.
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'DOTENV'
# Секреты — этот файл НЕ коммитится. Впиши свои значения:
YC_API_KEY=
YC_FOLDER_ID=
DOTENV
  echo "    создан $ENV_FILE — впиши YC_API_KEY и YC_FOLDER_ID"
else
  echo "    $ENV_FILE уже есть"
fi
# Подстраховка: убедимся, что .env игнорируется гитом.
if ! git check-ignore -q "$ENV_FILE" 2>/dev/null; then
  echo ".env" >> .gitignore
  echo "    добавил .env в .gitignore"
fi

echo ""
echo "Готово. Дальше:"
echo "  # впиши YC_API_KEY и YC_FOLDER_ID в $ENV_FILE"
echo "  uv run kedro run --env experiments/baseline    # запустить эксперимент"
