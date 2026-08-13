#!/usr/bin/env bash
# Общее у обоих входов в прогон (run_benchmarks.sh и run_benchmarks_two_models.sh):
# чтение флагов из команды бенчмарков, подстановка плейсхолдеров и публикация
# результатов. Правила вроде «коммитим только свой каталог» должны существовать
# в одном месте, иначе скрипты однажды разъедутся.
#
# Файл только определяет функции (source, не запуск) и ничего не выполняет сам.
# Все функции работают с массивом BENCH_CMD — командой бенчмарков после `--`.

# Плейсхолдер run_id: команда не может знать его заранее (в спеке диспетчера
# run_id — это имя файла), поэтому пишет его так, а подставляем мы.
RUN_ID_PLACEHOLDER='{RUN_ID}'

# --- порты -------------------------------------------------------------------
# На машине идёт несколько прогонов сразу, и порт — такой же общий ресурс, как
# карта. Занятый порт опаснее всего тем, что он не мешает: движок падает на
# bind, а проверка готовности видит ЧУЖОЙ сервер, отвечающий по тому же адресу,
# — и прогон честно меряет чужую модель с чужими настройками. Поэтому порт
# проверяется дважды: свободен ли он до старта и наш ли сервер на нём после.

# port_listening <порт> — слушает ли уже кто-то этот порт на localhost.
# /dev/tcp — встроенный в bash, никаких ss/lsof/nc в песочнице не требуется.
port_listening() {
  local port="$1"
  (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null || return 1
  exec 3>&-
  return 0
}

# require_free_port <порт> <роль> — отказ, если порт уже занят.
require_free_port() {
  local port="$1" role="${2:-сервер}" holder=""
  port_listening "${port}" || return 0

  # Имя владельца — подсказка, а не проверка: чужой процесс может быть и не наш,
  # тогда ss про него ничего не скажет.
  holder="$(ss -ltnp 2>/dev/null | grep -E "[:.]${port}\b" | head -n 1 || true)"
  echo "Порт ${port} (${role}) уже занят — на нём кто-то слушает." >&2
  [[ -n "${holder}" ]] && echo "  ${holder}" >&2
  echo "  Порт прогону выдаёт диспетчер (PORT/SECOND_PORT); при ручном запуске" >&2
  echo "  возьмите свободный: PORT=8200 $0 ..." >&2
  exit 1
}

# require_served_model <base_url> <имя модели> <роль> — на порту наш сервер?
#
# Проверка от подмены: чужой движок на том же порту отвечает на /models так же
# бодро, и без сверки имени прогон уедет на нём до конца, дав правдоподобные и
# неверные числа.
require_served_model() {
  local url="$1" expected="$2" role="${3:-сервер}" listing available
  listing="$(curl -fsS --max-time 10 "${url}/models" 2>/dev/null || true)"
  if [[ -z "${listing}" ]]; then
    echo "Сервер ${role} на ${url} не ответил на /models" >&2
    exit 1
  fi

  available="$(printf '%s' "${listing}" \
    | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | sed -E 's/.*"([^"]*)"$/\1/' || true)"
  if printf '%s\n' "${available}" | grep -qxF "${expected}"; then
    return 0
  fi

  echo "Сервер ${role} на ${url} отдаёт не нашу модель." >&2
  echo "  ожидали: ${expected}" >&2
  echo "  сервер отдаёт: ${available//$'\n'/, }" >&2
  echo "  Скорее всего это чужой прогон на том же порту." >&2
  exit 1
}

# require_no_hardcoded_address — отказ, если команда сама называет адрес сервера.
#
# Проверки выше стерегут порт, на котором мы подняли движок. Команда с зашитым
# `127.0.0.1:8137` их обходит: наш сервер поднят и здоров, а запросы уходят на
# другой порт — к соседнему прогону либо в пустоту. Вызывать ДО подстановки
# плейсхолдеров, иначе под правило попадут наши же подставленные адреса.
require_no_hardcoded_address() {
  local item address=""
  for item in "${BENCH_CMD[@]}"; do
    if [[ "${item}" =~ (127\.0\.0\.1|localhost|0\.0\.0\.0):[0-9]{2,5} ]]; then
      address="${BASH_REMATCH[0]}"
      break
    fi
    # Дефис перед port обязателен: иначе сюда попадёт любой флаг, чьё имя просто
    # кончается на эти буквы (--report, --transport).
    if [[ "${item}" =~ ^--([A-Za-z0-9-]*-)?port(=|$) ]]; then
      address="${item}"
      break
    fi
  done
  [[ -n "${address}" ]] || return 0

  echo "В команде зашит адрес сервера: ${address}" >&2
  echo "  Порт выдаёт диспетчер, и заранее он неизвестен — на этом адресе" >&2
  echo "  окажется чужой прогон или никто." >&2
  echo "  Напишите плейсхолдер: --base-url {BASE_URL} (второй сервер —" >&2
  echo "  {SECOND_BASE_URL}); свой скрипт может взять порт из \$PORT." >&2
  exit 1
}

# bench_cmd_opt <флаг> — значение флага в команде бенчмарков.
#
# Повторённый флаг: побеждает последний — так же его прочтёт argparse самой
# команды. Плейсхолдер значением не считается: на этом этапе он ещё не
# подставлен, а вернуть его наружу значит выдать '{RUN_ID}' за настоящий run_id.
bench_cmd_opt() {
  local flag="$1" value="" i
  for ((i = 0; i < ${#BENCH_CMD[@]}; i++)); do
    case "${BENCH_CMD[i]}" in
      "${flag}") value="${BENCH_CMD[i + 1]:-}" ;;
      "${flag}="*) value="${BENCH_CMD[i]#"${flag}"=}" ;;
    esac
  done
  [[ "${value}" == "{"*"}" ]] && value=""
  printf '%s' "${value}"
}

# substitute_placeholders <ИМЯ=значение>... — подставляет {ИМЯ} в BENCH_CMD.
#
# Адреса серверов и run_id известны только здесь, а команда задаётся заранее и
# без shell (в спеке диспетчера `$(...)` запрещён) — плейсхолдеры и есть
# единственный способ передать ей эти значения.
substitute_placeholders() {
  local pairs=("$@") pair name value i
  for ((i = 0; i < ${#BENCH_CMD[@]}; i++)); do
    for pair in "${pairs[@]}"; do
      name="${pair%%=*}"
      value="${pair#*=}"
      BENCH_CMD[i]="${BENCH_CMD[i]//\{${name}\}/${value}}"
    done
  done
}

# publish_results <results_dir> <run_id> <детали для сообщения коммита>
#
# Поведение задаётся переменной PUSH_RESULTS:
#   1     — коммит и push (по умолчанию у ручных прогонов);
#   0     — только локальный коммит;
#   none  — не коммитить вообще (так запускает диспетчер: он публикует сам).
publish_results() {
  local results_dir="$1" run_id="$2" detail="${3:-}" branch head
  local push="${PUSH_RESULTS:-1}"

  [[ "${push}" == "none" ]] && { echo ">>> PUSH_RESULTS=none — результаты не коммичу"; return 0; }
  if [[ ! -d "${results_dir}" ]]; then
    echo ">>> ${results_dir} не создан — коммитить нечего" >&2
    return 0
  fi

  branch="$(git symbolic-ref --quiet --short HEAD || true)"
  if [[ -z "${branch}" ]]; then
    echo ">>> HEAD отделён от ветки — результаты оставляю незакоммиченными" >&2
    return 0
  fi

  # Пути указаны явно (никаких `git add -A`): незакоммиченные правки в рабочем
  # дереве и уже проиндексированные чужие изменения не должны попасть в коммит,
  # поэтому и `git commit` вызывается с pathspec.
  git add -- "${results_dir}"
  if git diff --cached --quiet -- "${results_dir}"; then
    echo ">>> В ${results_dir} нет изменений — коммит не нужен"
    return 0
  fi

  git commit --quiet -m "bench: ${run_id}${detail:+ (${detail})}" -- "${results_dir}"
  head="$(git rev-parse --short HEAD)"
  echo ">>> Закоммитил ${head}: ${results_dir}"

  [[ "${push}" == "1" ]] || { echo ">>> PUSH_RESULTS=${push} — пушить не буду"; return 0; }
  if ! git remote get-url origin >/dev/null 2>&1; then
    echo ">>> Нет remote origin — коммит остался локальным" >&2
    return 0
  fi

  # Каждая попытка — в условии if: под set -e неудача последней команды в
  # &&-цепочке убила бы скрипт, не дав напечатать подсказку про ручной пуш.
  echo ">>> Пушу в origin/${branch}"
  if git push origin "HEAD:${branch}"; then return 0; fi

  # Ветка уехала вперёд: результаты лежат в каталоге с уникальным run_id,
  # так что ребейз поверх origin конфликтовать не должен. Пробуем ровно раз.
  echo ">>> Push отклонён, делаю rebase на origin/${branch} и пробую ещё раз" >&2
  if git pull --rebase origin "${branch}" && git push origin "HEAD:${branch}"; then return 0; fi

  echo ">>> Push не удался. Коммит ${head} остался локальным — запушь вручную:" >&2
  echo "    git push origin HEAD:${branch}" >&2
  return 0
}
