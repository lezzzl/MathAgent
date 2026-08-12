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
