# Диспетчер очереди на VM

Установка и эксплуатация. Всё живёт в домашнем каталоге вашей обычной учётной
записи: ни root, ни sudo, ни системных юнитов — только пользовательский
`crontab`.

## Как это устроено

```
студент пушит ветку Ilya/<тема>
   → PR в main с jobs/queue/<run_id>.yml
   → вы смотрите диф и вливаете
   → тик cron раз в минуту:
        git fetch → подбор завершившихся → очередь FIFO → аренда GPU → запуск
   → по завершении: коммит results/ в ветку студента, push, комментарий в PR
```

| Файл | Роль |
| --- | --- |
| `scripts/jobs/job_spec.py` | Разбор спеки и проверка команды. |
| `scripts/jobs/gpu_lease.py` | Бюджет карт через `flock`. |
| `scripts/jobs/sandbox.py` | Изоляция команды через bubblewrap. |
| `scripts/jobs/dispatcher.py` | Тик cron: подбор, очередь, запуск. |
| `scripts/jobs/run_job.py` | Один запуск: worktree, окружение, таймаут, уборка. |
| `scripts/jobs/publish.py` | Push результатов и комментарий в PR. |

**Автоматической проверки PR нет — её роль выполняете вы при ревью.** Смотрите
на диф: он должен содержать только новый файл в `jobs/queue/`. PR, который
трогает `scripts/jobs/**`, меняет код самого диспетчера, а тот выполняется от
вашего имени и вне песочницы.

## Установка

```bash
mkdir -p ~/mathagent/{gpu-slots,state,work,logs,cache}
chmod 700 ~/mathagent
git clone git@github.com:lezzzl/MathAgent.git ~/mathagent/repo
python3 -m venv ~/mathagent/venv
~/mathagent/venv/bin/pip install PyYAML
sudo apt install bubblewrap        # единственное, что требует root, и только раз
```

**Слоты GPU. Сколько файлов — столько карт разрешено занять; другого места, где
задаётся бюджет, нет.** Номер в имени попадает в `CUDA_VISIBLE_DEVICES`:

```bash
touch ~/mathagent/gpu-slots/gpu0.lock ~/mathagent/gpu-slots/gpu1.lock
```

**Доступ к GitHub** — ваш обычный: ssh-ключ для push и `gh auth login` для
комментариев в PR. Отдельных учётных данных заводить не нужно.

**Проверьте песочницу.** Она включена по умолчанию, и если bwrap не работает,
запуски будут падать — молча выполнять команду без изоляции нельзя:

```bash
~/mathagent/venv/bin/python -c "import sys; sys.path.insert(0,'$HOME/mathagent/repo'); from scripts.jobs import sandbox; print(sandbox.probe())"
```

Ожидается `(True, 'bubblewrap works')`. Если `False` (в Ubuntu 24.04
непривилегированные user namespaces ограничивает AppArmor) — почините или
выключите изоляцию осознанно: `touch ~/mathagent/sandbox.disabled`. Запишите
результат сюда, чтобы не выяснять заново:

> Состояние на этой машине: _(заполнить после установки)_

**Cron:**

```bash
crontab -e
```

```cron
* * * * * /usr/bin/flock -n $HOME/mathagent/dispatcher.lock $HOME/mathagent/repo/scripts/jobs/dispatch.sh >> $HOME/mathagent/logs/dispatcher.log 2>&1
```

`flock -n` гарантирует, что тики не пересекаются. Тик короткий: он отсоединяет
запуски и сразу выходит, поэтому часовой бенчмарк не блокирует очередь.

**Новый человек на курсе:** добавьте префикс его ветки в `BRANCH_PREFIXES`
в [`scripts/jobs/job_spec.py`](../scripts/jobs/job_spec.py).

## Что даёт песочница

Команда студента выполняется под `bwrap`:

- корень примонтирован **только на чтение**, записывать можно лишь в рабочую
  копию запуска и общие кэши (`HF_HOME`, `UV_CACHE_DIR`);
- `~/.ssh`, `~/.config/gh` и `~/.git-credentials` подменены пустым tmpfs —
  именно они дают возможность действовать от вашего имени;
- `/dev` проброшен, поэтому карты доступны;
- `--die-with-parent`: потомки не переживают запуск.

Окружение команды собирается с нуля и содержит ровно `PATH`, `HOME`, `TMPDIR`,
`LANG`, `CUDA_VISIBLE_DEVICES`, `HF_HOME`, `UV_CACHE_DIR`, `MATHAGENT_RUN_ID` —
ни токена, ни `GIT_SSH_COMMAND`. Режим изоляции пишется в первую строку каждого
`runner.log` (`isolation=bwrap` или `isolation=none`).

Чего песочница не делает: не ограничивает CPU, память и диск, и не мешает читать
всё, что доступно на чтение в системе.

## Эксплуатация

```bash
# что идёт и что заняло карты
~/mathagent/repo/scripts/jobs/dispatch.sh --status

# лог диспетчера и лог конкретного запуска
tail -f ~/mathagent/logs/dispatcher.log
tail -f ~/mathagent/logs/<run_id>.log

# перезапустить задачу заново
rm ~/mathagent/state/<run_id>.json

# отменить идущую задачу: следующий тик подберёт её и опубликует как failed
kill $(python3 -c "import json;print(json.load(open('$HOME/mathagent/state/<run_id>.json'))['pid'])")
```

Состояния запуска: `running` → `done` / `failed` / `timeout` / `orphaned`
(процесс исчез, не записав итог) или `invalid` (спека не прошла проверку — тогда
причина лежит в поле `error`).

Лог диспетчера обрезается им самим при превышении 10 МиБ; системный logrotate
не нужен.

## Проверка после установки

```bash
# сквозной прогон на временном репозитории, рабочие данные не трогает
scripts/jobs/e2e_check.sh

# юнит-тесты (на Linux дополнительно проверяют уборку процессов)
~/mathagent/venv/bin/python -m pytest tests/jobs -q
```

Ручная проверка на реальных картах:

1. спека с `command: bash scripts/qwen35-vllm-bench/serve.sh`, `gpus: 1`,
   `timeout_minutes: 1` — через минуту запуск должен быть убит, а `nvidia-smi`
   чист. Это главный сценарий: забытый vLLM держал бы карту;
2. три спеки сразу с `gpus: 1, 1, 2` — первые две идут параллельно, третья ждёт;
   `nvidia-smi` никогда не показывает больше двух занятых карт;
3. спека с командой, которая пробует прочитать `~/.ssh` — должна получить пустой
   каталог.
