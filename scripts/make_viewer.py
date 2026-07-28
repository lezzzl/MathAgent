"""Собирает самодостаточный HTML-просмотрщик траекторий решения.

    python scripts/make_viewer.py results/aime26/<run>_trajectory.json
    python scripts/make_viewer.py <trajectory.json> -o viewer.html

На вход — JSON от trajectory.RECORDER (флаг --trajectory у бенчмарк-раннера).
На выходе — один HTML-файл со встроенными данными: открывается двойным щелчком,
без сервера и без интернета. Работает для обоих пайплайнов (qwen4b и 9B): они
пишут одинаковые записи, различаясь только набором стадий.

Что показывает: дерево «шаг -> ветка -> стадия», полный промпт и сырой ответ
модели, оценку и обоснование оценщика, вердикт верификатора, а также сводку по
токенам, времени, стоимости и обрывам ответов.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


CSS = """
*{box-sizing:border-box}
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#12141a; --muted:#666e7d; --line:#e3e6ec;
  --accent:#3b6ef5; --ok:#12876f; --bad:#c8102e; --warn:#b5651d;
  --gen:#3b6ef5; --seg:#7c4dff; --eval:#12876f; --commit:#b5651d; --verify:#c8102e;
  --code:#f2f3f7;
}
:root[data-theme=dark],
html[data-theme=dark]{
  --bg:#0f1115; --panel:#171a21; --ink:#e8eaf0; --muted:#9aa3b2; --line:#262b36;
  --accent:#6f9bff; --ok:#3ddbb4; --bad:#ff6b81; --warn:#ffb454;
  --gen:#6f9bff; --seg:#b39dff; --eval:#3ddbb4; --commit:#ffb454; --verify:#ff6b81;
  --code:#11141a;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    --bg:#0f1115; --panel:#171a21; --ink:#e8eaf0; --muted:#9aa3b2; --line:#262b36;
    --accent:#6f9bff; --ok:#3ddbb4; --bad:#ff6b81; --warn:#ffb454;
    --gen:#6f9bff; --seg:#b39dff; --eval:#3ddbb4; --commit:#ffb454; --verify:#ff6b81;
    --code:#11141a;
  }
}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;
  padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel);
  position:sticky;top:0;z-index:20;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:650}
.sub{color:var(--muted);font-size:12px;margin-top:2px}
button{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);
  border-radius:8px;padding:6px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}
.wrap{display:grid;grid-template-columns:300px minmax(0,1fr);gap:16px;
  padding:16px;align-items:start;max-width:1600px;margin:0 auto}
@media(max-width:980px){.wrap{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}
.side{position:sticky;top:70px;max-height:calc(100vh - 90px);display:flex;flex-direction:column}
.side .hd{padding:12px 14px;border-bottom:1px solid var(--line);font-weight:600}
.tasklist{overflow:auto;padding:6px}
.task{display:flex;justify-content:space-between;gap:8px;align-items:center;
  padding:8px 10px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.task:hover{background:var(--code)}
.task.active{border-color:var(--accent);background:var(--code)}
.task .tid{font-weight:600}
.task .meta{color:var(--muted);font-size:11px}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}.dot.none{background:var(--muted)}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:14px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.metric .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.metric .v{font-size:18px;font-weight:650;margin-top:3px}
.metric .v.small{font-size:14px;font-weight:600}
.problem{padding:12px 14px;margin-bottom:14px;white-space:pre-wrap;
  border-left:3px solid var(--accent)}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
input[type=search]{font:inherit;color:inherit;background:var(--panel);
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;min-width:200px;flex:1}
.chip{border:1px solid var(--line);border-radius:999px;padding:4px 10px;font-size:12px;
  cursor:pointer;background:var(--panel);user-select:none}
.chip.off{opacity:.4}
.step{margin-bottom:14px}
.step-hd{display:flex;align-items:center;gap:10px;padding:9px 13px;font-weight:600;
  border-bottom:1px solid var(--line);cursor:pointer}
.step-hd .n{color:var(--muted);font-weight:500;font-size:12px}
.branch{border-top:1px dashed var(--line);padding:10px 13px}
.branch:first-child{border-top:none}
.branch-hd{display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:12px;color:var(--muted)}
.rec{border:1px solid var(--line);border-radius:9px;margin:7px 0;overflow:hidden}
.rec-hd{display:flex;align-items:center;gap:9px;padding:7px 11px;cursor:pointer;flex-wrap:wrap}
.rec-hd:hover{background:var(--code)}
.tag{font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;color:#fff;letter-spacing:.02em}
.tag.generate{background:var(--gen)}.tag.segment{background:var(--seg)}
.tag.segment_result{background:var(--seg)}.tag.evaluate{background:var(--eval)}
.tag.evaluate_result{background:var(--eval)}.tag.commit{background:var(--commit)}
.tag.verify{background:var(--verify)}.tag.verify_result{background:var(--verify)}
.pill{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.pill.ok{color:var(--ok);border-color:var(--ok)}
.pill.bad{color:var(--bad);border-color:var(--bad)}
.pill.warn{color:var(--warn);border-color:var(--warn)}
.rec-body{display:none;border-top:1px solid var(--line);padding:10px 12px}
.rec.open .rec-body{display:block}
.sec{margin:9px 0}
.sec>.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  margin-bottom:4px;cursor:pointer;user-select:none}
pre{margin:0;background:var(--code);border:1px solid var(--line);border-radius:8px;
  padding:10px;white-space:pre-wrap;word-break:break-word;overflow-x:auto;
  font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:460px;overflow-y:auto}
pre.collapsed{max-height:110px}
.hidden{display:none!important}
.empty{color:var(--muted);padding:24px;text-align:center}
mark{background:var(--warn);color:#000;border-radius:2px}
"""

JS = r"""
const DATA = JSON.parse(document.getElementById("data").textContent);
const $ = (s,r=document)=>r.querySelector(s);
const esc = s => (s??"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmtNum = n => (n==null?"—":n.toLocaleString("en-US"));
const fmtSec = s => s==null?"—":(s<60?s.toFixed(1)+"s":Math.floor(s/60)+"m "+Math.round(s%60)+"s");

/* Тема: системная по умолчанию, ручной выбор запоминается. */
const themeBtn=$("#theme");
function applyTheme(t){document.documentElement.setAttribute("data-theme",t);
  try{localStorage.setItem("tv-theme",t)}catch(e){}
  themeBtn.textContent = t==="dark"?"☀ Light":"☾ Dark";}
let saved=null; try{saved=localStorage.getItem("tv-theme")}catch(e){}
applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"));
themeBtn.onclick=()=>applyTheme(document.documentElement.getAttribute("data-theme")==="dark"?"light":"dark");

/* ---- сводка по задаче ---- */
function stats(t){
  const r=t.records||[];
  const calls=r.filter(x=>x.tokens&&x.tokens.total>0);
  const sum=(f)=>calls.reduce((a,x)=>a+(f(x)||0),0);
  return {
    calls: calls.length,
    tokens: sum(x=>x.tokens.total),
    cost: sum(x=>x.cost),
    llmTime: sum(x=>x.elapsed),
    truncated: r.filter(x=>x.truncated).length,
    truncEmpty: r.filter(x=>x.truncated_without_content).length,
    apiErr: r.filter(x=>x.error).length,
    steps: new Set(r.filter(x=>x.stage==="commit").map(x=>x.depth)).size,
  };
}
function verdict(t){
  if(t.final_answer==null||t.final_answer==="") return {cls:"none",txt:"нет ответа"};
  const gt=(t.ground_truth??"").toString().trim();
  const a=t.final_answer.toString().trim();
  if(!gt) return {cls:"none",txt:a};
  return a===gt ? {cls:"ok",txt:a+" ✓"} : {cls:"bad",txt:a+" ≠ "+gt};
}

/* ---- список задач ---- */
const listEl=$("#tasks");
DATA.tasks.forEach((t,i)=>{
  const v=verdict(t), s=stats(t);
  const el=document.createElement("div");
  el.className="task"; el.dataset.i=i;
  el.innerHTML=`<span class="dot ${v.cls}"></span>
    <span style="flex:1;min-width:0">
      <div class="tid">Task ${esc(t.task_id)}</div>
      <div class="meta">${s.steps} шаг(ов) · ${fmtNum(s.tokens)} ток. ${s.truncated?"· ⚠ обрыв":""}</div>
    </span>`;
  el.onclick=()=>select(i);
  listEl.appendChild(el);
});

/* ---- рендер одной задачи ---- */
let current=0;
function select(i){
  current=i;
  [...listEl.children].forEach((c,j)=>c.classList.toggle("active",j===i));
  const t=DATA.tasks[i], s=stats(t), v=verdict(t);

  $("#metrics").innerHTML=[
    ["Ответ", v.txt, v.cls==="ok"?"var(--ok)":v.cls==="bad"?"var(--bad)":"var(--muted)", true],
    ["Шагов", s.steps],
    ["Вызовов LLM", s.calls],
    ["Токенов", fmtNum(s.tokens)],
    ["Время задачи", fmtSec(t.elapsed)],
    ["Время в LLM", fmtSec(s.llmTime)],
    ["Стоимость", s.cost?("$"+s.cost.toFixed(4)):"—"],
    ["Обрывы", s.truncated + (s.truncEmpty?` (${s.truncEmpty} пустых)`:""),
      s.truncated?"var(--warn)":null],
    ["Сетевых сбоев", s.apiErr, s.apiErr?"var(--bad)":null],
  ].map(([k,val,color,small])=>`<div class="metric"><div class="k">${k}</div>
      <div class="v ${small?"small":""}" ${color?`style="color:${color}"`:""}>${esc(String(val))}</div></div>`).join("");

  $("#problem").textContent=t.problem||"";
  const gu=(t.metrics||{}).gave_up_reason;
  $("#gaveup").innerHTML = gu ? `<div class="panel problem" style="border-left-color:var(--bad)">
      <b>Сдался:</b> ${esc(gu)}</div>` : "";

  /* группируем записи по шагу (depth), внутри — по ветке */
  const byDepth=new Map();
  (t.records||[]).forEach(r=>{
    const d=r.depth==null?-1:r.depth;
    if(!byDepth.has(d)) byDepth.set(d,new Map());
    const b=r.branch==null?0:r.branch;
    const m=byDepth.get(d);
    if(!m.has(b)) m.set(b,[]);
    m.get(b).push(r);
  });

  const out=[];
  [...byDepth.keys()].sort((a,b)=>a-b).forEach(d=>{
    const branches=byDepth.get(d);
    const commit=(t.records||[]).find(r=>r.stage==="commit"&&r.depth===d);
    const nb=[...branches.keys()].filter(b=>b>0).length;
    out.push(`<section class="panel step" data-depth="${d}">
      <div class="step-hd" onclick="this.parentNode.classList.toggle('folded')">
        <span>${d<0?"Прочее":"Шаг "+(d+1)}</span>
        <span class="n">${nb?nb+" ветк(и)":""}</span>
        ${commit?`<span class="pill ${commit.score>=0.8?"ok":"bad"}">принята ветка ${commit.branch} · score ${commit.score}</span>`:""}
        ${commit&&commit.premature?`<span class="pill warn">ответ отклонён как преждевременный</span>`:""}
      </div>
      ${[...branches.keys()].sort((a,b)=>a-b).map(b=>renderBranch(b,branches.get(b))).join("")}
    </section>`);
  });
  $("#steps").innerHTML=out.join("") || `<div class="panel empty">Нет записей траектории</div>`;
  applyFilters();
}

function renderBranch(b,recs){
  return `<div class="branch" data-branch="${b}">
    ${b>0?`<div class="branch-hd">Ветка ${b}</div>`:""}
    ${recs.map(renderRec).join("")}</div>`;
}

const STAGE_RU={generate:"Генератор",segment:"Сегментатор",segment_result:"Шаг после сегментации",
  evaluate:"Оценщик",evaluate_result:"Вердикт оценщика",commit:"Принятый шаг",
  verify:"Верификатор",verify_result:"Вердикт верификатора"};

function renderRec(r){
  const pills=[];
  if(r.score!=null) pills.push(`<span class="pill ${r.score>=0.8?"ok":"bad"}">score ${r.score}</span>`);
  if(r.is_valid!=null) pills.push(`<span class="pill ${r.is_valid?"ok":"bad"}">is_valid: ${r.is_valid}</span>`);
  if(r.truncated) pills.push(`<span class="pill warn">⚠ обрыв (finish=length)${r.truncated_without_content?", пусто":""}</span>`);
  if(r.error) pills.push(`<span class="pill bad">сетевой сбой</span>`);
  if(r.fast_path===true) pills.push(`<span class="pill">быстрый путь</span>`);
  if(r.fast_path===false) pills.push(`<span class="pill">через сегментатор</span>`);
  if(r.reliable===false) pills.push(`<span class="pill warn">не распарсилось</span>`);
  if(r.answer) pills.push(`<span class="pill ok">answer: ${esc(String(r.answer))}</span>`);
  if(r.premature) pills.push(`<span class="pill warn">преждевременный</span>`);
  if(r.no_progress) pills.push(`<span class="pill warn">нет прогресса</span>`);
  if(r.thinking_retry) pills.push(`<span class="pill warn">повтор без размышлений</span>`);
  if(r.tokens&&r.tokens.total) pills.push(`<span class="pill">${fmtNum(r.tokens.total)} ток.</span>`);
  if(r.elapsed) pills.push(`<span class="pill">${fmtSec(r.elapsed)}</span>`);

  const secs=[];
  const add=(lbl,blob,collapsed)=>{
    if(!blob||!blob.text) return;
    const note=blob.clipped?` (показано ${blob.text.length} из ${blob.full_chars} символов)`:"";
    secs.push(`<div class="sec"><div class="lbl" onclick="this.nextElementSibling.classList.toggle('collapsed')">
      ${lbl}${note} ▾</div><pre class="${collapsed?"collapsed":""}">${esc(blob.text)}</pre></div>`);
  };
  add("system-промпт", r.system, true);
  add("user-промпт", r.user, true);
  add("Размышления (reasoning)", r.reasoning, true);
  add(r.stage.endsWith("_result")?"Текст":"Ответ модели", r.content, false);
  if(r.step_text) add("Оцениваемый шаг", {text:r.step_text,clipped:false}, true);

  return `<div class="rec" data-stage="${r.stage}">
    <div class="rec-hd" onclick="this.parentNode.classList.toggle('open')">
      <span class="tag ${r.stage}">${STAGE_RU[r.stage]||r.stage}</span>${pills.join("")}
    </div>
    <div class="rec-body">${secs.join("")||"<div class='empty'>Пусто</div>"}</div></div>`;
}

/* ---- фильтры стадий + поиск ---- */
const OFF=new Set();
$("#chips").addEventListener("click",e=>{
  const c=e.target.closest(".chip"); if(!c) return;
  const st=c.dataset.stage;
  if(OFF.has(st)){OFF.delete(st);c.classList.remove("off")}else{OFF.add(st);c.classList.add("off")}
  applyFilters();
});
$("#q").addEventListener("input",applyFilters);
function applyFilters(){
  const q=$("#q").value.trim().toLowerCase();
  document.querySelectorAll(".rec").forEach(el=>{
    const stageOk=!OFF.has(el.dataset.stage);
    const textOk=!q||el.textContent.toLowerCase().includes(q);
    el.classList.toggle("hidden",!(stageOk&&textOk));
    if(q&&textOk) el.classList.add("open");
  });
  document.querySelectorAll(".branch").forEach(b=>{
    b.classList.toggle("hidden",!b.querySelector(".rec:not(.hidden)"));
  });
  document.querySelectorAll(".step").forEach(s=>{
    s.classList.toggle("hidden",!s.querySelector(".rec:not(.hidden)"));
  });
}
document.addEventListener("keydown",e=>{
  if(e.key==="/"&&document.activeElement!==$("#q")){e.preventDefault();$("#q").focus()}
});
select(0);
"""


def build_html(data: dict) -> str:
    run = data.get("run", {}) or {}
    tasks = data.get("tasks", []) or []

    # Сводка по прогону: считаем ровно то, что нужно для сравнения прогонов.
    n = len(tasks)
    correct = sum(
        1 for t in tasks
        if str(t.get("final_answer") or "").strip()
        and str(t.get("final_answer")).strip() == str(t.get("ground_truth") or "").strip()
    )
    recs = [r for t in tasks for r in (t.get("records") or [])]
    tokens = sum((r.get("tokens") or {}).get("total", 0) for r in recs)
    cost = sum(r.get("cost") or 0 for r in recs)
    trunc = sum(1 for r in recs if r.get("truncated"))
    calls = sum(1 for r in recs if (r.get("tokens") or {}).get("total"))
    wall = sum(t.get("elapsed") or 0 for t in tasks)

    stages = ["generate", "segment", "segment_result", "evaluate",
              "evaluate_result", "commit", "verify", "verify_result"]
    present = [s for s in stages if any(r.get("stage") == s for r in recs)]
    chips = "".join(f'<span class="chip" data-stage="{s}">{s}</span>' for s in present)

    head = (
        f"{run.get('benchmark', '?')} · {run.get('model', '?')} · "
        f"pipeline={run.get('pipeline', '?')}"
    )
    sub = (
        f"{n} задач · верно {correct}/{n} · {calls} вызовов LLM · "
        f"{tokens:,} токенов · обрывов {trunc} · {wall / 3600:.1f} ч"
        + (f" · ${cost:.4f}" if cost else "")
    ).replace(",", " ")

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Траектории · {run.get('benchmark', 'run')}</title>
<style>{CSS}</style></head><body>
<header>
  <div><h1>{head}</h1><div class="sub">{sub}</div></div>
  <button id="theme">☾ Dark</button>
</header>
<div class="wrap">
  <aside class="panel side">
    <div class="hd">Задачи ({n})</div>
    <div class="tasklist" id="tasks"></div>
  </aside>
  <main>
    <div class="metrics" id="metrics"></div>
    <div class="panel problem" id="problem"></div>
    <div id="gaveup"></div>
    <div class="controls">
      <input id="q" type="search" placeholder="Поиск по промптам, ответам, обоснованиям…  (/)">
      <span id="chips">{chips}</span>
    </div>
    <div id="steps"></div>
  </main>
</div>
<script id="data" type="application/json">{payload}</script>
<script>{JS}</script>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path, help="JSON от --trajectory")
    ap.add_argument("-o", "--output", type=Path, help="куда писать HTML (по умолчанию рядом)")
    args = ap.parse_args()

    if not args.trajectory.exists():
        print(f"Не найден файл траекторий: {args.trajectory}")
        return 2

    data = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if not data.get("tasks"):
        print("В файле нет задач — записывался ли прогон с --trajectory?")
        return 1

    out = args.output or args.trajectory.with_suffix(".html")
    out.write_text(build_html(data), encoding="utf-8")
    size_mb = out.stat().st_size / 1024 ** 2
    print(f"Готово: {out}  ({size_mb:.1f} МБ, задач: {len(data['tasks'])})")
    print("Откройте файл в браузере — данные встроены, сервер не нужен.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
