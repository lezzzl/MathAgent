import operator
import json
import re
from typing import Annotated, List, Optional, Any, Dict, Tuple
from typing_extensions import TypedDict
from dataclasses import dataclass

import ollama
from math_verify import parse, verify
from langgraph.graph import StateGraph, END

import yaml
from pathlib import Path

MODEL_NAME = "llama3.2"
_BOXED_RE = re.compile(r"\\boxed\{")

@dataclass
class Role:
    name: str
    system_prompt: str
    user_template: str = "{context}"   
    temperature: float = 0.6
    json_format: bool = False
    num_predict: Optional[int] = None

    def build_messages(self, **kwargs) -> List[dict]:
        """Собирает messages для _chat() из system_prompt + user_template.
        kwargs — именованные значения для плейсхолдеров шаблона (например,
        context=..., step=...). Лишние kwargs, которых нет в шаблоне, просто
        игнорируются — так один и тот же вызов подходит и для ролей, которым
        нужен только context, и для тех, кому нужен ещё и step."""
        try:
            user_content = self.user_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Роль '{self.name}': user_template ссылается на плейсхолдер {e}, "
                f"которого нет среди переданных аргументов ({sorted(kwargs)}). "
                f"Проверьте user_template в agent-step-v1.yml для этой роли."
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


# Дефолты на случай, если YAML не найден
_DEFAULT_ROLE_DEFS: Dict[str, Dict[str, Any]] = {
    "generator": {
        "system": (
            "You are an experienced mathematics problem solver.\n"
            "You will be given a problem statement and any current solution steps. "
            "Your task is to generate ONLY THE NEXT SINGLE STEP of the solution that brings you closer to the answer. "
            "Do not solve the entire problem at once.\n"
            "If this step completes the solution and you know the final answer, end the step with the exact format:\n"
            "Final answer: \\boxed{ANSWER}"
        ),
        "user_template": "{context}",
        "temperature": 0.6,
        "json_format": False,
        "num_predict": 400,
    },
    "evaluator": {
        "system": (
            "You are a pedantic and strict evaluator of mathematical solution steps.\n"
            "Analyze the problem context and the proposed next step.\n"
            "You must respond with a single, valid JSON object matching this schema exactly:\n\n"
            "{\n"
            '  "mathematical_analysis": "Recalculate the step yourself. Is the arithmetic correct? '
            'Are the constraints of the original problem (e.g., distinct digits, non-zero) strictly maintained?",\n'
            '  "rationale": "Explanation of the quality of the step.",\n'
            '  "score": 0.0\n'
            "}\n\n"
            "CRITICAL RULES:\n"
            "1. If the step contains ANY mathematical error, wrong numbers, or violates a problem constraint "
            "(like accidentally allowing zero), you MUST give it a score BELOW 0.5.\n"
            "2. Never say a step has a mistake but give it a high score. If it has a mistake, fail it!\n"
            "3. If the step is perfectly accurate and logically advances the solution, give it a 1.0."
        ),
        "user_template": "{context}\nGive a score of the new step:\n{step}",
        "temperature": 0.1,
        "json_format": True,
        "num_predict": 900,
    },
    "verifier": {
        "system": (
            "You are a highly precise verifier of mathematical solutions.\n"
            "Review the entire step-by-step solution provided.\n"
            "You must respond with a single, valid JSON object matching this schema exactly.\n\n"
            'Crucial: You must write the "step_by_step_analysis" first to double-check every single '
            "calculation in the solution before deciding on the final verdict.\n\n"
            "{\n"
            '  "step_by_step_analysis": "Go through each step of the solution one by one. Recalculate everything yourself to ensure no errors exist.",\n'
            '  "rationale": "Detailed explanation of your verification verdict.",\n'
            '  "is_valid": false,\n'
            '  "confidence": 0.0,\n'
            '  "faulty_step_index": null\n'
            "}\n\n"
            "Verification Rules:\n"
            "- Double check your own math. Do not claim correct equations or calculations are incorrect.\n"
            "- Dividing both sides of an equation (e.g., -2x = -10) by a negative number (e.g., -2) results in a "
            "positive value (e.g., x = 5) and is perfectly correct. Equations are NOT inequalities; there is no "
            "direction to reverse.\n"
            "- When counting three-digit numbers (e.g., permutations), the order of digits matters (e.g., 123 is "
            "different from 321). Do NOT divide by factorials (like 3!) unless order does not matter.\n"
            '- "is_valid": boolean (true if the entire solution is flawless and correct, false otherwise).\n'
            '- "confidence": float from 0.0 to 1.0.\n'
            '- "faulty_step_index": integer (0-indexed) of the first incorrect step, or null if correct.'
        ),
        "user_template": "{context}",
        "temperature": 0.0,
        "json_format": True,
        "num_predict": 900,
    },
}

ROLES: Dict[str, Role] = {
    name: Role(
        name=name,
        system_prompt=cfg["system"],
        user_template=cfg["user_template"],
        temperature=cfg["temperature"],
        json_format=cfg["json_format"],
        num_predict=cfg["num_predict"],
    )
    for name, cfg in _DEFAULT_ROLE_DEFS.items()
}


def load_prompts_from_yaml(yaml_path: "Path | str") -> None:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        roles_cfg = config.get("roles", {})
        if not roles_cfg:
            print(f"[Prompts] Warning: {yaml_path} has no 'roles' section. Using hardcoded defaults.")
            return

        for role_name, role_cfg in roles_cfg.items():
            if role_name not in _DEFAULT_ROLE_DEFS:
                print(f"[Prompts] Warning: unknown role '{role_name}' in {yaml_path}, ignoring.")
                continue
            defaults = _DEFAULT_ROLE_DEFS[role_name]
            ROLES[role_name] = Role(
                name=role_name,
                system_prompt=role_cfg.get("system", defaults["system"]),
                user_template=role_cfg.get("user_template", defaults["user_template"]),
                temperature=float(role_cfg.get("temperature", defaults["temperature"])),
                json_format=bool(role_cfg.get("json_format", defaults["json_format"])),
                num_predict=role_cfg.get("num_predict", defaults["num_predict"]),
            )

        print(f"[Prompts] Successfully loaded prompts from {yaml_path}")
    except Exception as e:
        print(f"[Prompts] Warning: Could not load prompt file ({e}). Using hardcoded defaults.")


# --- Utilities ---
def _chat(
    messages: List[dict],
    json_format: bool = False,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    num_predict: Optional[int] = None,
) -> Tuple[str, int]:
    if num_predict is None:
        num_predict = 900 if json_format else 400
    options = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = seed

    kwargs: Dict[str, Any] = {"model": MODEL_NAME, "messages": messages, "options": options}
    if json_format:
        kwargs["format"] = "json"

    response = ollama.chat(**kwargs)
    total_tokens = response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
    return (response["message"].get("content") or "").strip(), total_tokens

def _build_context(problem: str, steps: List[str]) -> str:
    context = f"Task: {problem}\n\nCurrent solution:\n"
    for i, step in enumerate(steps):
        context += f"Step {i}: {step}\n"
    return context

def extract_answer(step_text: str) -> Optional[str]:
    matches = list(_BOXED_RE.finditer(step_text))
    if not matches:
        return None
    tail = step_text[matches[-1].start():]
    if not parse(tail):
        return None
    return tail


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    problem: str
    steps: Annotated[List[str], operator.add]  
    
    candidate_steps: List[str]
    candidate_scores: List[float]
    
    k_branches: int
    score_threshold: float
    branch_mode: str
    base_temperature: float
    
    tokens_used: Annotated[int, operator.add]
    token_budget: int  
    in_recovery: bool
    recovery_count: int  
    max_recoveries: int  
    total_recovery_events: Annotated[int, operator.add]  

    final_answer: Optional[str]
    is_valid: bool
    verifier_rationale: str
    gave_up: bool         
    gave_up_reason: str    


# ---------------------------------------------------------------------------
# 2. Graph Nodes
# ---------------------------------------------------------------------------
def generate_step(state: AgentState):
    current_depth = len(state.get('steps', []))
    print(f"\n[Node: Generate] Depth: {current_depth} | Tokens used: {state.get('tokens_used', 0)}")
    
    k = state['k_branches'] if (state['branch_mode'] == 'multi' or state['in_recovery']) else 1
    print(f"  -> Generating {k} candidate(s).")

    role = ROLES["generator"]
    candidates = []
    total_tokens = 0
    context = _build_context(state['problem'], state.get('steps', []))
    
    for i in range(k):
        temp = state['base_temperature'] + (0.15 * i) if k > 1 else state['base_temperature']
        messages = role.build_messages(context=context)
        
        step_text, tks = _chat(messages, temperature=temp, seed=1000 + i, num_predict=role.num_predict)
        candidates.append(step_text)
        total_tokens += tks
        
        print(f"    - Branch {i+1} generated (temp: {temp:.2f}, tokens: {tks})")
        print(f"      Step:\n{step_text}\n")
        
    return {"candidate_steps": candidates, "tokens_used": total_tokens}


def evaluate_steps(state: AgentState):
    print(f"\n[Node: Evaluate] Checking {len(state['candidate_steps'])} candidate(s)...")
    role = ROLES["evaluator"]
    scores = []
    total_tokens = 0
    context = _build_context(state['problem'], state.get('steps', []))
    
    for i, step in enumerate(state['candidate_steps']):
        messages = role.build_messages(context=context, step=step)
        content, tks = _chat(messages, json_format=role.json_format, temperature=role.temperature, num_predict=role.num_predict)
        total_tokens += tks

        try:
            result_dict = json.loads(content)
            score = max(0.0, min(1.0, float(result_dict.get("score", 0.5))))
            rationale = str(result_dict.get("rationale", "No rationale extracted"))
        except (json.JSONDecodeError, ValueError, TypeError):
            score, rationale = 0.5, "JSON Parse Error"

        scores.append(score)
        print(f"    - Candidate {i+1} Score: {score:.4f} | Rationale: {rationale}")
        
    return {"candidate_scores": scores, "tokens_used": total_tokens}


def trigger_recovery(state: AgentState):
    total_so_far = state.get('total_recovery_events', 0) + 1
    print(f"\n[Node: Recovery] Triggering k-branch recovery for the current step. "
          f"(Event {total_so_far}/{state['max_recoveries']} for the entire run)")
    return {
        "in_recovery": True,
        "recovery_count": state.get('recovery_count', 0) + 1,
        "total_recovery_events": 1,  
    }


def give_up(state: AgentState):
    reason = (
        f"The token budget is exhausted ({state.get('tokens_used', 0)}/{state.get('token_budget', 0)}), "
        f"clear \\boxed{{}} was not received."
    )
    print(f"\n[Node: Give Up] {reason}")
    return {
        "final_answer": None,
        "is_valid": False,
        "verifier_rationale": reason,
        "gave_up": True,
        "gave_up_reason": reason,
    }


def commit_step(state: AgentState):
    best_idx = max(range(len(state['candidate_scores'])), key=lambda i: state['candidate_scores'][i])
    best_step = state['candidate_steps'][best_idx]
    best_score = state['candidate_scores'][best_idx]
    
    print(f"\n[Node: Commit] Selected best candidate (Score: {best_score:.4f}). Appending to steps.")
    answer = extract_answer(best_step)
    if answer: print(f"  -> Explicit answer found: {answer}")
        
    return {
        "steps": [best_step],       
        "final_answer": answer if answer else "",
        "in_recovery": False,       
        "recovery_count": 0,        
        "candidate_steps": [],      
        "candidate_scores": []
    }


def verify_solution(state: AgentState):
    print("\n[Node: Verify] Running verifier...")
    role = ROLES["verifier"]
    context = _build_context(state['problem'], state.get('steps', []))
    messages = role.build_messages(context=context)
    content, tks = _chat(messages, json_format=role.json_format, temperature=role.temperature, num_predict=role.num_predict)

    try:
        result_dict = json.loads(content)
        is_valid = bool(result_dict.get("is_valid", False))
        rationale = str(result_dict.get("rationale", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        is_valid, rationale = False, "JSON Parse Error"

    print(f"  -> Valid: {is_valid} | Rationale: {rationale}")
    
    return {
        "is_valid": is_valid,
        "verifier_rationale": rationale,
        "tokens_used": tks
    }


# ---------------------------------------------------------------------------
# 3. Conditional Edge Routers
# ---------------------------------------------------------------------------
def route_after_eval(state: AgentState):
    best_score = max(state['candidate_scores'])
    tokens_used = state.get('tokens_used', 0)
    token_budget = state.get('token_budget', 10**9)

    if tokens_used >= token_budget:
        print(f"\n[Router] The token budget is exhausted ({tokens_used}/{token_budget}). "
              f"Commit the best available option without further attempts.")
        return "commit"

    total_recoveries = state.get('total_recovery_events', 0)

    if (best_score < state['score_threshold'] 
        and not state['in_recovery'] 
        and state['branch_mode'] == 'single'
        and total_recoveries < state['max_recoveries']):
        print(f"\n[Router] Best score {best_score:.4f} < Threshold {state['score_threshold']}. Initiating recovery.")
        return "recover"

    if best_score < state['score_threshold'] and not state['in_recovery'] and total_recoveries >= state['max_recoveries']:
        print(f"\n[Router] Recovery event limit for the entire run ({state['max_recoveries']}) already exhausted"
              f"({total_recoveries} used). Commit without a new branch.")
        return "commit"

    if best_score < state['score_threshold'] and state['in_recovery']:
        print(f"\n[Router] All {state['k_branches']} branches scored below threshold. Forcing commit of highest score ({best_score:.4f}).")
        return "commit"
        
    print(f"\n[Router] Score {best_score:.4f} meets threshold. Committing.")
    return "commit"

def route_after_commit(state: AgentState):
    if state.get("final_answer"):
        return "verify"
    if state.get('tokens_used', 0) >= state.get('token_budget', 10**9):
        return "give_up"
    return "generate"


# ---------------------------------------------------------------------------
# 4. Graph Construction
# ---------------------------------------------------------------------------
def build_solver_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_step", generate_step)
    workflow.add_node("evaluate_steps", evaluate_steps)
    workflow.add_node("trigger_recovery", trigger_recovery)
    workflow.add_node("commit_step", commit_step)
    workflow.add_node("verify_solution", verify_solution)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("generate_step")

    workflow.add_edge("generate_step", "evaluate_steps")
    workflow.add_conditional_edges("evaluate_steps", route_after_eval, {"recover": "trigger_recovery", "commit": "commit_step"})
    workflow.add_edge("trigger_recovery", "generate_step")
    workflow.add_conditional_edges("commit_step", route_after_commit, {"verify": "verify_solution", "generate": "generate_step", "give_up": "give_up"})
    workflow.add_edge("verify_solution", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()