import json
import logging
import re
import verifier_step as verifier
from pathlib import Path
from math_verify import parse, verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_and_verify_math(model_answer: str, ground_truth):
    """
    Сверка итогового ответа студента с ground_truth средствами math_verify.
    """
    result = {"answer_matched": False, "parsed_model_answer": None, "parsed_ground_truth": None}

    if not model_answer:
        return result

    boxed_matches = list(re.finditer(r'\\boxed{', model_answer))
    if not boxed_matches:
        return result
    
    trimmed_answer = model_answer[boxed_matches[-1].start():]
    try:
        parsed_gt = parse(str(ground_truth), parsing_timeout=0)
        parsed_answer = parse(trimmed_answer, parsing_timeout=0)
        result["parsed_ground_truth"] = str(parsed_gt) if parsed_gt else None
        result["parsed_model_answer"] = str(parsed_answer) if parsed_answer else None
        if parsed_answer and parsed_gt:
            result["answer_matched"] = bool(verify(parsed_gt, parsed_answer, timeout_seconds=0))
    except Exception as e:
        logger.error(f"Внутренняя ошибка math_verify: {e}")
    return result

def run_verification(input_file: str = "agent_output.json", output_file: str = "agent_output_verified.json"):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Входной файл '{input_file}' не найден. Создайте его рядом со скриптом.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            tasks_list = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Файл '{input_file}' содержит некорректный JSON-формат.")
            return

    if isinstance(tasks_list, dict):
        tasks_list = [tasks_list]

    logger.info(f"Загружено задач из '{input_file}' для проверки: {len(tasks_list)}")

    for idx, task in enumerate(tasks_list):
        if not isinstance(task, dict):
            logger.warning(f"Элемент под индексом {idx} пропущен, так как он не является JSON-объектом.")
            continue

        task_id = task.get('id', f'unknown_{idx}')
        logger.info(f"[{idx + 1}/{len(tasks_list)}] Верификация задачи ID: {task_id}")

        model_ans = task.get("model_answer", "")
        gt_val = task.get("ground_truth", "")
        question_text = task.get("question", "Условие задачи отсутствует")

        # Сверка с ground_truth (только для метрик, LLM её не видит).
        math_res = parse_and_verify_math(model_ans, gt_val)
        task["is_correct_answer"] = math_res["answer_matched"]
        task["parsed_model_answer"] = math_res["parsed_model_answer"]
        task["parsed_ground_truth"] = math_res["parsed_ground_truth"]

        logger.info(f"Вызываем {verifier.MODEL_NAME} для независимой проверки решения задачи {task_id} (без ground_truth)...")
        model_analysis = verifier.call_llm_verifier(question_text, str(model_ans), logger)
        logger.info(
            f"VERDICT (LLM, без ground_truth): {model_analysis['is_correct_solution']} | "
            f"Сверка с ground_truth (только для метрик): {task['is_correct_answer']} | "
            f"is_truncated: {model_analysis.get('is_truncated', False)}"
        )

        task["is_correct_solution"] = model_analysis["is_correct_solution"]
        task["analysis_thoughts"] = model_analysis["analysis_thoughts"]
        task["feedback_for_solver"] = model_analysis["feedback_for_solver"]
        # True, если на любом из шагов верификации не хватило
        # num_predict (а не просто сработала стоп-последовательность)
        # (такие записи стоит отделять от честного is_correct_solution=False
        # при подсчёте метрик, а не считать за подтверждённый вердикт).
        task["is_truncated"] = model_analysis.get("is_truncated", False)
        if "steps" in model_analysis:
            task["verification_steps"] = model_analysis["steps"]
            
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Верификация успешно завершена! Результаты сохранены в '{output_file}'")
    except Exception as e:
        logger.error(f"Не удалось сохранить выходной файл '{output_file}': {e}")
        
    logger.info(f"Проверка завершена. Результаты успешно записаны в новый файл: '{output_file}'")

if __name__ == '__main__':
    run_verification("agent_output.json", "agent_output_verified.json")