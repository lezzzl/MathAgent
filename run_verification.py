import json
import logging
import re
import verifier
from pathlib import Path
from math_verify import parse, verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_text_cutoff(model_answer: str):
    cutoff_pattern = r"\b(и|то|тогда|следовательно|получим|равно|если|так|как),\s*$"
    return bool(re.search(cutoff_pattern, model_answer.strip(), re.IGNORECASE))

def parse_and_verify_math(model_answer: str, ground_truth):
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

        #   небольшая эвристика для проверки обрыва решения; пока что работает не очень хорошо
        if not list(re.finditer(r'\\boxed{', model_ans)) and check_text_cutoff(model_ans):
            task["is_correct"] = False
            task["is_correct_solution"] = False
            task["is_cutoff"] = True
            cutoff_fr = model_ans[-30:].strip()
            task["feedback_for_solver"] = f"Решение прервалось на полуслове: '... {cutoff_fr}'. Продолжи мысль и выдай ответ."
            continue
                
        math_res = parse_and_verify_math(model_ans, gt_val)

        task["is_correct"] = math_res["answer_matched"]
        task["parsed_model_answer"] = math_res["parsed_model_answer"]
        task["parsed_ground_truth"] = math_res["parsed_ground_truth"]
        task["is_cutoff"] = False
        
        logger.info(f"Вызываем {verifier.MODEL_NAME} для анализа логики шагов задачи {task_id}...")
        model_analysis = verifier.call_llm_verifier(question_text, str(gt_val), str(model_ans), logger, math_res)

        final_verdict = math_res["answer_matched"] and model_analysis["is_correct_solution"]

        logger.info(f"VERDICT: {final_verdict}")

        task["is_correct_solution"] = final_verdict
        task["analysis_thoughts"] = model_analysis["analysis_thoughts"]
        task["feedback_for_solver"] = model_analysis["feedback_for_solver"]
            
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Верификация успешно завершена! Результаты сохранены в '{output_file}'")
    except Exception as e:
        logger.error(f"Не удалось сохранить выходной файл '{output_file}': {e}")
        
    logger.info(f"Проверка завершена. Результаты успешно записаны в новый файл: '{output_file}'")

if __name__ == '__main__':
    run_verification("agent_output.json", "agent_output_verified.json")
