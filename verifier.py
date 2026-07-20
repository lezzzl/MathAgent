import json
import re
import ollama
import yaml

MODEL_NAME = 'qwen3.5:4b'

with open("prompts/verificator-v0.yml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

def get_system_prompt():
    base_prompt = config["verifier_system_prompt"]
    return base_prompt
def get_user_prompt(question: str, gt: str, model_answer: str,  math_res: dict) -> str:
    parsed_gt = math_res.get("parsed_ground_truth", "Не распознано")
    parsed_ans = math_res.get("parsed_model_answer", "Не распознано")
    is_matched = math_res.get("answer_matched", False)

    if is_matched:
        equiv_text = f"ВЕРДИКТ СИСТЕМЫ: Ответ студента ({parsed_ans}) математически ЭКВИВАЛЕНТЕН ground_truth ({parsed_gt}). Разница в формате записи НЕ является ошибкой. Считайте итоговый ответ ВЕРНЫМ."
    else:
        equiv_text =  f"ВЕРДИКТ СИСТЕМЫ: Ответ студента ({parsed_ans}) математически НЕ РАВЕН ground_truth ({parsed_gt}). Итоговый ответ НЕВЕРНЫЙ"
    
    return f"""Вы — математический аудитор. Вы НЕ решаете задачи с нуля. Вы только читаете текст решения студента, ищете в нем логические или арифметические ошибки и сравниваете итоговый ответ с ground_truth.
    ВАЖНО: Для сверки итоговых ответов используйте блок <math_equivalence_check>. Если там указано, что ответы эквивалентны, НЕ считайте разницу в формате записи ошибкой!

    Вот пример того, как вы должны работать:

    <question>
    Найдите значение x, если 2x + 4 = 10.
    </question>
    <ground_truth>
    3
    </ground_truth>
    <student_solution>
    Переносим 4 вправо: 2x = 10 + 4. Получаем 2x = 14. Делим на 2: x = 7. Ответ: \\boxed{{7}}.
    </student_solution>
    <math_equivalence_check>
    ВЕРДИКТ СИСТЕМЫ: Ответ студента (7) математически НЕ РАВЕН ground_truth (3). Итоговый ответ НЕВЕРНЫЙ.
    </math_equivalence_check>

    <verdict>
    FALSE
    </verdict>
    <analysis>
    Ответ студента (7) не совпадает с ground_truth (3). В решении допущена алгебраическая ошибка при переносе слагаемого через знак равенства (студент прибавил 4 вместо вычитания).
    </analysis>
    <feedback>
    Вспомните правило переноса: при переносе через знак равенства знак слагаемого меняется на противоположный.
    </feedback>

    ---
    Теперь проведите аудит для следующей задачи:

    <question>
    {question}
    </question>

    <ground_truth>
    {gt}
    </ground_truth>

    <student_solution>
    {model_answer}
    </student_solution>

    <math_equivalence_check>
    {equiv_text}
    </math_equivalence_check>

    Сформируйте ответ, строго следуя этой структуре:
    <verdict>
    Напишите слово TRUE (если решение абсолютно верное) или FALSE (если есть ошибки или не совпадает ответ).
    </verdict>

    <analysis>
    Краткое резюме (3-5 предложений): почему студент прав или неправ. Укажите на конкретные шаги, где допущена ошибка, и сверьте ответ с ground_truth.
    </analysis>

    <feedback>
    Конструктивная подсказка: как студенту исправить ошибку и прийти к правильному ответу, либо похвала, если решение верное.
    </feedback>
    """
def clean_and_parse_llm(raw_text: str, logger):
    text = raw_text.strip()

    if not text.startswith("<verdict>"):
        text = "<verdict>\n" + text
    
    if not text.rstrip().endswith("</feedback>"):
        text = text.rstrip() + "\n</feedback>"

    # убираем возможные остатки пре-филла в середине текста
    text = re.sub(r"</verdict>\s*<verdict>", "</verdict>", text, flags=re.DOTALL)
    text = re.sub(r"</analysis>\s*<analysis>", "</analysis>", text, flags=re.DOTALL)
    text = re.sub(r"</feedback>\s*<feedback>", "</feedback>", text, flags=re.DOTALL)
    
    def extract_tag(tag_name: str, source_text: str):
        """Извлекает содержимое между указанными тегами."""
        pattern = fr"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, source_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    verdict_raw = extract_tag("verdict", text)
    analysis = extract_tag("analysis", text) or "Модель не заполнила описание."
    feedback = extract_tag("feedback", text) or "Решение требует проверки."
    
    is_correct = False
    if verdict_raw:
        verdict_clean = verdict_raw.lower().strip()
        is_correct = "true" in verdict_clean
    
    result = {
        "is_correct_solution": bool(is_correct),
        "analysis_thoughts": analysis,
        "feedback_for_solver": feedback
    }
    
    return result

def call_llm_verifier(question: str, gt: str, model_answer: str, logger, math_res: dict):
    sys_prompt = get_system_prompt()
    user_content = get_user_prompt(question, gt, model_answer, math_res)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
                {"role":"assistant", "content": "<verdict>\n"}
            ],
            options={'temperature': 0.1, 
                     'num_ctx': 8192,       
                    'num_predict': 2048,
                    'stop': ['</feedback>']
                    }
        )

        raw_content = response['message'].get('content', '') 
        # logger.info(f"raw_content_message: {raw_content}")
        return clean_and_parse_llm(raw_content, logger)
    
    except Exception as e:
        logger.error(f"Сбой при подключении к LLM или обработке её ответа: {e}")
        return {
            "is_correct_solution": False,
            "analysis_thoughts": f"Сбой работы экспертной LLM: {str(e)}",
            "feedback_for_solver": raw_content.strip() if 'raw_content' in locals() else "Ошибка верификатора."
        }
