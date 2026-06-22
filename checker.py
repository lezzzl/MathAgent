import re
from typing import Optional, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIMEValidator")

class AIMEAnswerValidator:
    def __init__(self):
        # Регулярное выражение для поиска контента внутри \boxed{...}
        # Учитывает возможные переносы строк (re.DOTALL)
        self.boxed_regex = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)
        
    def extract_raw_answer(self, model_output: str) -> Optional[str]:
        """
        Этап 1: Extraction. Извлекает содержимое из последнего блока \boxed{}.
        """
        if not model_output or not isinstance(model_output, str):
            return None
        
        matches = self.boxed_regex.findall(model_output)
        if not matches:
            logger.warning("Валидация провалена: Блок \\boxed{} не найден в ответе агента.")
            return None
        
        final_match = matches[-1].strip()
        return final_match

    def normalize_aime_format(self, raw_str: str) -> Optional[int]:
        """
        Этап 2: Normalization. Очищает LaTeX-мусор и приводит ответ к строгому Int (0-999).
        """
        if not raw_str:
            return None
        
        cleaned = raw_str.replace(" ", "").replace("$", "")
        cleaned = re.sub(r'\\text\{.*?\}', '', cleaned)
        cleaned = cleaned.replace("{", "").replace("}", "") 
        
        digit_matches = re.findall(r'\d+', cleaned)
        if not digit_matches:
            logger.warning(f"Канонизация провалена: Не удалось найти цифры в строке '{raw_str}'")
            return None
        
        target_digits = digit_matches[-1]
        
        try:
            val = int(target_digits)

            # Валидация по спецификации бенчмарка AIME (целое число от 0 до 999)
            if 0 <= val <= 999:
                return val
            else:
                logger.error(f"Аномалия: Число {val} выходит за рамки диапазона AIME (0-999).")
                return None
        except ValueError:
            return None

    def verify(self, agent_response: str, ground_truth: Union[str, int]) -> bool:
        """
        Этап 3: Comparison. Основная точка входа для проверки.
        """
        try:
            gt_int = int(str(ground_truth).strip())
        except ValueError:
            logger.critical(f"Ошибка датасета: Истинный ответ '{ground_truth}' не является числом.")
            return False

        raw_boxed = self.extract_raw_answer(agent_response)
        
        # Фолбэк-стратегия, если агент забыл \boxed{}, но мы не хотим терять метрики
        if raw_boxed is None:
            return self._fallback_verification(agent_response, gt_int)
            
        agent_int = self.normalize_aime_format(raw_boxed)
        
        if agent_int is None:
            return False
            
        # Строгое сравнение математических значений
        return agent_int == gt_int

    def _fallback_verification(self, agent_response: str, gt_int: int) -> bool:
        """
        Резервный метод на случай нарушения контракта генерации (нет \boxed).
        Ищет число в последних 150 символах ответа агента.
        """
        logger.info("Запуск Fallback-стратегии парсинга без \\boxed{}...")
        tail = agent_response[-150:]
        digit_matches = re.findall(r'\d+', tail)
        
        if digit_matches:
            try:
                if int(digit_matches[-1]) == gt_int:
                    logger.info("Fallback успешно спас правильный ответ!")
                    return True
            except ValueError:
                pass
        return False