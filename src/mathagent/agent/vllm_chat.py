from typing import Any

from langchain_openai import ChatOpenAI


class ChatVLLM(ChatOpenAI):
    """Расширяет ChatOpenAI сохранением reasoning из ответа vLLM.

    Он дополнительно переносит нестандартные
    поля vLLM ``reasoning`` и ``reasoning_content``, которые обычный ChatOpenAI
    отбрасывает, в ``AIMessage.additional_kwargs``. Это позволяет отдельно
    записывать reasoning и финальный content модели в JSONL.
    """

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        """Дополняет результат ChatOpenAI полями reasoning из сырого ответа."""
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        result = super()._create_chat_result(response, generation_info)
        for generation, choice in zip(
            result.generations,
            response_dict.get("choices") or [],
            strict=False,
        ):
            raw_message = choice.get("message") or {}
            for field in ("reasoning", "reasoning_content"):
                value = raw_message.get(field)
                if isinstance(value, str) and value:
                    generation.message.additional_kwargs[field] = value
        return result
