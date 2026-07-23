"""Самый базовый агентский луп на LangGraph — без инструментов.

Структура графа сейчас:  START → agent → (END).
Это скелет настоящего ReAct-лупа: узел `agent` вызывает модель, а условное
ребро `should_continue` решает — продолжать цикл или завершить. Пока
инструментов нет, модель сразу выдаёт финальный ответ, поэтому цикл делает
ровно один проход и выходит в END.

Когда добавим инструменты, появится узел `tools`, и ветка "continue" будет
вести agent → tools → agent — вот тогда граф закрутится в полноценный луп.
"""
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Состояние агента — история сообщений.

    Reducer `add_messages` дописывает новые сообщения к истории, а не
    затирает её (иначе на каждом шаге терялся бы контекст).
    """

    messages: Annotated[list, add_messages]


def build_agent_loop(model: Any) -> Any:
    """Собрать и скомпилировать минимальный агентский граф вокруг модели.

    `model` — любой LangChain chat-model (у нас ChatOpenAI на OpenAI-совместимый
    эндпоинт: Yandex AI Studio, vLLM или ollama на GPU-карте).
    """

    def agent(state: AgentState) -> dict[str, Any]:
        """Один шаг лупа: вызвать модель на текущей истории и дописать ответ."""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Продолжать цикл или выходить.

        Без инструментов ответ модели финальный → "end".
        С инструментами: если модель попросила tool_call — вернуть "continue"
        (пойдём в узел инструментов и вернёмся в agent). Это и замыкает луп.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "continue"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        # "continue" пока замкнут на сам agent (без тулов сюда не попадаем);
        # с инструментами здесь будет: {"continue": "tools", "end": END}.
        {"continue": "agent", "end": END},
    )
    return graph.compile()
