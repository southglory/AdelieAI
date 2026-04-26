from typing import Callable

from langgraph.graph import END, START, StateGraph

from core.agent.nodes import (
    make_planner_node,
    make_reasoner_node,
    make_reporter_node,
    make_retriever_node,
)
from core.agent.state import AgentState
from core.retrieval.protocols import Retriever
from core.serving.protocols import LLMClient


def build_agent_graph(
    *,
    llm: LLMClient,
    retriever: Retriever | None,
    on_event: Callable[[str, dict], None] | None = None,
):
    graph = StateGraph(AgentState)
    graph.add_node("planner", make_planner_node(llm, on_event=on_event))
    graph.add_node(
        "retriever", make_retriever_node(retriever, on_event=on_event)
    )
    graph.add_node("reasoner", make_reasoner_node(llm, on_event=on_event))
    graph.add_node("reporter", make_reporter_node(on_event=on_event))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()
