from langgraph.graph import StateGraph, END
from .graph_state import GraphState
from .nodes import (
    generate_initial_answer,
    run_tool_node,
    should_continue,
    generate_casual_answer,
    generate_historic_answer,
    final_answer,
    quality_gate_node,
    should_continue_after_verify,
    check_for_followup,
)
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(GraphState)
memory = MemorySaver()

# Add the nodes
workflow.add_node("follow_up_check", check_for_followup)
workflow.add_node("generate_initial_answer", generate_initial_answer)
workflow.add_node("casual_response", generate_casual_answer)
workflow.add_node("historic_reponse", generate_historic_answer)
workflow.add_node("run_tools", run_tool_node)
workflow.add_node("final_answer", final_answer)
workflow.add_node("quality_gate", quality_gate_node)

# Set the entry point
workflow.set_entry_point("follow_up_check")
workflow.add_edge("follow_up_check", "generate_initial_answer")

# Add the conditional edge
workflow.add_conditional_edges(
    "generate_initial_answer",
    should_continue,
    {
        "casual": "casual_response",
        "historic": "historic_reponse",
        "needs_search": "run_tools",
    },
)

# Add the normal edges
workflow.add_edge("casual_response", END)
workflow.add_edge("historic_reponse", "final_answer")
workflow.add_edge("run_tools", "final_answer")
workflow.add_edge("final_answer", "quality_gate")
workflow.add_conditional_edges(
    "quality_gate",
    should_continue_after_verify,
    {"continue": "follow_up_check", "end": END},
)

# Compile the graph
final_graph = workflow.compile(checkpointer=memory)
print(final_graph.get_graph().draw_mermaid())
