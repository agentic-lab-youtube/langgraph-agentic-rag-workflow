from .graph_state import GraphState
from .chains import (
    first_responder,
    second_responder,
    casual_response_chain,
    history_aware_chain,
    verifier_chain,
    is_follow_up_chain,
)
from .tool_executor import run_queries, run_query_feedback
from langchain_core.messages import AIMessage, HumanMessage
from .schemas import QueryType


def check_for_followup(state: GraphState):
    """Checks Whether the asked Query Is a followup Question or not"""
    print("---CHECK FOR FOLLOWUP---")
    latest_query = state.get("query", "")
    chat_history = state["messages"]
    is_follow_up = is_follow_up_chain.invoke(
        {"chat_history": chat_history, "query": latest_query}
    )
    print(f"---FOLLOWUP Answer---{is_follow_up}")
    return {"is_follow_up": is_follow_up}


def generate_initial_answer(state: GraphState):
    """Generates the initial decision."""
    print("---GENERATING INITIAL ANSWER---")

    latest_query = state.get("query", "")
    chat_history = state["messages"]
    is_follow_up = state["is_follow_up"]
    print(f"---QUERY--- {latest_query}")
    response = first_responder.invoke(
        {
            "chat_history": chat_history,
            "query": latest_query,
            "is_follow_up": is_follow_up,
        }
    )
    print(f"initial-ans {response}")
    return {"initial_answer": response}


def generate_casual_answer(state: GraphState):
    """Generates the casual answer for casual workflow."""
    print("---GENERATING CASUAL ANSWER---")
    latest_query = state.get("query", "")
    chat_history = state["messages"]
    response = casual_response_chain.invoke(
        {"chat_history": chat_history, "query": latest_query}
    )
    return {"answer": response, "metadata": []}


def generate_historic_answer(state: GraphState):
    """Generates the historic answer for historic workflow."""
    print("---GENERATING HISTORIC ANSWER---")
    latest_query = state.get("query", "")
    chat_history = state["messages"]
    response = history_aware_chain.invoke(
        {"chat_history": chat_history, "query": latest_query}
    )
    return {"messages": [AIMessage(content=response)], "metadata": []}


def run_tool_node(state: GraphState):
    print("---RUNNING TOOLS---")
    ia = state.get("initial_answer", {}) or {}
    search_queries = list(ia.get("search_queries", []) or [])
    search_numbers = ia.get("list_of_incident_numbers", []) or []

    nested_references = run_queries(search_queries, search_numbers)

    nested_feedacks = run_query_feedback(search_queries)

    print(f"references {nested_references}")

    if not nested_references:
        return {"references": "", "metadata": []}

    flat_feedback = [doc for sublist in nested_feedacks for doc in sublist]
    flat_references = [doc for sublist in nested_references for doc in sublist]
    page_contents = [doc.page_content for doc in flat_references]
    feedback_contents = [doc.page_content for doc in flat_feedback]
    combined_context = "\n".join(page_contents)
    combined_feedback = "\n".join(feedback_contents)
    metadata_list = [doc.metadata for doc in flat_references]

    return {
        "references": combined_context + combined_feedback,
        "metadata": metadata_list,
    }


def final_answer(state: GraphState):
    """Finalizes the answer based on the retrieved references."""
    print("---FINALIZING ANSWER---")
    response = second_responder.invoke(
        {
            "chat_history": state["messages"],
            "query": state["query"],
            "references": state.get("references", ""),
        }
    )
    return {"messages": [AIMessage(content=response)], "answer": response}


def quality_gate_node(state: GraphState):
    """
    This node acts as a quality gate. It verifies the answer and decides whether to end the workflow
    or send it back for another revision.
    """
    query = state["query"]
    answer = state["answer"]

    verification_result = verifier_chain.invoke({"query": query, "answer": answer})

    if verification_result["is_sufficient"]:
        print(
            "---QUALITY GATE: Answer is sufficient or max revisions reached. Ending workflow.---"
        )
        return {"verification": verification_result}
    else:
        reflection_message = HumanMessage(
            content=f"{verification_result.get('reflection','')}"
        )
        return {
            "messages": [reflection_message],
            "verification": verification_result,
            "query": verification_result["reflection"],
        }


def should_continue_after_verify(state: GraphState) -> str:
    """
    Conditional edge logic. Determines if we should continue revising or end.
    """
    verification = state.get("verification", {})
    is_complete = verification.get("is_sufficient")
    if is_complete:
        return "end"
    else:
        return "continue"


def should_continue(state: GraphState):
    """Decides whether to continue to the tool node or end."""
    print("---CHECKING FOR DECISION---")
    print(state["initial_answer"]["query_type"])
    if state["initial_answer"]["query_type"] == QueryType.CASUAL:
        print("---DECISION: CASUAL---")
        return "casual"
    elif state["initial_answer"]["query_type"] == QueryType.NEEDS_SEARCH:
        print("---DECISION: NEEDS_SEARCH---")
        return "needs_search"
    elif state["initial_answer"]["query_type"] == QueryType.HISTORICAL:
        print("---DECISION: HISTORIC---")
        return "historic"
