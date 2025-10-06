from typing import List, TypedDict, Annotated
from .schemas import AnswerQuestion, VerificationModel
from langchain_core.messages import BaseMessage


def trim_to_most_recent_ten(
    messages: List[BaseMessage], new_messages: List[BaseMessage]
) -> List[BaseMessage]:
    """
    A custom reducer that combines messages and returns only the 5 most recent ones.
    """
    combined_messages = messages + new_messages
    return combined_messages[-10:]


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    query: str
    is_follow_up: bool
    messages: Annotated[list, trim_to_most_recent_ten]
    answer: str
    initial_answer: AnswerQuestion
    references: str
    verification: VerificationModel
    metadata: List[dict]
