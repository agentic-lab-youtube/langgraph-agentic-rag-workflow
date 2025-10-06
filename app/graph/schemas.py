from typing import List

from pydantic import BaseModel, Field
from enum import Enum


class QueryType(str, Enum):
    """Enumeration for the type of user query."""

    CASUAL = "casual"
    HISTORICAL = "historic"
    NEEDS_SEARCH = "needs_search"


class AnswerQuestion(BaseModel):
    """
    Categorizes the user's query and provides necessary search terms if required.
    """

    query_type: QueryType = Field(
        ...,
        description=(
            "Categorize the user's query. Use 'casual' for greetings/chit-chat, "
            "'historic' if the answer is in the chat history, and "
            "'needs_search' if a new search is required."
        ),
    )

    search_queries: List[str] = Field(
        ...,
        description=(
            "A list of 1-2 search queries for the RAG system. "
            "Return an empty list if query_type is 'casual' or 'historical'."
        ),
    )

    list_of_incident_numbers: List[str] = Field(
        ...,
        description=(
            "A list of exact incident numbers (e.g., 'INC12345') from the user's query. "
            "If no incident numbers are mentioned, return an empty list."
        ),
    )


class VerificationModel(BaseModel):
    """A structured assessment of the generated answer."""

    is_sufficient: bool = Field(
        ...,
        description="Is the answer detailed, accurate, and sufficient to fully address the user's query?",
    )
    reflection: str = Field(
        ...,
        description="If not sufficient, provide a concise critique. What is missing? What is superfluous? This will be used as a new query to generate a better answer. If the answer is sufficient return empty string.",
    )
