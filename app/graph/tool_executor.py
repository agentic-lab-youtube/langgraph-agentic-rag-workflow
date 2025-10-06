from app.services import get_all_documents, get_all_feedbacks
from typing import List, Optional


def run_queries(
    search_queries: List[str], search_numbers: Optional[List[str]] = None, **kwargs
):
    """
    Runs searches based on text queries and incident numbers separately,
    then returns the aggregated results.
    """
    all_retrieved_docs = []

    print(f"search nums {search_numbers}")

    if search_numbers:
        print("Executing 'if search_numbers:' block...")
        for number in search_numbers:
            documents_for_number = get_all_documents(
                query=number, incident_number=number
            )
            # print(f"search number res: {documents_for_number}")
            if documents_for_number:
                all_retrieved_docs.append(documents_for_number)

    for query in search_queries:
        documents_for_query = get_all_documents(query)
        if documents_for_query:
            all_retrieved_docs.append(documents_for_query)

    return all_retrieved_docs


def run_query_feedback(search_queries: List[str], **kwargs):
    """
    Runs searches based on text queries and incident numbers separately,
    then returns the aggregated results.
    """
    all_retrieved_feedback = []

    for query in search_queries:
        documents_for_query = get_all_feedbacks(query)
        if documents_for_query:
            all_retrieved_feedback.append(documents_for_query)

    return all_retrieved_feedback
