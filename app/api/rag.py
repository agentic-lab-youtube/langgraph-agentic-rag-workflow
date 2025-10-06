from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from datetime import datetime, timezone
from app.services import load_vector_db
from app.graph.workflow import final_graph
import uuid

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: str = None


@router.post("/search_vector_documents")
async def search_vector_documents(request: QueryRequest):
    query = request.query

    if request.session_id:
        thread_id = request.session_id
        print(f"Continuing conversation with thread_id: {thread_id}")
    else:
        thread_id = str(uuid.uuid4())
        print(f"Starting new conversation with thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    inputs = {"messages": [HumanMessage(content=query)], "query": query}

    final_state = final_graph.invoke(inputs, config)

    response = {
        "result": final_state["answer"],
        "documents": final_state["metadata"],
        "session_id": thread_id,
    }

    return response


CHROMA_DB_PATH = "chroma_vector_db"


class IncidentFeedbackRequest(BaseModel):
    user_query: str = Field(..., description="The actual query combined.")
    feedback: str = Field(..., description="The Actual feedback from user.")
    content: str = Field(..., description="The combined content.")


def archive_incident_feedback(request: IncidentFeedbackRequest):
    """
    Processes incident feedback, creates a new 'feedback' document,
    and adds it to the existing vector store.
    """
    # 1. Combine the query, content, and feedback into a single sentence
    page_content = (
        f"A user provided feedback on an incident query. "
        f"User's Query: '{request.user_query}'. "
        f"Provided Content: '{request.content}'. "
        f"User's Feedback: '{request.feedback}'."
    )

    # 2. Create the metadata for the new feedback document
    metadata = {
        "type": "feedback",
        "feedback_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # 3. Create the LangChain Document
    feedback_document = Document(page_content=page_content, metadata=metadata)

    # 4. Load, update, and save the vector database
    print("Loading existing vector database to add incident feedback...")
    vector_store = load_vector_db(CHROMA_DB_PATH)
    if vector_store is None:
        raise FileNotFoundError(
            "Could not load the vector database. Ensure it has been created."
        )

    print("Adding new incident feedback document...")
    vector_store.add_documents([feedback_document], ids=[str(uuid.uuid4())])

    vector_store.persist()

    print("Incident feedback document successfully processed and stored.")


@router.post("/incident_feedback")
def submit_incident_feedback(request: IncidentFeedbackRequest):
    """
    Receives incident-specific feedback and archives it in the vector store.
    """
    try:
        archive_incident_feedback(request)
        return {
            "status": "success",
            "message": "Incident feedback has been successfully archived. Thank you!",
        }
    except FileNotFoundError as e:
        # Handle cases where the vector DB doesn't exist
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"Failed to process incident feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while archiving the feedback.",
        )
