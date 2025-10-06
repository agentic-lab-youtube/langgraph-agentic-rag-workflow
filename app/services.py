import re
import time
import logging
from datetime import datetime
import pandas as pd
import os
from langchain.schema import Document

# --- Start of Changes ---
from langchain_community.vectorstores import Chroma

from typing import Optional, List

from llms import embeddings


logger = logging.getLogger(__name__)

# Define a constant for the ChromaDB path
CHROMA_DB_PATH = "chroma_vector_db"


def create_documents_from_excel(excel_path):
    """Extract data from Excel and create LangChain Documents"""

    # Load Excel file
    try:
        df = pd.read_excel(excel_path)
        print("Excel file loaded successfully.")
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return []

    # Clean dataframe
    df = df.dropna(how="all")

    documents = []

    for index, row in df.iterrows():
        content_lines = []

        # Extract all fields (same as your original code)
        incident_number = str(row["incident_number"]).strip().lower()
        location = str(row["location"]).strip().lower()
        title = str(row["title"]).strip()
        description = str(row["description"]).strip().lower()
        priority = str(row["priority"]).strip().lower()
        caller = str(row["caller"]).strip().lower()
        assignment_group = str(row["assignment_group"]).strip().lower()
        assigned_to = str(row["assigned_to"]).strip().lower()
        state = str(row["state"]).strip().lower()
        created = str(row["created"])
        updated = str(row["updated"])
        close_notes = str(row["close_notes"]).strip().lower()
        resolved_time = str(row["resolved_time"])
        updated_by = str(row["updated_by"])
        work_notes = str(row["work_notes"]).strip().lower()
        category = str(row["category"]).strip().lower()
        additional_comments = str(row["additional_comments"]).strip().lower()

        # Build content (same as your original code)
        content_lines.append(f"INCIDENT_NUMBER: {incident_number}")

        if pd.notna(title):
            content_lines.append(
                f"HAS_REPORTED_ISSUE: For incident number:{incident_number} -> Reported Issue: {title}"
            )
        if pd.notna(description):
            content_lines.append(
                f"HAS_DESCRIPTION: For incident number:{incident_number} -> Description: {description}"
            )
        if pd.notna(location):
            content_lines.append(
                f"HAS_LOCATION: For incident number: {incident_number} -> Location: {location}"
            )
        if pd.notna(close_notes):
            content_lines.append(
                f"HAS_CLOSE_NOTES: For incident number:{incident_number} -> Close Notes: {close_notes}"
            )
        if pd.notna(priority):
            content_lines.append(
                f"HAS_PRIORITY: For incident number:{incident_number} -> Priority: {priority}"
            )
        if pd.notna(caller):
            content_lines.append(
                f"HAS_CALLER: For incident number:{incident_number} -> Caller: {caller}"
            )
        if pd.notna(assignment_group):
            content_lines.append(
                f"HAS_ASSIGNMENT_GROUP: For incident number:{incident_number} -> Assignment_Group: {assignment_group}"
            )
        if pd.notna(assigned_to):
            content_lines.append(
                f"HAS_ASSIGNED_TO: For incident number:{incident_number} -> Assigned_To: {assigned_to}"
            )
        if pd.notna(state):
            content_lines.append(
                f"HAS_STATE:For incident number: {incident_number} -> State: {state}"
            )
        if pd.notna(created):
            content_lines.append(
                f"HAS_CREATED_DATE:For incident number: {incident_number} -> Created on: {created}"
            )
        if pd.notna(updated):
            content_lines.append(
                f"HAS_UPDATED_DATE:For incident number: {incident_number} -> Updated on: {updated}"
            )
        if pd.notna(resolved_time):
            content_lines.append(
                f"HAS_RESOLVED_TIME: For incident number:{incident_number} -> Resolved time: {resolved_time}"
            )
        if pd.notna(updated_by):
            content_lines.append(
                f"HAS_UPDATED_BY: For incident number:{incident_number} -> Updated by: {updated_by}"
            )
        if pd.notna(work_notes):
            content_lines.append(
                f"HAS_WORK_NOTES: For incident number:{incident_number} -> Work notes: {work_notes}"
            )
        if pd.notna(category):
            content_lines.append(
                f"HAS_CATEGORY: For incident number:{incident_number} -> Category: {category}"
            )
        if pd.notna(additional_comments):
            content_lines.append(
                f"HAS_ADDITIONAL_COMMENTS: For incident number:{incident_number} -> Additional comments: {additional_comments}"
            )

        content = "\n".join(content_lines) + "\n"

        # Create metadata with all document values
        metadata = {
            "index": index,
            "incident_number": incident_number,
            "location": location,
            "title": title,
            "description": description,
            "priority": priority,
            "caller": caller,
            "assignment_group": assignment_group,
            "assigned_to": assigned_to,
            "state": state,
            "created": created,
            "updated": updated,
            "close_notes": close_notes,
            "resolved_time": resolved_time,
            "updated_by": updated_by,
            "work_notes": work_notes,
            "category": category,
            "additional_comments": additional_comments,
            "type": "doc",
        }

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    print(f"Created {len(documents)} documents from Excel file")
    return documents


# --- Start of Changes: Replaced FAISS with ChromaDB ---
def create_and_save_vector_db(documents, vector_db_path=CHROMA_DB_PATH):
    """Create or update a persistent ChromaDB vector database."""
    if not os.path.exists(vector_db_path):
        if not documents:
            print("No documents provided to create or update the vector database.")
            return None

        # Instantiate Chroma with a persistent directory.
        # This will create the directory if it doesn't exist, or load it if it does.
        vector_store = Chroma(
            persist_directory=vector_db_path, embedding_function=embeddings
        )

        batch_size = 50
        delay = 1.0

        print(f"Processing {len(documents)} documents in batches of {batch_size}...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size

            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)..."
            )

            try:
                # Add documents to the Chroma collection. This handles both creation and updates.
                vector_store.add_documents(batch)
                print(f"Batch {batch_num} completed successfully")

            except Exception as e:
                print(f"Error processing batch {batch_num}: {str(e)}")

        # Chroma automatically persists changes to the directory, so no explicit save is needed.
        print(f"Vector database at {vector_db_path} is up to date.")
        return vector_store


def load_vector_db(vector_db_path=CHROMA_DB_PATH):
    """Load the ChromaDB vector database from a persistent directory."""

    try:
        # Simply instantiate Chroma with the path and embedding function to load it.
        vector_store = Chroma(
            persist_directory=vector_db_path, embedding_function=embeddings
        )
        print("ChromaDB vector database loaded successfully!")
        return vector_store
    except Exception as e:
        print(f"Error loading ChromaDB vector database: {e}")
        return None


def get_all_documents(query: str, incident_number: Optional[str] = None):
    """
    Get all documents from the vector database, with an optional filter.
    """
    vector_store = load_vector_db(CHROMA_DB_PATH)
    if not vector_store:
        return None

    k_value = 2
    question = query

    # Build a list of conditions for the filter
    conditions = [{"type": {"$eq": "doc"}}]

    if incident_number:
        question = incident_number
        conditions.append({"incident_number": {"$eq": incident_number.lower()}})
        k_value = 8

    # Fix: Chroma expects dict, not list
    if len(conditions) == 1:
        filter_criteria = conditions[0]  # ✅ dict
    else:
        filter_criteria = {"$and": conditions}  # ✅ dict with $and

    all_docs = vector_store.similarity_search(
        question,
        k=k_value,
        filter=filter_criteria,
    )
    return all_docs


def get_all_feedbacks(query: str) -> Optional[List[Document]]:
    """
    Get all documents of type 'feedback' from the vector database.
    """
    vector_store = load_vector_db(CHROMA_DB_PATH)
    if not vector_store:
        print("Vector database not found.")
        return None

    # Single predicate can be passed directly; $eq is fine but not required for equality
    filter_criteria = {"type": {"$eq": "feedback"}}

    all_feedbacks = vector_store.similarity_search(
        query,
        k=2,
        filter=filter_criteria,
    )

    # print(f"Found {len(all_feedbacks)} feedback documents.")
    return all_feedbacks
