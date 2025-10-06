from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.api.rag import router
from app.services import create_documents_from_excel, create_and_save_vector_db
from app.config import EXCEL_PATH
import uvicorn

rag_app = FastAPI()

# CORS middleware
rag_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_app.include_router(router, prefix="/rag-api")

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # excel_path = os.path.join("data", "Incident_Data.xlsx")
    documents = create_documents_from_excel(EXCEL_PATH)

    if not documents:
        print("No documents created. Exiting.")

    # Create and save vector database
    vector_store = create_and_save_vector_db(documents)

    print("Vector database creation completed!")

    uvicorn.run(rag_app, host="0.0.0.0", port=8010)
