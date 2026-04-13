from fastapi import APIRouter, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langsmith import traceable
from dotenv import load_dotenv
import tempfile
import os
import uuid

load_dotenv()

router = APIRouter()

# Note: We use psycopg here because vector_store.add_documents is a sync function
SYNC_DB_URL = "postgresql+psycopg://admin:password@localhost:5432/rag_db"

# 1. FIXED: We MUST define the vector store here so the router knows where the DB is!
vector_store = PGVector(
    embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    collection_name="user_documents",
    connection=SYNC_DB_URL,
    use_jsonb=True,
)


@router.post("/upload")
@traceable(name="Document Ingestion Pipeline")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    print(f"[{user_id}] Uploading document: {file.filename} of size {file.size}")
    
    # 2. FIXED INDENTATION: Save the uploaded file temporarily, then cleanly close it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    # --- Notice we are intentionally NOT indented anymore! ---
    
    # 3. Extract Text
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # 4. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # 5. MULTI-TENANCY METADATA: Stamp every chunk with the user_id!
    for doc in split_docs:
        doc.metadata = {"user_id": user_id, "filename": file.filename}

    if not split_docs:
        os.remove(tmp_file_path)
        return {"error": "Failed to extract any text from this PDF. Make sure it is not just a scanned image!"}

    # 6. Embed the chunks and save to PostgreSQL with manual IDs to prevent DB crash!
    chunk_ids = [str(uuid.uuid4()) for _ in split_docs]
    vector_store.add_documents(split_docs, ids=chunk_ids)
    
    # 7. Clean up using the correct variable name
    os.remove(tmp_file_path)
    
    return {"message": f"Successfully processed {len(split_docs)} chunks for {user_id}"}
