import uuid
import io
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader

from schemas import ChatRequest, ChatResponse, IngestResponse, FileUploadResponse
from agent import build_agent
from ingestion import ingest_documents
from database import lifespan, qdrant_delete_data, mongodb_delete_data
import database
from utils import ingest_documents

app = FastAPI(lifespan=lifespan, title='RAG AGENT', version='1.0')

agent_app = build_agent()

@app.get('/')
def health_check():
    """Simple health check to verify server is running."""
    return {'status': 'running'}

@app.post('/chat', response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Sends a message to the RAG agent. 
    If thread_id is provided, continues that session.
    Otherwise, starts a new session.
    """
    try:
        thread_id = payload.thread_id or str(uuid.uuid4())
        config = {'configurable': {'thread_id': thread_id}}

        input_message = HumanMessage(content=payload.message)

        final_state = agent_app.invoke(
            {'messages': [input_message]},
            config=config
        )

        messages = final_state.get('messages', [])

        if messages and messages[-1].type == 'ai':
            response_text = messages[-1].content
        else:
            response_text = "I'm sorry, I couldn't generate a response."

        return ChatResponse(
            response=response_text,
            thread_id=thread_id
        )
    
    except Exception as e:
        print(f'Error processing chat request: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post('/ingestion', response_model=IngestResponse)
# def ingest_endpoint():
#     """
#     Triggers the document ingestion process.
#     This replaces the need to run p_ingestion.py manually.
#     """
#     try:
#         ingest_documents()
#         return IngestResponse(
#             status='success',
#             message='Document ingested and vector database updated'
#         )
#     except Exception as e:
#         print(f'Ingestion error: {e}')
#         raise HTTPException(status_code=500, detail=str(e))
    
# @app.post('/upload_mongodb')
# def upload_mongodb(
#     file: UploadFile = File(...) # open for file upload
# ):
#     """
#     Upload pdf into data pipeline
#     File will be stored MongoDB as a raw file and vector data inside Qdrant
#     upload --> MongoDB
#            --> vector embedding --> Qdrant
#     Using the same file_id both in MongoDB and Qdrant
#     """
#     # get file id
#     try:
#         file_id = database.fs.put(file.file, filename=file.filename)
#         uploaded_at = datetime.now(timezone.utc)
#         return {
#             'file_id': str(file_id),
#             'uploaded_at': str(uploaded_at)
#         }
#     except Exception as e:
#         print(f'Ingestion error: {e}')
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post('/upload_qdrant')
# async def upload_qdrant(
#     file: UploadFile = File(...)
# ):
#     """
#     Upload pdf into data pipeline
#     File will be stored MongoDB as a raw file and vector data inside Qdrant
#     upload --> MongoDB
#            --> vector embedding --> Qdrant
#     Using the same file_id both in MongoDB and Qdrant
#     """
#     file_content = ''
#     try:
#         file_content = await file.read()
#         file_name = file.filename
#         file_id = str(uuid.uuid4())

#         pdf_reader = PdfReader(io.BytesIO(file_content))
#         text_content = ''
#         for page in pdf_reader.pages:
#             text_content += page.extract_text() + '\n'
#         if not text_content.strip():
#             raise HTTPException(status_code=400, detail='PDF is empty or unreadable.')
#         else:
#             response = ingest_documents(text_content, file_id, file_name)
#             return response

#     except Exception as e:
#         print(f'Ingestion error: {e}')
#         raise HTTPException(status_code=500, detail=str(e))

@app.post('/upload_file')
async def upload_file(
    file: UploadFile = File(...)
):
    try:
        # read file content
        file_content = await file.read()
        file_name = file.filename

        # upload raw file to MongoDB
        file_id_oid = database.fs.put(file_content, file_name=file_name)
        file_id = str(file_id_oid)
        uploaded_at = datetime.now(timezone.utc)

        # process for Qdrant
        text_content = ''  
        if file_name.lower().endswith('.pdf'):
            try:
                pdf_reader = PdfReader(io.BytesIO(file_content))
                for page in pdf_reader.pages:
                    extract = page.extract_text()
                    if extract:
                        text_content += extract + '\n'
            except Exception as e:
                database.fs.delete(file_id)
                raise HTTPException(status_code=400, detail=f'Failed to process PDF: {str(e)}')
        else:
            raise HTTPException(status_code=400, detail='Unsupported file type (needs to be PDF format)')
        
        # ingest into Qdrant
        try:
            ingest_documents(text_content, file_id, file_name)
        except Exception as e:
            database.fs.delete(file_id)
            raise HTTPException(status_code=500, detail=f'Vector ingestion failed: {str(e)}')

        return {
            'status': 'success',
            'file_id': file_id,
            'filename': file_name,
            'uploaded_at': str(uploaded_at),
            'message': 'File stored in MongoDB and Qdrant successfully.'
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f'Upload error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/check_files/')
def list_files():
    """
    Check files inside MongoDB
    """
    files = []

    for grid_out in database.fs.find():
        files.append(
            {
                'filename': grid_out.filename,
                'file_id': str(grid_out._id),
                'size_kb': round(grid_out.length / 1024, 2),
                'upload_date': grid_out.uploadDate
            }
        )
    return {'count': len(files), 'files': files}

@app.delete('/delete/{file_id}')
def delete_file(file_id: str):
    try:
        mongo_deleted = mongodb_delete_data(file_id)
        if not mongodb_delete_data:
            print(f"File {file_id} not found in MongoDB or invalid ID, checking Qdrant...")
        
        qdrant_deleted = qdrant_delete_data(file_id)
        qdrant_status = qdrant_deleted.status if qdrant_deleted else 'failed'

        if not mongo_deleted and qdrant_status != "completed":
             raise HTTPException(status_code=404, detail='File not found in storage.')
        
        return {
            'message': 'File deletion processed.',
            'file_id': file_id,
            'mongodn_deleted': mongo_deleted,
            'qdrant_deleted': qdrant_status
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f'Delete error: {e}')
        raise HTTPException(status_code=500, detail=str(e))