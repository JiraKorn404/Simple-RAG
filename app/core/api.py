import uuid
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from langchain_core.messages import HumanMessage
from datetime import datetime, timezone
from pypdf import PdfReader

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse, FileUploadResponse, CheckFiles, DeleteFileResponse
from app.core.agent import build_agent
from app.database_setup.mongodb_setup import lifespan, mongodb_delete_data
import app.database_setup.mongodb_setup as mongo
from app.database_setup.qdrant_setup import qdrant_delete_data
from app.utils.text_processing import ingest_documents

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

@app.post('/upload_file', response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...)
):
    try:
        # read file content
        file_content = await file.read()
        file_name = file.filename

        # upload raw file to MongoDB
        file_id_oid = mongo.fs.put(file_content, filename=file_name) # THIS <----------------------------------------------
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
                mongo.fs.delete(file_id)
                raise HTTPException(status_code=400, detail=f'Failed to process PDF: {str(e)}')
        else:
            raise HTTPException(status_code=400, detail='Unsupported file type (needs to be PDF format)')
        
        # ingest into Qdrant
        try:
            ingest_documents(text_content, file_id, file_name)
        except Exception as e:
            mongo.fs.delete(file_id)
            raise HTTPException(status_code=500, detail=f'Vector ingestion failed: {str(e)}')

        return FileUploadResponse(
            status='success',
            file_id=file_id,
            filename=file_name,
            uploaded_at=str(uploaded_at),
            message='File stored in MongoDB and Qdrant successfully.'
        )
    
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

    for grid_out in mongo.fs.find():
        files.append(
            CheckFiles(
                filename=grid_out.filename,
                file_id=str(grid_out._id),
                size_kb=round(grid_out.length / 1024, 2),
                uploaded_at=str(grid_out.uploadDate)
            )
        )
    return {'count': len(files), 'files': files}

@app.delete('/delete/{file_id}')
def delete_file(file_id: str):
    try:
        mongo_deleted = mongodb_delete_data(file_id)
        if not mongo_deleted:
            print(f"File {file_id} not found in MongoDB or invalid ID, checking Qdrant...")
        
        qdrant_deleted_bool = False

        try:
            qdrant_result = qdrant_delete_data(file_id)
            if qdrant_result and getattr(qdrant_result, 'status', '') == 'completed':
                qdrant_deleted_bool = True
        except:
            if "doesn't exist" in str(e) or "404" in str(e):
                print(f"Qdrant collection missing, skipping: {e}")
            else:
                print(f"Qdrant error: {e}")

        if not mongo_deleted and not qdrant_deleted_bool:
             raise HTTPException(status_code=404, detail='File not found in storage.')
        
        return DeleteFileResponse(
            message='File deletion processed.',
            file_id=file_id,
            mongodb_deleted=mongo_deleted,     # Passes True/False
            qdrant_deleted=qdrant_deleted_bool # Passes True/False
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f'Delete error: {e}')
        raise HTTPException(status_code=500, detail=str(e))