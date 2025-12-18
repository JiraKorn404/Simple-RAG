import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from p_agent import build_agent
from p_ingestion import ingest_documents

app = FastAPI(title='RAG AGENT', version='1.0')

agent_app = build_agent()

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    # sources: List[str] = []

class IngestResponse(BaseModel):
    status: str
    message: str

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
    
@app.post('/ingest', response_model=IngestResponse)
def ingest_endpoint():
    """
    Triggers the document ingestion process.
    This replaces the need to run p_ingestion.py manually.
    """
    try:
        ingest_documents()
        return IngestResponse(
            status='success',
            message='Document ingested and vector database updated'
        )
    except Exception as e:
        print(f'Ingestion error: {e}')
        raise HTTPException(status_code=500, detail=str(e))