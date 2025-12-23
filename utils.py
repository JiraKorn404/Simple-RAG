import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import get_vector_store, init_collection

def generate_chunk_id(
        content: str,
        file_id: str,
) -> str:
    raw_str = f'{file_id}_{content}'
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def ingest_documents(
        text_content: str,
        file_id: str,
        file_name: str
):
    # 1. Initialize DB
    init_collection(force_recreate=False)
    vector_store = get_vector_store()

    # 2. Load and split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.create_documents(
        texts=[text_content],
        metadatas=[{
            'filename': file_name,
            'file_id': file_id,
        }]
    )

    processed_chunks = []
    ids = []

    for chunk in chunks:
        chunk_id = generate_chunk_id(chunk.page_content, file_id)
            
        chunk.metadata['filename'] = file_name
        chunk.metadata['file_id'] = file_id
        chunk.metadata['chunk_id'] = chunk_id

        processed_chunks.append(chunk)
        ids.append(chunk_id)

    # 3. Upsert to Qdrant
    if processed_chunks:
        # print(f'Upserting {len(processed_chunks)} chunks...')
        vector_store.add_documents(documents=processed_chunks, ids=ids)
        # print('Ingestion complete.')
        return {
            'status': 'Ingestion complete.'
        }
    else:
        # print('No documents to ingest.')
        return {
            'No documents to ingest.'
        }