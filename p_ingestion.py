import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from p_database import get_vector_store, init_collection
from p_config import settings

def generate_chunk_id(
        content: str,
        source: str,
        page: int
) -> str:
    raw_str = f'{source}_{page}_{content}'
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def ingest_documents():
    # 1. Initialize DB
    init_collection(force_recreate=True)
    vector_store = get_vector_store()

    # 2. Load and split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    processed_chunks = []
    ids = []

    files = [f for f in os.listdir(settings.DOCS_DIR) if f.endswith('.pdf')]
    print(f'Found {len(files)} PDF documents.')

    for file in files:
        file_path = os.path.join(settings.DOCS_DIR, file)
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        chunks = text_splitter.split_documents(pages)
        for chunk in chunks:
            source = file
            page_num = chunk.metadata.get('page', 0)
            chunk_id = generate_chunk_id(chunk.page_content, source, page_num)
            
            chunk.metadata['filename'] = source
            chunk.metadata['page_number'] = page_num
            chunk.metadata['chunk_id'] = chunk_id

            processed_chunks.append(chunk)
            ids.append(chunk_id)

    # 3. Upsert to vector DB
    if processed_chunks:
        print(f'Upserting {len(processed_chunks)} chunks...')
        vector_store.add_documents(documents=processed_chunks, ids=ids)
        print('Ingestion complete.')
    else:
        print('No documents to ingest.')

if __name__ == '__main__':
    ingest_documents()