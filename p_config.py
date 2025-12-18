import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    # paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCS_DIR = os.path.join(BASE_DIR, 'docs')

    # Qdrant
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    COLLECTION_NAME = 'simple_vector_db'
    VECTOR_DIM = 3072

    # models
    EMBEDDING_MODEL = 'text-embedding-3-large'
    LLM_MODEL = 'gpt-4.1'

settings = Config()