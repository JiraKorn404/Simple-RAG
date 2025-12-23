import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    # paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCS_DIR = os.path.join(BASE_DIR, 'docs')

    # MongoDB
    MONGODB_URL = os.getenv('MONGODB_URL')
    MONGODB_NAME = os.getenv('MONGODB_NAME')

    # Qdrant
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    COLLECTION_NAME = 'simple_vector_db'
    VECTOR_NAME = 'text_embedding'
    VECTOR_DIM = 3072

    # models
    EMBEDDING_MODEL = 'text-embedding-3-large'
    LLM_MODEL = 'gpt-4.1'
    LLM_API_KEY = os.getenv('OPENAI_API_KEY')

settings = Config()