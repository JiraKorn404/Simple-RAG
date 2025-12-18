from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings
from config import settings

vector_name = 'text_embedding'

def get_qdrant_client():
    return QdrantClient(url=settings.QDRANT_URL)

def get_vector_store():
    client = get_qdrant_client()

    dense_embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name='Qdrant/bm25')

    return QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=vector_name
    )

def init_collection(force_recreate: bool = False):
    client = get_qdrant_client()

    if force_recreate and client.collection_exists(settings.COLLECTION_NAME):
        print(f'Deleting existing collection: {settings.COLLECTION_NAME}')
        client.delete_collection(settings.COLLECTION_NAME)

    if not client.collection_exists(settings.COLLECTION_NAME):
        print(f'Create collection: {settings.COLLECTION_NAME}')
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                vector_name: models.VectorParams(
                size=settings.VECTOR_DIM,
                distance=Distance.COSINE
                ),
            },
            
            sparse_vectors_config={
                'langchain-sparse': SparseVectorParams()
            }
        )
    else:
        print(f'Collection {settings.COLLECTION_NAME} ready.')