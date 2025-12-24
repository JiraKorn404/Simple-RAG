from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings

from app.config.config import settings

vector_name = settings.VECTOR_NAME

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

def qdrant_delete_data(
        file_id: str
):
    file_id_key = 'metadata.file_id'
    client = get_qdrant_client()

    try:
        result = client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=file_id_key,
                            match=models.MatchValue(value=file_id)
                        )
                    ]
                )
            )
        )
        return result
    except Exception as e:
        print(f'Error deleting from Qdrant: {e}')
        return None