from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from flashrank import Ranker, RerankRequest
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from p_database import get_vector_store
from p_config import settings

ranker = Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='./opt')

# 1. Define retrieval tool
@tool
def retrieve_knowledge(query: str):
    """
    Retrieves information from the internal knowledge base.
    Use this tool whenever the user asks a question about the uploaded documents.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=20)
    if not results:
        return 'No relevant documents found.'
    
    passages = [
        {'id': str(i), 'text': doc.page_content, 'meta': doc.metadata}
        for i, doc in enumerate(results)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)

    context = ''
    for hit in reranked_results[:5]:
        doc_meta = hit["meta"]
        content = hit["text"]
        score = hit["score"]
        
        filename = doc_meta.get('filename', 'Unknown')
        page = doc_meta.get('page_number', 'N/A')
        context += f"\n[Source: {filename}, Page: {page}, Relevance: {score:.4f}]\n{content}\n"

    # for doc in results:
    #     filename = doc.metadata.get('filename', 'Unknown')
    #     page = doc.metadata.get('page_number', 'N/A')
    #     context += f'\n[Source: {filename}, Page: {page}]\n{doc.page_content}\n'

    return context

# 2. Setup agent
def build_agent():
    # llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0)
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    tools = [retrieve_knowledge]

    memory = MemorySaver()

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory
    )
    return agent_executor