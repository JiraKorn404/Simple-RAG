from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from flashrank import Ranker, RerankRequest
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from app.database_setup.qdrant_setup import get_vector_store
from app.config.config import settings

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

    RELEVANT_THRESHOLD = 0.3
    top_result = reranked_results[0] if reranked_results else None

    # if not top_result or top_result['score'] < RELEVANT_THRESHOLD:
    #     return 'No sufficiently relevant documents found in the knowledge base for this query.'

    context = ''
    for hit in reranked_results[:5]:
        doc_meta = hit["meta"]
        content = hit["text"]
        score = hit["score"]
        
        filename = doc_meta.get('filename', 'Unknown')
        page = doc_meta.get('page_number', 'N/A')
        context += f"\n[Source: {filename}, Page: {page}, Relevance: {score:.4f}]\n{content}\n"

    return context

# 2. Setup agent
def build_agent():
    # llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0)
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    tools = [retrieve_knowledge]
    memory = MemorySaver()

    system_prompt = """You are a helpful AI assistant with access to a knowledge base through the retrieve_knowledge tool.
    IMPORTANT RULES:
    1. You MUST use the retrieve_knowledge tool for ANY question that could be answered from documents.
    2. If the tool returns "No relevant documents found" or "No sufficiently relevant documents found", you MUST tell the user that you cannot answer their question based on the available documents.
    3. NEVER use your general knowledge to answer questions about topics that should be in the documents.
    4. Only answer from the retrieved context. If the context doesn't contain the answer, say so.
    5. Always cite your sources when answering (filename and page number).

    If you cannot find relevant information in the knowledge base, respond with:
    "I apologize, but I couldn't find relevant information about that in the available documents. Please ask questions related to the uploaded documents."
    """

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
    )
    return agent_executor