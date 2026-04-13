# Multi-Tenancy     
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict,Annotated
from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from api.routers.document import vector_store
from dotenv import load_dotenv
load_dotenv() 
import os

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

llm_groq = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,

)


class RagState(MessagesState):
    user_id: str
    llm_model: str
    documents: list[str]

def retriever_node(state:RagState):
    user_id=state["user_id"]
    query=state["messages"][-1].content
    print(f"[{user_id}] Searching vector database for: '{query}'")
    # 1. Search the Vector Database!
    # THE MAGIC: We filter by user_id so User A NEVER sees User B's private PDFs
    results = vector_store.similarity_search(
        query=query, 
        k=3, # Pull the top 3 most relevant chunks
        filter={"user_id": user_id}
    )
    # 2. Extract just the raw text from the Langchain Document objects
    document_texts = [doc.page_content for doc in results]

    # 3. Fallback if they haven't uploaded anything relevant yet
    if not document_texts:
        document_texts = ["No relevant documents found in your private database."]
        
    return {"documents": document_texts}

def generator_node(state:RagState):
    user_id=state["user_id"]
    llm_model=state.get("llm_model", "gemini-2.5-flash")
    query=state["messages"][-1].content
    content="\n\n".join(state["documents"])

    system_prompt = f"""You are a helpful assistant. Answer the user's question ONLY using this context:
    Context: {content}
    """
    print(f"[{user_id}] Generating answer using {llm_model} for: '{query}'")

    message_to_send=[HumanMessage(content=system_prompt)] + state["messages"]
    
    # 🌟 MAGIC: Dynamically select the LLM based on the API request!
    active_llm = llm_groq if llm_model == "llama-3.1-8b-instant" else llm
    
    # LangGraph context-vars automatically pipe the config down to the LLM invocation, no injection needed!
    ai_response=active_llm.invoke(message_to_send)
    return {"messages": [ai_response]}


# 1. Setup the Connection Pool
DB_URI = os.getenv("SYNC_DB_URI", "postgresql://admin:password@localhost:5432/rag_db")

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)


# 2. Tell LangGraph to use Postgres
checkpoint = PostgresSaver(pool)

graph=StateGraph(RagState)
graph.add_node("retriever",retriever_node)
graph.add_node("generator",generator_node)
graph.add_edge(START,"retriever")
graph.add_edge("retriever","generator")
graph.add_edge("generator",END)

# 3. Create the Database tables (LangGraph does this automatically!)
checkpoint.setup()

app = graph.compile(checkpointer=checkpoint)


if __name__ == "__main__":
    # 1. Define the initial state (what the user asks, and who the user is)
    config = {
        "configurable": {
            "thread_id": "1"  # Unique identifier to track workflow execution
        }
    }
    test_input = {
        "user_id": "User_123",
        "messages": [HumanMessage(content="What exactly is LangGraph?")]
    }

    print("--- STARTING AGENT ---")
    
    # 2. Invoke the graph!
    result = app.invoke(test_input,config=config)
    
    # 3. Print the final message the AI generated
    print("\n--- FINAL RESULT ---")
    print(result["messages"][-1].content)


    




