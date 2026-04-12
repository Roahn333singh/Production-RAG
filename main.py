from fastapi import FastAPI
from agent.rag_state import app as rag_app
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from api.routers.document import router as document_router
app=FastAPI(
    title="LangGraph RAG API",
    description="A simple RAG API using LangGraph",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    user_id: str
    query: str

# Tell FastAPI to include our upload routes
app.include_router(document_router, prefix="/docs", tags=["Documents"])

@app.post("/chat")
def chat(request:ChatRequest):
    test_graph={
        "user_id":request.user_id,
        "messages":[HumanMessage(content=request.query)]
    }
    config = {
        "configurable": {
            "thread_id": request.user_id  # Unique identifier to track workflow execution
        }
    }
    result=rag_app.invoke(test_graph,config=config)
    return {"response":result["messages"][-1].content, "Snapshot":rag_app.get_state(config=config)}
