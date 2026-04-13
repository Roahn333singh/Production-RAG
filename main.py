from fastapi import FastAPI
from agent.rag_state import app as rag_app
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from api.routers.document import router as document_router
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessageChunk, RemoveMessage

app=FastAPI(
    title="LangGraph RAG API",
    description="A simple RAG API using LangGraph",
    version="1.0.0"
)

from enum import Enum

class ModelChoice(str, Enum):
    gemini = "gemini-2.5-flash"
    llama = "llama-3.1-8b-instant"

class ChatRequest(BaseModel):
    user_id: str
    query: str
    llm_model: ModelChoice = ModelChoice.gemini

# Tell FastAPI to include our upload routes
app.include_router(document_router, prefix="/docs", tags=["Documents"])

@app.post("/chat")
def chat(request:ChatRequest):
    test_graph={
        "user_id":request.user_id,
        "llm_model": request.llm_model,
        "messages":[HumanMessage(content=request.query)]
    }
    config = {
        "configurable": {
            "thread_id": request.user_id  # Unique identifier to track workflow execution
        }
    }
    result=rag_app.invoke(test_graph,config=config)
    return {"response":result["messages"][-1].content, "Snapshot":rag_app.get_state(config=config)}

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    config = {"configurable": {"thread_id": request.user_id}}
    test_graph = {
        "user_id": request.user_id, 
        "llm_model": request.llm_model,
        "messages": [HumanMessage(content=request.query)]
    }

    # This is our Generator Function
        # This is our Generator Function
    def event_generator():
        for msg, metadata in rag_app.stream(test_graph, config=config, stream_mode="messages"):
            
            # THE FIX: Only stream AIMessageChunk (live chunks), ignore the final AIMessage!
            if isinstance(msg, AIMessageChunk) and msg.content and metadata.get("langgraph_node") == "generator":
                
                clean_chunk = msg.content.replace("\n", " ") 
                yield f"data: {clean_chunk}\n\n"
                
        yield "data: [DONE]\n\n"


    # Return the StreamingResponse, telling the browser to expect a live stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/chat/models")
def get_models():
    return {"models": ["gemini-2.5-flash", "llama-3.1-8b-instant"]}

@app.delete("/chat/history")
def delete_history(user_id:str):
    config = {
        "configurable": {
            "thread_id": user_id
        }
    }
    
    # LangGraph doesn't have a simple '.clear()'. Instead, we issue 'RemoveMessage' commands for all past messages
    state = rag_app.get_state(config)
    messages = state.values.get("messages", [])
    
    if messages:
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages]
        rag_app.update_state(config, {"messages": messages_to_remove})
        
    return {"message": f"History cleared successfully for {user_id}"}
    
    