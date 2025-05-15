from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Imports from backend modules ===
from backend.chatbot.core import (
    run_chat,
    call_character_bot,
    ChatState
)
from backend.chatbot.documents import (
    load_scene_chunks,
    create_documents
)
from backend.chatbot.vectorstore import initialize_vectorstore
from backend.chatbot.prompt import create_prompt_template

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# === Initialize chatbot components ===
print("🔧 Initializing chatbot components...")

# Load and prepare documents
scene_chunks = load_scene_chunks("data/scene_chunks_with_emotions.jsonl")
documents = create_documents(scene_chunks)

# Initialize vectorstore and LLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = initialize_vectorstore(documents, embedding_model, "data/vector_databases/scene_db_with_emotions")

llm = ChatGroq(model="llama3-8b-8192", temperature=0.6)
prompt_template = create_prompt_template()

# Setup LangGraph workflow
graph = StateGraph(ChatState)
graph.add_node("model", lambda state: call_character_bot(state, llm, prompt_template, vectorstore))
graph.set_entry_point("model")
workflow = graph.compile(checkpointer=MemorySaver())

print("✅ Chatbot backend initialized.")

# === FastAPI app setup ===
app = FastAPI()

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Models ===
class Message(BaseModel):
    type: str  # "human" or "ai"
    content: str

class ChatRequest(BaseModel):
    query: str
    character: str
    user_name: str
    memory: List[Message] = []

# === Endpoint ===
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    print("💬 /chat endpoint hit")
    response = run_chat(
        user_name=request.user_name,
        character=request.character,
        query=request.query,
        memory=request.memory,
        workflow=workflow,
        llm=llm,
        prompt_template=prompt_template,
        vectorstore=vectorstore
    )
    return response
