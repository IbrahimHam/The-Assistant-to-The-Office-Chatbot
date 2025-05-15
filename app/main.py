from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add the ChatbotVersions directory to the path
chatbot_versions_path = Path(__file__).resolve().parent.parent / "ChatbotVersions"
sys.path.append(str(chatbot_versions_path))

# Load env variables
load_dotenv()

# === Imports from chatbot file ===
from office_chatbot_with_emotions import (
    run_chat,
    load_scene_chunks,
    create_documents,
    initialize_vectorstore,
    create_prompt_template,
    call_character_bot,
    ChatState
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# === Initialize everything ONCE ===
print("🔧 Initializing components...")
scene_chunks = load_scene_chunks("data/scene_chunks_with_emotions.jsonl")
documents = create_documents(scene_chunks)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = initialize_vectorstore(documents, embedding_model, "data/vector_databases/scene_db_with_emotions")

llm = ChatGroq(model="llama3-8b-8192", temperature=0.6)
prompt_template = create_prompt_template()

# Setup LangGraph workflow
graph = StateGraph(ChatState)
graph.add_node("model", lambda state: call_character_bot(
    state, llm, prompt_template, vectorstore))
graph.set_entry_point("model")
workflow = graph.compile(checkpointer=MemorySaver())

print("✅ Backend initialized with real prompt, LLM, and vectorstore")

# === FastAPI app setup ===
app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request models ===
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
