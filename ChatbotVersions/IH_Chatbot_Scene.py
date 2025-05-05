import os
import json
import textwrap
from typing import Sequence
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -------------------------------
# CONFIGURATION
# -------------------------------
character = "Pam"
thread_id = f"{character.lower()}-chat-thread"
messages = []

# -------------------------------
# LOAD DATA
# -------------------------------
load_dotenv()

scene_chunks_path = "../data/scene_chunks.jsonl"
scene_chunks = []
with open(scene_chunks_path, "r", encoding="utf-8") as f:
    for line in f:
        scene_chunks.append(json.loads(line))

documents = [
    Document(
        page_content=scene["text"],
        metadata={
            "scene_id": scene.get("scene_id", idx),
            "speakers": scene.get("speakers", [])
        }
    )
    for idx, scene in enumerate(scene_chunks)
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore_path = "../data/vector_databases/scene_db"

if not os.path.exists(vectorstore_path):
    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(
        documents=documents, embedding=embedding_model)
    vectorstore.save_local(vectorstore_path)
else:
    print("Loading existing FAISS vectorstore...")
    vectorstore = FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

# FILTER BY CHARACTER

def get_character_lines(text: str, character: str) -> str:
    return "\n".join([
        line for line in text.split("\n")
        if line.startswith(f"{character}:")
    ])


def get_relevant_docs(character: str, query: str):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = base_retriever.invoke(query)  # use new .invoke method
    filtered = [
        doc for doc in docs if character in doc.metadata.get("speakers", [])]
    for doc in filtered:
        doc.page_content = get_character_lines(doc.page_content, character)
    return filtered


# -------------------------------
# LLM + PROMPT + GRAPH
# -------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question", "character", "user_name"],
    template="""
You are {character} from the TV series THE OFFICE (US).
You are speaking to a user named {user_name}.
Stay in character and respond with your unique tone, quirks, and style.
Do not break character or refer to the real world.

Use the following context from the show to answer:
------------------------
{context}

Question: {question}
Answer:"""
)


class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    character: str
    query: str


llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)


def call_character_bot(state: ChatState) -> ChatState:
    query = state["query"]
    character = state["character"]
    docs = get_relevant_docs(character, query)
    context = "\n\n".join(doc.page_content for doc in docs)

    state["messages"].append(HumanMessage(content=query))

    full_prompt = prompt_template.format(
        context=context,
        question=query,
        character=character,
        user_name=user_name
    )
    response = llm.invoke([HumanMessage(content=full_prompt)])
    state["messages"].append(response)
    return {"messages": state["messages"]}


graph = StateGraph(state_schema=ChatState)
graph.add_node("model", call_character_bot)
graph.set_entry_point("model")

memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)

# -------------------------------
# START CHAT
# -------------------------------
print("\nPam: Hi! I'm Pam Beesly, the receptionist at Dunder Mifflin.")
user_name = input("Pam: What's your name? ").strip().title()
print(f"Pam: Nice to meet you, {user_name}! How can I help you today?\n")

print(f"✅ You're now chatting with {character}!")
print("Type 'exit' to quit, or '/switch <Character>' to change who you're talking to.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        memory.delete_thread(thread_id)
        print("👋 Goodbye!")
        break

    if user_input.lower() == "/summary":
        print("🧠 Memory so far:")
        for msg in messages:
            print(f"- {msg.type.capitalize()}: {msg.content}")
        continue

    if user_input.lower().startswith("/switch "):
        memory.delete_thread(thread_id)
        new_char = user_input[8:].strip().title()
        character = new_char
        thread_id = f"{character.lower()}-chat-thread"
        messages = []
        print(f"\n🎭 Switching to {character}...")
        print(f"✅ You're now chatting with {character}!\n")
        continue

    result = workflow.invoke(
        {"messages": messages, "query": user_input, "character": character},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]
    wrapped = textwrap.fill(messages[-1].content, width=80)
    print(f"{character}:\n{wrapped}\n")
