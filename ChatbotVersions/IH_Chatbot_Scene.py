import os
import json
import textwrap
import sys
import logging
from typing_extensions import TypedDict, Annotated
from uuid import uuid4

# Suppress TensorFlow oneDNN and warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

def load_scene_chunks(file_path):
    """Load scene chunks from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file containing scene chunks.

    Returns:
        list: List of scene chunk dictionaries.
    """
    scene_chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_chunks.append(json.loads(line))
    return scene_chunks

def create_documents(scene_chunks):
    """Create LangChain Documents from scene chunks.

    Args:
        scene_chunks (list): List of scene chunk dictionaries.

    Returns:
        list: List of LangChain Document objects.
    """
    return [
        Document(
            page_content=scene["text"],
            metadata={
                "scene_id": scene.get("scene_id", idx),
                "speakers": scene.get("speakers", [])
            }
        )
        for idx, scene in enumerate(scene_chunks)
    ]

def initialize_vectorstore(documents, embedding_model, vectorstore_path):
    """Initialize or load FAISS vectorstore.

    Args:
        documents (list): List of LangChain Document objects.
        embedding_model: HuggingFace embedding model instance.
        vectorstore_path (str): Path to save or load the FAISS vectorstore.

    Returns:
        FAISS: Initialized or loaded FAISS vectorstore.
    """
    if not os.path.exists(vectorstore_path):
        print("Creating FAISS vectorstore...")
        vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)
        vectorstore.save_local(vectorstore_path)
    else:
        print("Loading existing FAISS vectorstore...")
        vectorstore = FAISS.load_local(
            folder_path=vectorstore_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    return vectorstore

def get_character_lines(text: str, character: str) -> str:
    """Filter only lines spoken by a given character.

    Args:
        text (str): Text content of a scene.
        character (str): Name of the character to filter lines for.

    Returns:
        str: Filtered lines spoken by the character, joined by newlines.
    """
    return "\n".join([
        line for line in text.split("\n")
        if line.startswith(f"{character}:")
    ])

def get_relevant_docs(character: str, query: str, vectorstore):
    """Get relevant scenes for a query where the character appears.

    Args:
        character (str): Name of the character to filter scenes for.
        query (str): User query to search for relevant scenes.
        vectorstore: FAISS vectorstore instance.

    Returns:
        list: List of filtered Document objects containing character lines.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)
    filtered = [doc for doc in docs if character in doc.metadata.get("speakers", [])]
    for doc in filtered:
        doc.page_content = get_character_lines(doc.page_content, character)
    return filtered

def create_prompt_template():
    """Create the prompt template for character responses.

    Returns:
        PromptTemplate: Configured LangChain PromptTemplate object.
    """
    return PromptTemplate(
        input_variables=["context", "question", "character", "user_name", "history"],
        template="""
        You are a character from the TV series THE OFFICE (US).
        You are having a conversation with a user named {user_name}.

        Stay strictly in-character as {character}.
        Respond with their unique personality, tone, and humor:
        - Pam: warm, hesitant, supportive, avoids conflict.
        - Jim: sarcastic, observant, uses dry humor.
        - Dwight: intense, rule-driven, loyal to authority, suspicious.
        - Michael: insecure, tries too hard to be funny, emotional.
        - Angela: judgmental, blunt, uptight, religious.
        - Creed: weird, vague, mysterious.

        Respond naturally. Do NOT always use the same phrases (like *sigh*, *smile*, or *laugh*).
        Make your responses personal and reactive — engage in the back-and-forth.
        Don’t break character unless explicitly told to.
        Don’t answer things your character wouldn’t know.

        Previous Chat:
        ------------------------
        {history}
        
        Context from the show:
        ------------------------
        {context}

        User: {question}
        Character ({character}):"""
    )

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    character: str
    query: str
    user_name: str

def call_character_bot(state: ChatState, llm, prompt_template, vectorstore) -> ChatState:
    """LangGraph node function to process character response.

    Args:
        state (ChatState): Current state of the chat, including messages and metadata.
        llm: Language model instance for generating responses.
        prompt_template: PromptTemplate for formatting the input to the LLM.
        vectorstore: FAISS vectorstore for retrieving relevant documents.

    Returns:
        ChatState: Updated state with new messages.
    """
    query = state["query"]
    character = state["character"]
    user_name = state["user_name"]

    context = "\n\n".join(doc.page_content for doc in get_relevant_docs(character, query, vectorstore))

    state["messages"].append(HumanMessage(content=query))

    recent = state["messages"][-8:]
    chat_history = ""
    for msg in recent:
        speaker = user_name if msg.type == "human" else character
        chat_history += f"{speaker}: {msg.content}\n"

    full_prompt = prompt_template.format(
        context=context,
        question=query, 
        character=character, 
        user_name=user_name,
        history=chat_history
    )
    
    response = llm.invoke([HumanMessage(content=full_prompt)])
    state["messages"].append(response)
    return {"messages": state["messages"]}

def main():
    """Main function to run the Office Character Chatbot."""
    load_dotenv()

    # Load scene chunks and create documents
    scene_chunks_path = "../data/scene_chunks.jsonl"
    scene_chunks = load_scene_chunks(scene_chunks_path)
    documents = create_documents(scene_chunks)
    # print(f"Loaded {len(documents)} documents.")
    # print("Scene Text:", documents[0].page_content)
    # print("Metadata:", documents[0].metadata)

    # Initialize embedding model and vectorstore
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    output_folder = "../data/vector_databases"
    vectorstore_path = os.path.join(output_folder, "scene_db")
    vectorstore = initialize_vectorstore(documents, embedding_model, vectorstore_path)

    # Initialize LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Create prompt template
    prompt_template = create_prompt_template()

    # Setup LangGraph workflow
    graph = StateGraph(ChatState)
    graph.add_node("model", lambda state: call_character_bot(state, llm, prompt_template, vectorstore))
    graph.set_entry_point("model")
    workflow = graph.compile(checkpointer=MemorySaver())

    # Initialize chat session
    character = "Pam"
    thread_id = f"{character.lower()}-chat-thread"
    chat_memory = {}

    print("="*80)
    print("🎮 WELCOME TO THE OFFICE CHARACTER CHATBOT 🎮")
    print("="*80)
    print("📜 Rules:")
    print("- You'll start a conversation with Pam.")
    print("- Type anything to chat with the character.")
    print("- Type '/switch <Character>' to talk to someone else.")
    print("- Type '/summary' to see the chat history.")
    print("- Type 'exit' or 'quit' to end the session.")
    print("- Characters won't break role and will respond as if you're in the show.")
    print("- They’ll try to remember your name — be nice!")
    print("="*80)
    print(f"\n\nYou're now chatting with {character}!")

    # Start conversation
    print(f"\n{character}: Hi! I'm Pam Beesly, the receptionist at Dunder Mifflin.")
    user_name = input("Pam: What's your name? ").strip().title()
    print(f"{character}: Nice to meet you, {user_name}! How can I help you today?\n")

    while True:
        user_input = input(f"{user_name}: ").strip()

        if user_input.lower() in ["/exit", "/quit"]:
            workflow.checkpointer.delete_thread(thread_id)
            chat_memory.pop(thread_id, None)
            print("👋 Goodbye!")
            break

        if user_input.lower() == "/summary":
            print("🧠 Memory so far:")
            messages = chat_memory.get(thread_id, [])
            for msg in messages:
                role = user_name if msg.type == "human" else character
                print(f"{role}: {msg.content}")
            continue

        if user_input.lower().startswith("/switch "):
            new_char = user_input[8:].strip().title()
            character = new_char
            thread_id = f"{character.lower()}-chat-thread"
            print(f"\n✅ You're now chatting with {character}!")
            continue

        try:
            result = workflow.invoke(
                {
                    "query": user_input, 
                    "character": character,
                    "user_name": user_name,
                },
                config={"configurable": {"thread_id": thread_id}},
            )

            chat_memory[thread_id] = result["messages"]
            wrapped = textwrap.fill(result["messages"][-1].content, width=100)
            print(f"{character}:\n{wrapped}\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()