import os
import sys
import time
import textwrap
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from backend.chatbot.core import ChatState, call_character_bot
from backend.chatbot.prompt import create_prompt_template
from backend.chatbot.documents import load_scene_chunks, create_documents
from backend.chatbot.vectorstore import initialize_vectorstore


def main():
    """Main function to run the Office Character Chatbot in the terminal."""
    load_dotenv()

    # Load documents
    scene_chunks_path = os.path.abspath(
        "data/scene_chunks_with_emotions.jsonl")
    scene_chunks = load_scene_chunks(scene_chunks_path)
    documents = create_documents(scene_chunks)

    # Initialize vectorstore
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_path = os.path.abspath(
        "data/vector_databases/scene_db_with_emotions")
    vectorstore = initialize_vectorstore(
        documents, embedding_model, vectorstore_path)

    # LLM + prompt
    llm = ChatGroq(model=os.getenv("CHAT_MODEL"), temperature=0.6)
    prompt_template = create_prompt_template()

    # LangGraph workflow
    graph = StateGraph(ChatState)
    graph.add_node("model", lambda state: call_character_bot(
        state, llm, prompt_template, vectorstore))
    graph.set_entry_point("model")
    workflow = graph.compile(checkpointer=MemorySaver())

    # Start console session
    character = "Pam"
    VALID_CHARACTERS = {"Pam", "Jim", "Dwight", "Michael",
                        "Angela", "Creed", "Kevin", "Oscar", "Stanley", "Toby"}
    chat_memory = {}
    thread_id = f"{character.lower()}-chat-thread"
    RESPONSE_DELAY_SECONDS = 2.0

    print("="*80)
    print("🎮 WELCOME TO THE OFFICE CHARACTER CHATBOT 🎮")
    print("="*80)
    print("📜 Rules:")
    print("- Start chatting with Pam.")
    print("- Use '/switch <Character>' to switch.")
    print("- Use '/summary' to view memory.")
    print("- Use '/exit' or '/quit' to leave.")
    print("="*80)
    print(f"\n\nYou're now chatting with {character}!")
    print(f"{character}: Hi! I'm Pam Beesly, the receptionist at Dunder Mifflin.")
    user_name = input("Pam: What's your name? ").strip().title()
    print(f"{character}: Nice to meet you, {user_name}! How can I help you today?\n")

    from backend.chatbot.retrieval import used_scene_ids
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
                speaker = user_name if msg.type == "human" else character
                print(f"{speaker}: {msg.content}")
            continue

        if user_input.lower().startswith("/switch "):
            new_char = user_input[8:].strip().title()
            if new_char in VALID_CHARACTERS:
                character = new_char
                thread_id = f"{character.lower()}-chat-thread"
                used_scene_ids.clear()
                print(f"\n✅ You're now chatting with {character}!")
            else:
                print(
                    f"\n❌ Invalid character. Try: {', '.join(VALID_CHARACTERS)}")
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
            time.sleep(RESPONSE_DELAY_SECONDS)
            print(f"{character}:\n{wrapped}\n")
            sys.stdout.flush()

        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()
