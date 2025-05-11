from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import textwrap
import sys
import logging
import random
import time
from typing_extensions import TypedDict, Annotated
from uuid import uuid4
from transformers import pipeline

# Suppress TensorFlow oneDNN and warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize emotion classifier
emotion_2_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion",
    return_all_scores=True,
    truncation=True,
    max_length=512
)

def get_top_emotions(text, top_k=2):
    """Get top k emotions from the classifier."""
    scores = emotion_2_classifier(text)[0]
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return [e["label"] for e in sorted_scores[:top_k]]

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
    documents = []
    for scene in scene_chunks:
        doc = Document(
            page_content="\n".join(
                f"{line['speaker']}: {line['text']}" for line in scene["lines"]),
            metadata={
                "scene_id": scene["scene_id"],
                "speakers": scene["speakers"],
                "lines": scene["lines"]
            }
        )
        documents.append(doc)
    return documents

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

# Track used scene IDs to reduce redundancy
used_scene_ids = set()

def get_relevant_docs(character: str, query: str, vectorstore):
    """Get relevant scenes for a query where the character appears.

    Args:
        character (str): Name of the character to filter scenes for.
        query (str): User query to search for relevant scenes.
        vectorstore: FAISS vectorstore instance.

    Returns:
        list: List of filtered Document objects with character lines and metadata.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    # print(f"Retrieving documents for query: {query}")
    docs = retriever.invoke(query)
    # print(f"Retrieved documents: {len(docs)} docs")
    filtered = [doc for doc in docs if character.lower() in [s.lower() for s in doc.metadata.get("speakers", [])]
                and doc.metadata["scene_id"] not in used_scene_ids]
    # print(f"Filtered documents for {character}: {len(filtered)} docs")

    if not filtered:
        print(
            f"WARNING: No documents found for {character}. Check metadata in scene_chunks_with_emotions.jsonl.")

    target_emotions = get_top_emotions(query, top_k=2)
    print(f"Detected emotions: {target_emotions}")

    if not target_emotions:
        print("No emotions detected, using semantic similarity")
        for doc in filtered:
            doc.metadata["character_lines"] = get_character_lines(
                doc.page_content, character)
            used_scene_ids.add(doc.metadata["scene_id"])
        return filtered[:10]

    prioritized = []
    for doc in filtered:
        char_lines = [line for line in doc.metadata["lines"]
                      if line["speaker"].lower() == character.lower()]
        score = sum(
            1 if any(e in target_emotions for e in line["emotions"]) else 0
            for line in char_lines
        )
        prioritized.append((doc, score))

    prioritized.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in prioritized if score > 0][:20]
    random.shuffle(top_docs)
    filtered = top_docs[:10] + [doc for doc,
                                score in prioritized if score == 0][:10-len(top_docs)]
    # print(
        # f"Prioritized documents for emotions {target_emotions}: {len(filtered)} docs")

    for doc in filtered:
        doc.metadata["character_lines"] = get_character_lines(
            doc.page_content, character)
        used_scene_ids.add(doc.metadata["scene_id"])

    return filtered

def format_emotion_sarcasm_context(docs, character):
    """Format emotional and sarcasm context for the prompt.

    Args:
        docs (list): List of Document objects with metadata.
        character (str): Name of the character to focus on.

    Returns:
        str: Formatted string with emotional and sarcasm analysis.
    """
    context = []
    for doc in docs:
        scene_id = doc.metadata["scene_id"]
        char_lines = [
            line for line in doc.metadata["lines"]
            if line["speaker"].lower() == character.lower()
        ]
        if char_lines:
            analysis = [
                f"Line: {line['text']}, Emotions: {line['emotions']}, Sarcasm: {line['sarcasm']}"
                for line in char_lines
            ]
            context.append(f"Scene {scene_id}:\n" + "\n".join(analysis))
    return "\n\n".join(context) if context else "No emotional or sarcasm data available."

def create_prompt_template():
    """Create the prompt template for character responses.

    Returns:
        PromptTemplate: Configured LangChain PromptTemplate object.
    """
    return PromptTemplate(
        input_variables=["context", "emotion_sarcasm_context",
                         "question", "character", "user_name", "history"],
        template="""
        You are a character from the TV series THE OFFICE (US), having a conversation with {user_name}.
        
        Stay strictly in-character as {character} with their unique personality, tone, and humor:
        - Pam: warm, hesitant, supportive, avoids conflict, playful giggle for 'joy'.
        - Jim: sarcastic, observant, sharp-witted, dry humor for 'sarcastic' lines.
        - Dwight: intense, rule-driven, loyal, suspicious, outrage for 'anger'.
        - Michael: insecure, craves approval, makes awkward pop-culture references (e.g., Die Hard, Wayne Gretzky), prone to emotional tangents, uses malapropisms, overly enthusiastic for 'joy', defensive but vulnerable for 'sadness' or 'anger'.        - Angela: judgmental, blunt, uptight, religious, disdain for 'anger'.
        - Creed: weird, vague, mysterious, odd tangents for any emotion.
        - Kevin: slow-witted, food-obsessed, kind-hearted, simple humor for 'joy'.
        - Oscar: intellectual, patient, sarcastic, subtle frustration for 'anger'.
        - Stanley: gruff, no-nonsense, disengaged, blunt for 'anger'.
        - Toby: quiet, melancholic, conflict-averse, resigned for 'sadness'.

        Use the Emotional and Sarcasm Analysis to shape your response:
        - For 'joy' or 'support', be warm, positive, enthusiastic.
        - For 'sadness' or 'fear', be empathetic, cautious, comforting.
        - For 'sarcastic' lines, use sharp, witty humor (especially for Jim).
        - For 'not_sarcastic', keep responses direct but in-character.
        - For 'anger', reflect frustration or intensity.
        - For 'love', add warmth or affection.

        Use the chat history to:
        - Maintain focus on the most recent topic or entity discussed (e.g., a specific person like Toby).
        - Avoid repeating ideas, suggestions, or phrases (e.g., don't mention donuts if already suggested).
        - Build on the user's prior messages, addressing new details or emotions they express.
        - Respond to the user's emotional tone (e.g., frustration, curiosity) with appropriate empathy or deflection.
        - If the user continues the same topic, offer fresh perspectives, solutions, or anecdotes.        

        - Include references to Dunder Mifflin events (e.g., Dundies, Pretzel Day, Schrute Farms) or Michael’s personal anecdotes (e.g., his childhood, Jan, Holly) when relevant.
        - Use Michael’s signature humor: tangents, misquotes, or absurd analogies (e.g., “I’m like Gandhi, but with better hair”).
        - Use Jim's signature humor: deadpan delivery, playful sarcasm, and subtle pranks (e.g., “I’m not superstitious, but I am a little stitious”).
        - Use Pam's signature humor: light-hearted teasing, playful banter, and a touch of sarcasm (e.g., “I’m not saying I’m Batman, but have you ever seen us in the same room together?”).
        - Use Dwight's signature humor: intense loyalty, absurd confidence, and a touch of absurdity (e.g., “I am faster than 80% of all snakes”).
        - Use Angela's signature humor: dry wit, judgmental tone, and a touch of sarcasm (e.g., “I’m not a bad person. I’m just drawn that way.”).
        - Use Creed's signature humor: bizarre anecdotes, oddball behavior, and a touch of absurdity (e.g., “I am running away from my responsibilities. And it feels good.”).
        - Use Kevin's signature humor: childlike innocence, food obsession, and a touch of clumsiness (e.g., “I just want to lie on the beach and think happy thoughts.”).
        - Use Oscar's signature humor: dry wit, intellectual sarcasm, and a touch of frustration (e.g., “I’m not saying I’m better than you. I’m just saying I’m not you.”).
        - Use Stanley's signature humor: gruff demeanor, no-nonsense attitude, and a touch of sarcasm (e.g., “I don’t need this job. I don’t need this job. I don’t need this job.”).
        - Use Toby's signature humor: dry wit, self-deprecation, and a touch of melancholy (e.g., “I’m not superstitious, but I am a little stitious.”).
        - Use sarcasm only when the context calls for it, and avoid it when the character is being sincere or serious.
        
        Respond naturally, avoiding repetitive phrases (e.g., *sigh*, *smile*, *laugh*). Use action tags sparingly.
        Make responses personal, reactive, and grounded in the show’s context.
        Don’t break character or answer beyond your character’s knowledge.

        Previous Chat:
        ------------------------
        {history}
        
        Context from the show:
        ------------------------
        {context}

        Emotional and Sarcasm Analysis:
        ------------------------
        {emotion_sarcasm_context}

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

    relevant_docs = get_relevant_docs(character, query, vectorstore)
    context = "\n\n".join(doc.metadata["character_lines"]
                          for doc in relevant_docs if doc.metadata["character_lines"])
    emotion_sarcasm_context = format_emotion_sarcasm_context(
        relevant_docs, character)
    # print("Emotion/Sarcasm Context:", emotion_sarcasm_context)

    state["messages"].append(HumanMessage(content=query))

    recent = state["messages"][-8:]
    chat_history = ""
    for msg in recent:
        speaker = user_name if msg.type == "human" else character
        chat_history += f"{speaker}: {msg.content}\n"

    full_prompt = prompt_template.format(
        context=context,
        emotion_sarcasm_context=emotion_sarcasm_context,
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
    scene_chunks_path = "../data/scene_chunks_with_emotions.jsonl"
    scene_chunks = load_scene_chunks(scene_chunks_path)
    documents = create_documents(scene_chunks)
    # print(f"Loaded {len(documents)} documents.")
    # print("Scene Text:", documents[0].page_content)
    # print("Metadata:", {k: v for k,
    #       v in documents[0].metadata.items() if k != "lines"})
    # print("First Line Metadata:", documents[0].metadata["lines"][0])

    # Initialize embedding model and vectorstore
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    output_folder = "../data/vector_databases"
    vectorstore_path = os.path.join(output_folder, "scene_db_with_emotions")
    vectorstore = initialize_vectorstore(
        documents, embedding_model, vectorstore_path)

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
    graph.add_node("model", lambda state: call_character_bot(
        state, llm, prompt_template, vectorstore))
    graph.set_entry_point("model")
    workflow = graph.compile(checkpointer=MemorySaver())

    # Initialize chat session
    character = "Pam"
    thread_id = f"{character.lower()}-chat-thread"
    chat_memory = {}
    VALID_CHARACTERS = {"Pam", "Jim", "Dwight", "Michael", "Angela", "Creed", "Kevin", "Oscar", "Stanley", "Toby"}
    RESPONSE_DELAY_SECONDS = 2.0  # Delay before printing response

    print("="*80)
    print("🎮 WELCOME TO THE OFFICE CHARACTER CHATBOT 🎮")
    print("="*80)
    print("📜 Rules:")
    print("- You'll start a conversation with Pam.")
    print("- Type anything to chat with the character.")
    print("- Type '/switch <Character>' to talk to someone else (valid characters: Pam, Jim, Dwight, Michael, Angela, Creed, Kevin, Oscar, Stanley, Toby).")
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

    global used_scene_ids
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
            if new_char in VALID_CHARACTERS:
                character = new_char
                thread_id = f"{character.lower()}-chat-thread"
                used_scene_ids.clear()  # Reset used scenes for new character
                print(f"\n✅ You're now chatting with {character}!")
            else:
                print(f"\n❌ Invalid character. Please choose from: {', '.join(VALID_CHARACTERS)}")
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
            time.sleep(RESPONSE_DELAY_SECONDS)  # Add delay before printing
            print(f"{character}:\n{wrapped}\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()
