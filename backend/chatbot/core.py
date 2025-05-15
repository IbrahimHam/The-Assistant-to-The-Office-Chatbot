from typing_extensions import TypedDict, Annotated
from typing import List
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

from backend.chatbot.retrieval import get_relevant_docs
from backend.chatbot.emotion import get_top_emotions
from backend.chatbot.prompt import create_prompt_template

# LangGraph state


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    character: str
    query: str
    user_name: str


def call_character_bot(state: ChatState, llm, prompt_template, vectorstore) -> ChatState:
    """
    LangGraph node function to process character response.

    Args:
        state (ChatState): Current state of the chat, including messages and metadata.
        llm: LLM instance.
        prompt_template: LangChain PromptTemplate object.
        vectorstore: FAISS vectorstore instance.

    Returns:
        ChatState: Updated chat state with the new character response.
    """
    query = state["query"]
    character = state["character"]
    user_name = state["user_name"]

    relevant_docs = get_relevant_docs(character, query, vectorstore)
    context = "\n\n".join(
        doc.metadata["character_lines"]
        for doc in relevant_docs if doc.metadata.get("character_lines")
    ) or "No relevant scenes available."

    emotions = get_top_emotions(query, top_k=2)
    state["messages"].append(HumanMessage(content=query))

    recent = state["messages"][-6:]
    chat_history = "\n".join([
        f"{user_name if msg.type == 'human' else character}: {msg.content}"
        for msg in recent
    ])

    full_prompt = prompt_template.format(
        context=context,
        emotions=emotions,
        question=query,
        character=character,
        user_name=user_name,
        history=chat_history
    )

    response = llm.invoke([HumanMessage(content=full_prompt)])
    state["messages"].append(response)

    return {"messages": state["messages"]}


def run_chat(user_name: str, character: str, query: str, memory: List, workflow,
             llm, prompt_template, vectorstore):
    """
    Run a single chat turn through the backend logic and return the AI's response.

    Args:
        user_name (str): The user's name.
        character (str): The Office character being spoken to.
        query (str): The user's current message.
        memory (list): Past conversation messages (type, content).
        workflow: LangGraph compiled graph instance.
        llm: ChatGroq or other LLM instance.
        prompt_template: Prompt template for formatting.
        vectorstore: FAISS vectorstore.

    Returns:
        dict: Final response from the character.
    """
    messages = [
        HumanMessage(content=m.content) if m.type == "human"
        else SystemMessage(content=m.content)
        for m in memory
    ]

    recent = messages[-6:]
    chat_history = "\n".join([
        f"{user_name if isinstance(msg, HumanMessage) else character}: {msg.content[:300]}"
        for msg in recent
    ])

    relevant_docs = get_relevant_docs(character, query, vectorstore)
    context = "\n\n".join(
        doc.metadata["character_lines"]
        for doc in relevant_docs if doc.metadata.get("character_lines")
    ) or "No relevant scenes available."

    emotions = get_top_emotions(query, top_k=2)

    full_prompt = prompt_template.format(
        context=context,
        emotions=emotions,
        question=query,
        character=character,
        user_name=user_name,
        history=chat_history
    )

    print("=" * 100)
    print("🧠 RUN_CHAT DEBUG")
    print(f"👤 User: {user_name}")
    print(f"🎭 Character: {character}")
    print(f"💬 Query: {query}")
    print(f"🎭 Emotions: {emotions}")
    print(f"📄 Retrieved Scene IDs: {[doc.metadata['scene_id'] for doc in relevant_docs]}")
    print(f"🧾 Full Prompt:\n{full_prompt}")
    print("=" * 100)

    result = workflow.invoke(
        {
            "query": query,
            "character": character,
            "user_name": user_name,
            "messages": messages
        },
        config={"configurable": {"thread_id": f"{character.lower()}-web-thread"}}
    )

    return {"response": result["messages"][-1].content}
