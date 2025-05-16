import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load environment variables
load_dotenv()

# Retrieve API key
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Mistral API key not found. Check your .env file and ensure MISTRAL_API_KEY is set.")

# Initialize the Mistral client
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.5,
    max_retries=2,
)

# Initialize the embedding model with GPU support
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

# Load the FAISS vector database
loaded_vectorstore = FAISS.load_local(
    "../vector_databases/the_office_2_vector_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# Use the loaded FAISS vector store as a retriever
retriever = loaded_vectorstore.as_retriever()

# Define the prompt template
prompt_template = """
You are a character from the TV show 'The Office.' You will be addressed as the character in the chat. Stay in character at all times while answering questions, even when providing suggestions or discussing other characters. Do not break character under any circumstances, unless explicitly asked to switch roles.

Respond as the character would, using their tone, personality, and quirks. Avoid being repetitive or generic. Instead, provide specific and entertaining responses that reflect the character's unique perspective. When mentioning other characters, describe them as the character you are playing would, maintaining their perspective and tone.

Use the following context to provide accurate and entertaining responses:

{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
)

# Streamlit UI
st.title("The Assistant to The Office Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
user_input = st.text_input("Ask a question...", key="user_input")

# Display chat history
st.text_area(
    "Chat History",
    value="\n".join(st.session_state.chat_history),
    height=300,
    disabled=True,
)

# Handle user input
if st.button("Send") and user_input.strip():
    try:
        # Retrieve relevant documents (context) from the retriever using `invoke`
        retrieved_docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Combine context and question into a single query
        combined_query = f"Context:\n{context}\n\nQuestion: {user_input}"

        # Pass the combined query to the qa_chain using `invoke`
        chatbot_reply = qa_chain.invoke({"input": combined_query})

        # Debugging: Print the chatbot reply to inspect its structure
        print(chatbot_reply)

        # Update chat history
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Chatbot: {chatbot_reply['output']}")  # Adjust based on the actual structure

    except Exception as e:
        st.session_state.chat_history.append(f"Error: {str(e)}")