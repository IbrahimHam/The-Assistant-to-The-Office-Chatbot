

from langchain_mistralai import ChatMistralAI

# %%
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import textwrap
import torch
import ipywidgets as widgets

# Load environment variables
load_dotenv()

# Retrieve API key
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError(
        "Mistral API key not found. Check your .env file and ensure MISTRAL_API_KEY is set."
    )

print("Mistral API key loaded from .env")

# Initialize the Mistral client
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.5,
    max_retries=2,
)





# %%
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch



# %%
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the FAISS vector database with dangerous deserialization allowed
loaded_vectorstore = FAISS.load_local(
    "../vector_databases/the_office_2_vector_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,  # Enable deserialization
)

# Perform a similarity search
query = "What does Michael Scott say about leadership?"
results = loaded_vectorstore.similarity_search(query, k=5)

# Display the results
for result in results:
    print(result)

# %%
# Use the loaded FAISS vector store as a retriever
retriever = loaded_vectorstore.as_retriever()

# Define a custom prompt template
prompt_template = """
You are a character from the TV show 'The Office.' Stay in character while answering questions.
Use the following context to provide accurate and entertaining responses:

{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the RetrievalQA chain with a single output key
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,  # Exclude source documents
    chain_type_kwargs={"prompt": prompt},
)

# Test the RetrievalQA chain
query = "Hey Pam, what's the best prank that was pulled on Dwight?"
response = qa_chain.run(query)  # Now `run` will work

# Display the response
print("\n".join(textwrap.wrap(response, width=80)))  # Adjust width as needed

# %%
# Test the RetrievalQA chain
query = "Hi Michael, what do you think about data science?"
response = qa_chain.run(query)  # Now `run` will work

# Display the response
print("\n".join(textwrap.wrap(response, width=80)))  # Adjust width as needed

# %%
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# %%
# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# %%
# Create a Conversational RetrievalQA chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,  # Exclude source documents
)

# %%
prompt_template = """
You are a character from the TV show 'The Office.' Stay in character while answering questions.
Use the following context to provide accurate and entertaining responses:

{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# %%
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,  # Exclude source documents
    memory=memory,
    chain_type_kwargs={"prompt": prompt},
)

# %% [markdown]
# 

# %%


# %%
# Define the chatbot response function
import ipywidgets as widgets
from IPython.display import display, clear_output

# Initialize a list to store the chat history
chat_history = []


def chatbot_response(change):
    user_input = text_box.value
    if user_input.strip():  # Ensure input is not empty
        try:
            # Pass only the query to the qa_chain
            chatbot_reply = qa_chain.run(user_input)  # Pass the query directly

            # Add the user input and chatbot reply to the chat history
            chat_history.append(
                f"You: {user_input}\n"
            )  # Add a line break after the question
            chat_history.append(
                f"Chatbot:\n{chatbot_reply}\n"
            )  # Add the response in a new paragraph

            # Update the chat window
            chat_window.value = "\n".join(chat_history)

            # Clear the input box after submission
            text_box.value = ""
        except Exception as e:
            chat_history.append(f"Error: {str(e)}\n")
            chat_window.value = "\n".join(chat_history)


def clear_chat(_):
    """Clear the chat history and reset the chat window."""
    global chat_history
    chat_history = []  # Reset the chat history
    chat_window.value = ""  # Clear the chat window


# Create the input text box
text_box = widgets.Text(placeholder="Ask a question...")
text_box.observe(chatbot_response, names="value")

# Disable continuous updates (trigger only on Enter)
text_box.continuous_update = False

# Create the chat window (TextArea widget to display chat history)
chat_window = widgets.Textarea(
    value="",
    placeholder="Chat history will appear here...",
    description="",
    layout=widgets.Layout(width="100%", height="300px"),
    style={"font_size": "16px"},  # Increase font size
    disabled=True,  # Make it read-only
)

# Create the "Clear Chat" button
clear_button = widgets.Button(
    description="Clear Chat",
    button_style="danger",  # Red button
    tooltip="Clear the chat history",
    icon="trash",  # Trash icon
)
clear_button.on_click(clear_chat)

# Display the UI
display(chat_window, text_box, clear_button)

# %%



