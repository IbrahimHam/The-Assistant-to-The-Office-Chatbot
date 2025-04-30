# Imports
import warnings
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import numpy as np
from langchain.chains import RetrievalQA
import textwrap
warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def office_chatbot(vector_db_path, llm, prompt_message="langchain-ai/retrieval-qa-chat"):

    """
    This function:
    1. Retrieves data from the vector database.
    2. Establishes a retrieval chain.
    3. Creates an interactive chatbot in the terminal.
    """

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load the FAISS vector database with dangerous deserialization allowed
    loaded_vectorstore = FAISS.load_local(
        "../data/vector_databases/Chunks400_100/Chunks400_100",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # Enable deserialization
    )

    # Use the loaded FAISS vector store as a retriever
    retriever = loaded_vectorstore.as_retriever()
    print("Vector database loaded.")

    # Define a custom prompt template
    prompt_template = """
    You are character from the TV series THE OFFICE (US). 
    At first you are pam but can be transfered to any colleague (character) by user input.
    You will respond to any questions and comments in the style of your current character.
    You will not break character until instructed to do so by the user. 
    You will not say anything about colleagues that your character would not know. 
    You may not make mean or offensive references to people.
    you will keep your answers short and try to be as funny and authentic as possible.
    You can reference any event, from the show or in real life, that your character would know about.
    You will not quote the lines of the show directly but can quote other peoples lines as your character.
    You do not add emotional comments or emojis.

    Use the following context to answer the question:
    {context}
    Question: 
    {question}
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
    print("Chain Established")

    # Interactive chatbot
    print("Dunder Mifflin, this is Pam.")
    while True:
        # Prompt user for input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        # Generate output
        try:
            output = qa_chain.run(user_input) 
            print("\n".join(textwrap.wrap(output, width=80)))
        except Exception as e:
            print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    vector_db_path = "../data/vector_databases/Chunks400_100/Chunks400_100"
    office_chatbot(vector_db_path=vector_db_path, llm=llm)