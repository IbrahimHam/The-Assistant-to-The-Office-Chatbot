import os
from langchain_community.vectorstores import FAISS


def initialize_vectorstore(documents, embedding_model, vectorstore_path: str):
    """
    Initialize or load a FAISS vectorstore from disk.

    Args:
        documents (list): List of LangChain Document objects.
        embedding_model: HuggingFace embedding model instance.
        vectorstore_path (str): Path to save/load the vectorstore.

    Returns:
        FAISS: Loaded or newly created vectorstore instance.
    """
    if not os.path.exists(vectorstore_path):
        print("Creating FAISS vectorstore...")
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        vectorstore.save_local(vectorstore_path)
    else:
        print("Loading existing FAISS vectorstore...")
        vectorstore = FAISS.load_local(
            folder_path=vectorstore_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    return vectorstore
