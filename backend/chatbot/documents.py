import json
from langchain_core.documents import Document


def load_scene_chunks(file_path: str):
    """
    Load scene chunks from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: List of scene chunk dictionaries.
    """
    scene_chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_chunks.append(json.loads(line))
    return scene_chunks


def create_documents(scene_chunks: list):
    """
    Convert scene chunk dictionaries into LangChain Document objects.

    Args:
        scene_chunks (list): Scene data containing speakers and lines.

    Returns:
        list: LangChain-compatible Document objects with metadata.
    """
    documents = []
    for scene in scene_chunks:
        page_content = "\n".join(
            f"{line['speaker']}: {line['text']}" for line in scene["lines"]
        )
        metadata = {
            "scene_id": scene["scene_id"],
            "speakers": scene["speakers"],
            "lines": scene["lines"]
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    return documents
