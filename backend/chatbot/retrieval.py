import random
from backend.chatbot.emotion import get_top_emotions

# Reused across sessions to avoid scene duplication
used_scene_ids = set()


def get_character_lines(text: str, character: str) -> str:
    """
    Filter only lines spoken by a given character from the full scene text.

    Args:
        text (str): Full text content of a scene.
        character (str): Character name.

    Returns:
        str: Only lines spoken by the character.
    """
    return "\n".join([
        line for line in text.split("\n")
        if line.startswith(f"{character}:")
    ])


def get_relevant_docs(character: str, query: str, vectorstore):
    """
    Retrieve documents relevant to the user's query that involve the target character.

    Args:
        character (str): The name of the character (e.g., "Pam").
        query (str): The user's input.
        vectorstore: A FAISS vectorstore instance.

    Returns:
        list: Filtered and prioritized list of Document objects.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(query)

    filtered = [
        doc for doc in docs
        if character.lower() in [s.lower() for s in doc.metadata.get("speakers", [])]
        and doc.metadata["scene_id"] not in used_scene_ids
    ]

    if not filtered:
        print(
            f"WARNING: No documents found for {character}. Check scene metadata.")

    target_emotions = get_top_emotions(query, top_k=2)
    print(f"Detected emotions: {target_emotions}")

    if not target_emotions:
        print("No emotions detected. Using semantic similarity instead.")
        for doc in filtered:
            doc.metadata["character_lines"] = get_character_lines(
                doc.page_content, character)
            used_scene_ids.add(doc.metadata["scene_id"])
        return filtered[:10]

    prioritized = []
    for doc in filtered:
        char_lines = [
            line for line in doc.metadata["lines"]
            if line["speaker"].lower() == character.lower()
        ]
        score = sum(
            1 if any(e in target_emotions for e in line["emotions"]) else 0
            for line in char_lines
        )
        prioritized.append((doc, score))

    prioritized.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in prioritized if score > 0][:20]
    random.shuffle(top_docs)

    filtered = top_docs[:10] + [doc for doc,
                                score in prioritized if score == 0][:10 - len(top_docs)]

    for doc in filtered:
        doc.metadata["character_lines"] = get_character_lines(
            doc.page_content, character)
        used_scene_ids.add(doc.metadata["scene_id"])

    return filtered
