import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline

# Suppress TensorFlow oneDNN and warning messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize emotion classifier once
emotion_2_classifier = pipeline(
    "text-classification",
    model=os.getenv("EMOTION_ANALYSIS_MODEL"),
    top_k=None,
    truncation=True,
    max_length=512
)


def get_top_emotions(text: str, top_k: int = 2):
    """
    Get top k emotions from the classifier for a given text.

    Args:
        text (str): Input text from the user.
        top_k (int): Number of top emotions to return.

    Returns:
        list: List of top-k emotion labels.
    """
    scores = emotion_2_classifier(text)[0]
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return [e["label"] for e in sorted_scores[:top_k]]
