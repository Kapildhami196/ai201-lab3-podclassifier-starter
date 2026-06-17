import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    instruction = """You are classifying podcast episodes by their format.
Classify the episode into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion, with no guests or assembled external sources
- panel: multiple speakers with roughly equal standing who discuss or debate a topic together
- narrative: a story assembled from external sources such as interviews, archival audio, documents, or reporting

Use the episode structure, not its topic, tone, or marketing language.

Return exactly two lines:
Label: <interview, solo, panel, or narrative>
Reasoning: <one brief explanation>
"""

    example_blocks = []

    for example in labeled_examples:
        example_block = "\n".join(
            [
                f"Title: {example.get('title', '')}",
                f"Description: {example.get('description', '')}",
                f"Label: {example.get('label', '')}",
            ]
        )
        example_blocks.append(example_block)

    examples_text = "\n\n---\n\n".join(example_blocks)
    clean_description = description.strip()

    if not clean_description:
        clean_description = (
            "No description was provided. Make the best classification possible "
            "without inventing details."
        )

    prompt_parts = [instruction.strip()]

    if examples_text:
        prompt_parts.extend(
            [
                "Labeled examples:",
                examples_text,
            ]
        )
    else:
        prompt_parts.append(
            "No labeled examples are available. "
            "Classify using only the taxonomy above."
        )

    prompt_parts.extend(
        [
            "Episode to classify:",
            f"Description: {clean_description}",
            "Label: ?",
            (
                "Return exactly two lines using the required "
                "Label and Reasoning format."
            ),
        ]
    )

    return "\n\n".join(prompt_parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    return {
        "label": None,
        "reasoning": "Classifier not yet implemented. Complete Milestone 2.",
    }
