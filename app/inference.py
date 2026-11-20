
"""Inference for the Customer Feedback Analyzer."""

from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


def load_feedback_model(
    model_dir="models/feedback_finetuned"
):
    """Load the fine-tuned GPT-2 model and tokenizer."""

    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            "Fine-tuned model not found. "
            "Run python train_model.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


def predict_feedback_category(
    comment,
    model,
    tokenizer,
    category_tokens,
    max_length=128
):
    """
    Classify customer feedback using GPT-2's
    next-token category probabilities.
    """

    if not isinstance(comment, str):
        raise ValueError(
            "Feedback comment must be a string"
        )

    comment = comment.strip()

    if not comment:
        raise ValueError(
            "Feedback comment cannot be empty"
        )

    if max_length < 16:
        raise ValueError(
            "max_length must be at least 16"
        )

    if not category_tokens:
        raise ValueError(
            "Category tokens cannot be empty"
        )

    prompt = (
        f"Feedback: {comment}\n"
        "Category:"
    )

    # Match the training prompt from Milestone 3.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    category_ids = []

    for token in category_tokens:
        ids = tokenizer.encode(
            token,
            add_special_tokens=False
        )

        if len(ids) != 1:
            raise ValueError(
                f"Invalid category token: {token}"
            )

        category_ids.append(ids[0])

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

        # GPT-2 predicts the next token from
        # the final position of the prompt.
        next_token_logits = outputs.logits[
            0, -1, :
        ]

        # Restrict predictions to known categories.
        category_logits = next_token_logits[
            category_ids
        ]

        probabilities = torch.softmax(
            category_logits,
            dim=-1
        )

    best_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_token = category_tokens[
        best_index
    ]

    predicted_category = (
        predicted_token.strip("<>")
    )

    confidence = float(
        probabilities[best_index].item()
    )

    scores = {
        token.strip("<>"): round(
            float(probabilities[index].item()),
            4
        )
        for index, token in enumerate(
            category_tokens
        )
    }

    return {
        "comment": comment,
        "predicted_category": predicted_category,
        "predicted_token": predicted_token,
        "confidence": round(confidence, 4),
        "category_scores": scores
    }
