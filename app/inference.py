"""Inference using the fine-tuned GPT-2 category-token scores."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_feedback_model(model_dir="models/feedback_finetuned"):
    if not Path(model_dir).is_dir():
        raise FileNotFoundError("Fine-tuned model missing. Run python train_model.py")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def predict_feedback_category(comment, model, tokenizer, category_tokens, max_length=128):
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("Feedback comment cannot be empty")
    if max_length < 16 or not category_tokens:
        raise ValueError("Invalid max_length or category tokens")
    prompt = f"Feedback: {comment.strip()}\nCategory:"
    # Use left truncation so the category cue is retained for long comments.
    previous_side = getattr(tokenizer, "truncation_side", "right")
    try:
        tokenizer.truncation_side = "left"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    finally:
        tokenizer.truncation_side = previous_side
    ids = []
    for token in category_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Invalid category token: {token}")
        ids.append(encoded[0])
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        scores = torch.softmax(logits[ids], dim=-1)
    best = int(torch.argmax(scores).item())
    token = category_tokens[best]
    return {
        "comment": comment.strip(),
        "predicted_category": token.strip("<>"),
        "predicted_token": token,
        "confidence": round(float(scores[best].item()), 4),
        "category_scores": {
            name.strip("<>"): round(float(scores[i].item()), 4)
            for i, name in enumerate(category_tokens)
        },
    }
