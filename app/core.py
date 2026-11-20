"""All five Customer Feedback Analyzer milestones."""

from collections import Counter
from pathlib import Path
import json

CATEGORIES = (
    "PRODUCT_ISSUE", "STAFF_PRAISE", "DELIVERY_ISSUE",
    "PRICING_FEEDBACK", "GENERAL_FEEDBACK",
)


def milestone_1():
    """Load and validate the labeled customer feedback dataset."""
    path = Path(__file__).resolve().parent / "feedback_dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a nonempty list")
    cleaned, seen = [], set()
    for item in data:
        if not isinstance(item, dict) or not all(k in item for k in ("id", "text", "label")):
            raise ValueError("Dataset entry must have id, text, and label")
        if not isinstance(item["text"], str) or not isinstance(item["label"], str):
            raise ValueError("Text and label must be strings")
        comment, label = item["text"].strip(), item["label"].strip()
        if not comment or label not in CATEGORIES or comment.casefold() in seen:
            raise ValueError("Empty, invalid, or duplicate feedback entry")
        seen.add(comment.casefold())
        cleaned.append({"id": item["id"], "text": comment, "label": label})
    return {
        "categories": list(CATEGORIES),
        "category_tokens": [f"<{label}>" for label in CATEGORIES],
        "dataset": cleaned,
        "training_examples": [
            {"text": f"Feedback: {x['text']}\nCategory: <{x['label']}>", "label": x["label"]}
            for x in cleaned
        ],
        "total_examples": len(cleaned),
        "category_counts": dict(Counter(x["label"] for x in cleaned)),
    }


def milestone_2(model_name="sshleifer/tiny-gpt2", output_dir="models/feedback_gpt2"):
    """Add category tokens and resize the GPT-2 model embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokens = milestone_1()["category_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original = len(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    token_ids = {}
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token was not added correctly: {token}")
        token_ids[token] = ids[0]
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if embedding_size != len(tokenizer):
        raise ValueError("Embedding and tokenizer sizes differ")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output))
    model.save_pretrained(str(output), safe_serialization=True)
    return {
        "model": model, "tokenizer": tokenizer, "model_name": model_name,
        "original_vocab_size": original, "new_vocab_size": len(tokenizer),
        "added_tokens": added, "category_tokens": tokens,
        "token_ids": token_ids, "embedding_size": embedding_size,
        "output_dir": str(output),
    }


def milestone_3(model_dir="models/feedback_gpt2", output_dir="models/feedback_finetuned",
                epochs=5, batch_size=4, learning_rate=0.0001, max_length=128):
    """Fine-tune the extended GPT-2 model with Hugging Face Trainer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from app.training import FeedbackDataset, FeedbackCollator

    if epochs <= 0 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    if not Path(model_dir).is_dir():
        raise FileNotFoundError("Run python extend_tokenizer.py first")
    torch.manual_seed(42)
    data = milestone_1()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    for token in data["category_tokens"]:
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
            raise ValueError(f"Missing category token: {token}")
    dataset = FeedbackDataset(data["dataset"], tokenizer, max_length=max_length)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=0.01,
        logging_steps=1, save_strategy="no", report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0,
        fp16=False, seed=42, disable_tqdm=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=FeedbackCollator(tokenizer.pad_token_id),
    )
    outcome = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    return {
        "model": trainer.model, "tokenizer": tokenizer, "trainer": trainer,
        "training_loss": float(outcome.training_loss),
        "training_examples": len(dataset), "epochs": epochs,
        "batch_size": batch_size, "category_tokens": data["category_tokens"],
        "output_dir": str(output),
    }


def milestone_4(comment, model=None, tokenizer=None, model_dir="models/feedback_finetuned"):
    """Classify a comment using category-token next-token probabilities."""
    from app.inference import load_feedback_model, predict_feedback_category
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("Please enter a valid feedback comment")
    if model is None or tokenizer is None:
        model, tokenizer = load_feedback_model(model_dir)
    return predict_feedback_category(
        comment, model, tokenizer, milestone_1()["category_tokens"]
    )


def milestone_5(comment, model=None, tokenizer=None, model_dir="models/feedback_finetuned"):
    """Format a prediction for the Streamlit user interface."""
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("Please enter a customer feedback comment")
    result = milestone_4(comment, model=model, tokenizer=tokenizer, model_dir=model_dir)
    predicted = result["predicted_category"]
    return {
        **result,
        "display_category": predicted.replace("_", " ").title(),
        "confidence_percentage": round(result["confidence"] * 100, 2),
    }
