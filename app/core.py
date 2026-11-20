"""Your code. One stub per milestone; build them in order and keep each one small."""

def milestone_1():
    """
    Load and validate labeled customer feedback.

    Prepare category tokens and training examples
    for GPT-2 fine-tuning.
    """
    import json
    from pathlib import Path
    from collections import Counter

    dataset_path = (
        Path(__file__).resolve().parent
        / "feedback_dataset.json"
    )

    with dataset_path.open(
        "r", encoding="utf-8"
    ) as file:
        feedback_data = json.load(file)

    categories = [
        "PRODUCT_ISSUE",
        "STAFF_PRAISE",
        "DELIVERY_ISSUE",
        "PRICING_FEEDBACK",
        "GENERAL_FEEDBACK"
    ]

    if not isinstance(feedback_data, list):
        raise ValueError("Dataset must be a list")

    if not feedback_data:
        raise ValueError("Dataset cannot be empty")

    cleaned_data = []
    training_examples = []
    seen_comments = set()

    for item in feedback_data:
        if not isinstance(item, dict):
            raise ValueError("Invalid feedback entry")

        if not all(
            key in item
            for key in ("id", "text", "label")
        ):
            raise ValueError(
                "Feedback entry is missing required fields"
            )

        if not isinstance(item["text"], str):
            raise ValueError(
                "Feedback text must be a string"
            )

        if not isinstance(item["label"], str):
            raise ValueError(
                "Feedback label must be a string"
            )

        text = item["text"].strip()
        label = item["label"].strip()

        if not text:
            raise ValueError(
                "Feedback text cannot be empty"
            )

        if label not in categories:
            raise ValueError(
                f"Unknown feedback category: {label}"
            )

        if text.casefold() in seen_comments:
            raise ValueError(
                "Duplicate feedback comment found"
            )

        seen_comments.add(text.casefold())

        cleaned_data.append({
            "id": item["id"],
            "text": text,
            "label": label
        })

        # Format examples for GPT-2 fine-tuning
        training_text = (
            f"Feedback: {text}\n"
            f"Category: <{label}>"
        )

        training_examples.append({
            "text": training_text,
            "label": label
        })

    category_counts = Counter(
        item["label"]
        for item in cleaned_data
    )

    category_tokens = [
        f"<{category}>"
        for category in categories
    ]

    return {
        "categories": categories,
        "category_tokens": category_tokens,
        "dataset": cleaned_data,
        "training_examples": training_examples,
        "total_examples": len(cleaned_data),
        "category_counts": dict(category_counts)
    }




def milestone_2(
    model_name="sshleifer/tiny-gpt2",
    output_dir="models/feedback_gpt2"
):
    """
    Extend the GPT-2 tokenizer and model with
    customer feedback category tokens.
    """
    from pathlib import Path

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )

    # Load the categories from Milestone 1
    dataset = milestone_1()
    category_tokens = dataset["category_tokens"]

    print("Loading pretrained tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    original_vocab_size = len(tokenizer)

    print("Loading pretrained GPT-2 model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    # GPT-2 does not have a default padding token.
    # Reuse EOS instead of adding another token.
    tokenizer.pad_token = tokenizer.eos_token

    print("Adding category tokens...")

    added_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": category_tokens
    })

    # The model must have an embedding for every token.
    model.resize_token_embeddings(
        len(tokenizer)
    )

    new_vocab_size = len(tokenizer)

    # Verify each category is a single special token
    token_ids = {}

    for token in category_tokens:
        token_id = tokenizer.convert_tokens_to_ids(
            token
        )

        encoded = tokenizer.encode(
            token,
            add_special_tokens=False
        )

        if encoded != [token_id]:
            raise ValueError(
                f"Category token was not registered: {token}"
            )

        token_ids[token] = token_id

    # Verify tokenizer and model have matching sizes
    embedding_size = (
        model.get_input_embeddings().weight.shape[0]
    )

    if embedding_size != new_vocab_size:
        raise ValueError(
            "Tokenizer and model vocabulary sizes differ"
        )

    # Save the extended model and tokenizer
    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save_pretrained(
        str(output_path)
    )

    model.save_pretrained(
        str(output_path),
        safe_serialization=True
    )

    print("\nTOKENIZER EXTENSION COMPLETE")
    print("Original vocabulary:", original_vocab_size)
    print("New vocabulary:", new_vocab_size)
    print("Added tokens:", added_tokens)
    print("Model saved to:", output_path)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "added_tokens": added_tokens,
        "category_tokens": category_tokens,
        "token_ids": token_ids,
        "embedding_size": embedding_size,
        "output_dir": str(output_path)
    }



# --- Milestone 3: Fine-tune the model using the Hugging Face Trainer.
# Hint: Use the Hugging Face Trainer class, as shown in the drill, to manage the fine-tuning process.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Adapt the inference pipeline from the drill to process new customer comments and generate category predictions.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Use Streamlit widgets to create an input text area and display the classification results.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
