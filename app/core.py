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




def milestone_3(
    model_dir="models/feedback_gpt2",
    output_dir="models/feedback_finetuned",
    epochs=5,
    batch_size=4,
    learning_rate=0.0001,
    max_length=128
):
    """
    Fine-tune the extended GPT-2 model
    using Hugging Face Trainer.
    """
    from pathlib import Path

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )

    from app.training import (
        FeedbackDataset,
        FeedbackCollator
    )

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            "Extended model not found. "
            "Run python extend_tokenizer.py first."
        )

    torch.manual_seed(42)

    # Load the labeled feedback dataset
    data = milestone_1()

    print("Loading extended GPT-2 model...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # Confirm all category tokens are available
    for token in data["category_tokens"]:
        token_ids = tokenizer.encode(
            token,
            add_special_tokens=False
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"Missing category token: {token}"
            )

    train_dataset = FeedbackDataset(
        examples=data["dataset"],
        tokenizer=tokenizer,
        max_length=max_length
    )

    data_collator = FeedbackCollator(
        pad_token_id=tokenizer.pad_token_id
    )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save checkpoints separately from the final model
    checkpoint_path = (
        output_path / "checkpoints"
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=False,
        seed=42,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    print("\nStarting GPT-2 fine-tuning...")
    print("Training examples:", len(train_dataset))
    print("Epochs:", epochs)
    print("Batch size:", batch_size)

    train_result = trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    training_loss = float(
        train_result.training_loss
    )

    print("\nTraining completed!")
    print("Average training loss:", training_loss)
    print("Model saved:", output_path)

    return {
        "model": trainer.model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "training_loss": training_loss,
        "training_examples": len(train_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "category_tokens": data["category_tokens"],
        "output_dir": str(output_path)
    }




def milestone_4(
    comment,
    model=None,
    tokenizer=None,
    model_dir="models/feedback_finetuned"
):
    """
    Classify new customer feedback using
    the fine-tuned GPT-2 model.
    """
    from app.inference import (
        load_feedback_model,
        predict_feedback_category
    )

    if not isinstance(comment, str) or not comment.strip():
        raise ValueError(
            "Please enter a valid feedback comment"
        )

    # Load the trained model if not supplied.
    if model is None or tokenizer is None:
        model, tokenizer = load_feedback_model(
            model_dir=model_dir
        )

    # Get category tokens from Milestone 1.
    dataset = milestone_1()

    result = predict_feedback_category(
        comment=comment,
        model=model,
        tokenizer=tokenizer,
        category_tokens=dataset["category_tokens"]
    )

    return result




def milestone_5(
    comment,
    model=None,
    tokenizer=None,
    model_dir="models/feedback_finetuned"
):
    """
    Prepare a customer feedback classification
    result for display in the Streamlit app.
    """
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError(
            "Please enter a customer feedback comment."
        )

    result = milestone_4(
        comment=comment,
        model=model,
        tokenizer=tokenizer,
        model_dir=model_dir
    )

    categories = {
        "PRODUCT_ISSUE": "Product Issue",
        "STAFF_PRAISE": "Staff Praise",
        "DELIVERY_ISSUE": "Delivery Issue",
        "PRICING_FEEDBACK": "Pricing Feedback",
        "GENERAL_FEEDBACK": "General Feedback"
    }

    predicted = result["predicted_category"]

    return {
        **result,
        "display_category": categories.get(
            predicted,
            predicted.replace("_", " ").title()
        ),
        "confidence_percentage": round(
            result["confidence"] * 100,
            2
        )
    }

