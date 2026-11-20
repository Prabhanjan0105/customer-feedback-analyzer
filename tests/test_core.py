"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest


# Milestone 1: Create labeled feedback dataset
def test_milestone_1():
    from app.core import milestone_1

    result = milestone_1()

    assert isinstance(result, dict)

    assert "categories" in result
    assert "category_tokens" in result
    assert "dataset" in result
    assert "training_examples" in result
    assert "total_examples" in result
    assert "category_counts" in result

    # Verify category count
    assert len(result["categories"]) == 5

    # Verify dataset size
    assert result["total_examples"] == 30
    assert len(result["dataset"]) == 30

    # Verify valid labels and comments
    for item in result["dataset"]:
        assert "id" in item
        assert "text" in item
        assert "label" in item

        assert item["text"].strip()
        assert item["label"] in result["categories"]

    # Verify no duplicate comments
    comments = [
        item["text"].casefold()
        for item in result["dataset"]
    ]

    assert len(comments) == len(set(comments))

    # Verify category tokens
    assert "<PRODUCT_ISSUE>" in result["category_tokens"]
    assert "<STAFF_PRAISE>" in result["category_tokens"]
    assert "<DELIVERY_ISSUE>" in result["category_tokens"]
    assert "<PRICING_FEEDBACK>" in result["category_tokens"]
    assert "<GENERAL_FEEDBACK>" in result["category_tokens"]

    # Verify balanced dataset
    for category in result["categories"]:
        assert result["category_counts"][category] == 6

    # Verify training examples
    assert len(result["training_examples"]) == 30

    for example in result["training_examples"]:
        assert "Feedback:" in example["text"]
        assert "Category:" in example["text"]
        assert example["label"] in result["categories"]




# Milestone 2: Extend GPT-2 tokenizer and model
def test_milestone_2(tmp_path):
    from app.core import milestone_2

    result = milestone_2(
        model_name="sshleifer/tiny-gpt2",
        output_dir=str(tmp_path / "extended_model")
    )

    # Verify result structure
    assert isinstance(result, dict)

    assert "model" in result
    assert "tokenizer" in result
    assert "category_tokens" in result
    assert "token_ids" in result

    # Verify five category tokens
    assert len(result["category_tokens"]) == 5
    assert result["added_tokens"] == 5

    # Verify vocabulary extension
    assert (
        result["new_vocab_size"]
        == result["original_vocab_size"] + 5
    )

    # Verify model embedding size
    assert (
        result["embedding_size"]
        == result["new_vocab_size"]
    )

    tokenizer = result["tokenizer"]
    model = result["model"]

    assert (
        model.get_input_embeddings().weight.shape[0]
        == len(tokenizer)
    )

    # Verify category tokens are recognized individually
    for token in result["category_tokens"]:
        token_id = result["token_ids"][token]

        encoded = tokenizer.encode(
            token,
            add_special_tokens=False
        )

        assert encoded == [token_id]

    # Verify saved model and tokenizer
    from pathlib import Path

    output_path = Path(result["output_dir"])

    assert (output_path / "config.json").exists()

    assert (
        (output_path / "model.safetensors").exists()
        or (output_path / "pytorch_model.bin").exists()
    )

    # Verify model can be reloaded
    from transformers import AutoModelForCausalLM

    restored = AutoModelForCausalLM.from_pretrained(
        str(output_path)
    )

    assert (
        restored.get_input_embeddings().weight.shape[0]
        == result["new_vocab_size"]
    )




# Milestone 3: Fine-tune GPT-2 using Hugging Face Trainer
def test_milestone_3(tmp_path):
    import math

    from app.core import milestone_3
    from app.training import (
        FeedbackDataset,
        FeedbackCollator
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer
    )

    model_dir = "models/feedback_gpt2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    # Verify dataset preparation
    examples = [
        {
            "text": "The headphones are broken.",
            "label": "PRODUCT_ISSUE"
        },
        {
            "text": "The staff were very helpful.",
            "label": "STAFF_PRAISE"
        }
    ]

    dataset = FeedbackDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=64
    )

    assert len(dataset) == 2

    for item in dataset:
        assert len(item["input_ids"]) == len(
            item["labels"]
        )

        # Only the category token should be trained
        assert item["labels"][-1] != -100

        assert all(
            label == -100
            for label in item["labels"][:-1]
        )

    collator = FeedbackCollator(
        pad_token_id=tokenizer.pad_token_id
    )

    batch = collator([
        dataset[0],
        dataset[1]
    ])

    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == (
        batch["input_ids"].shape
    )

    # Run a short real training session
    output_dir = tmp_path / "finetuned"

    result = milestone_3(
        model_dir=model_dir,
        output_dir=str(output_dir),
        epochs=1,
        batch_size=4,
        learning_rate=0.0001,
        max_length=64
    )

    assert result["training_examples"] == 30
    assert result["epochs"] == 1

    assert math.isfinite(
        result["training_loss"]
    )

    assert result["training_loss"] >= 0

    assert (output_dir / "config.json").exists()

    assert (
        (output_dir / "model.safetensors").exists()
        or (output_dir / "pytorch_model.bin").exists()
    )

    # Verify the trained model can be reloaded
    restored_model = (
        AutoModelForCausalLM.from_pretrained(
            str(output_dir)
        )
    )

    restored_tokenizer = (
        AutoTokenizer.from_pretrained(
            str(output_dir)
        )
    )

    assert (
        restored_model.get_input_embeddings()
        .weight.shape[0]
        == len(restored_tokenizer)
    )

    assert len(
        restored_tokenizer.additional_special_tokens
    ) == 5

# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

