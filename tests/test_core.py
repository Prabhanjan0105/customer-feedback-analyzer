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



# Milestone 3: Fine-tune the model using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

