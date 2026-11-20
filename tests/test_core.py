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



# Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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

