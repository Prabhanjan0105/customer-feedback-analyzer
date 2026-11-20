"""Tests for all five milestones."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_milestone_1():
    from app.core import milestone_1
    result = milestone_1()
    assert len(result["categories"]) == 5
    assert result["total_examples"] == 30
    assert len(result["dataset"]) == 30
    assert len(result["training_examples"]) == 30
    assert len({x["text"].casefold() for x in result["dataset"]}) == 30
    assert all(result["category_counts"][c] == 6 for c in result["categories"])
    assert result["category_tokens"] == [f"<{c}>" for c in result["categories"]]


def test_milestone_2(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.core import milestone_2
    output = tmp_path / "extended"
    result = milestone_2(output_dir=str(output))
    assert result["added_tokens"] == 5
    assert result["new_vocab_size"] == result["original_vocab_size"] + 5
    assert result["embedding_size"] == result["new_vocab_size"]
    assert all(len(result["tokenizer"].encode(t, add_special_tokens=False)) == 1
               for t in result["category_tokens"])
    assert (output / "config.json").exists()
    restored_tokenizer = AutoTokenizer.from_pretrained(output)
    restored_model = AutoModelForCausalLM.from_pretrained(output)
    assert restored_model.get_input_embeddings().weight.shape[0] == len(restored_tokenizer)


def test_milestone_3(tmp_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from app.core import milestone_2, milestone_3
    from app.training import FeedbackDataset, FeedbackCollator
    model_dir = tmp_path / "extended"
    milestone_2(output_dir=str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataset = FeedbackDataset([{"text": "Late delivery", "label": "DELIVERY_ISSUE"}], tokenizer)
    assert len(dataset) == 1
    assert dataset[0]["labels"][-1] != -100
    assert all(x == -100 for x in dataset[0]["labels"][:-1])
    batch = FeedbackCollator(tokenizer.pad_token_id)([dataset[0]])
    assert batch["input_ids"].shape == batch["labels"].shape
    output = tmp_path / "finetuned"
    result = milestone_3(model_dir=str(model_dir), output_dir=str(output), epochs=1)
    assert result["training_examples"] == 30
    assert math.isfinite(result["training_loss"])
    restored = AutoModelForCausalLM.from_pretrained(output)
    assert restored.get_input_embeddings().weight.shape[0] == len(
        AutoTokenizer.from_pretrained(output)
    )


def test_milestone_4():
    import torch
    from torch import nn
    from app.core import milestone_4

    class FakeTokenizer:
        truncation_side = "right"
        def encode(self, text, add_special_tokens=False):
            return [{"<PRODUCT_ISSUE>": 10, "<STAFF_PRAISE>": 11,
                     "<DELIVERY_ISSUE>": 12, "<PRICING_FEEDBACK>": 13,
                     "<GENERAL_FEEDBACK>": 14}[text]]
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])}

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
        def forward(self, **kwargs):
            logits = torch.zeros(1, 3, 20)
            logits[0, -1, 12] = 10
            return SimpleNamespace(logits=logits)

    result = milestone_4("My package was late.", model=FakeModel(), tokenizer=FakeTokenizer())
    assert result["predicted_category"] == "DELIVERY_ISSUE"
    assert result["predicted_token"] == "<DELIVERY_ISSUE>"
    assert 0 <= result["confidence"] <= 1
    assert len(result["category_scores"]) == 5
    assert sum(result["category_scores"].values()) == pytest.approx(1, abs=0.001)
    with pytest.raises(ValueError):
        milestone_4("", model=FakeModel(), tokenizer=FakeTokenizer())


def test_milestone_5(monkeypatch):
    from app import core
    def fake_classifier(comment, model=None, tokenizer=None, model_dir=None):
        return {"comment": comment, "predicted_category": "DELIVERY_ISSUE",
                "predicted_token": "<DELIVERY_ISSUE>", "confidence": 0.85,
                "category_scores": {"DELIVERY_ISSUE": 0.85}}
    monkeypatch.setattr(core, "milestone_4", fake_classifier)
    result = core.milestone_5("Package late", model=object(), tokenizer=object())
    assert result["display_category"] == "Delivery Issue"
    assert result["confidence_percentage"] == 85.0
    with pytest.raises(ValueError):
        core.milestone_5("   ")
