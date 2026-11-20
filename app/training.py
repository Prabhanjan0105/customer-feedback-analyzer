
"""Dataset preparation for GPT-2 feedback classification."""

import torch
from torch.utils.data import Dataset


class FeedbackDataset(Dataset):
    """
    Train GPT-2 to predict a category token
    after a customer feedback comment.
    """

    def __init__(
        self,
        examples,
        tokenizer,
        max_length=128
    ):
        self.records = []

        if max_length < 16:
            raise ValueError(
                "max_length must be at least 16"
            )

        for example in examples:
            text = example["text"]
            label = example["label"]

            prompt = (
                f"Feedback: {text}\n"
                "Category:"
            )

            category_token = f"<{label}>"

            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False
            )

            category_ids = tokenizer.encode(
                category_token,
                add_special_tokens=False
            )

            if len(category_ids) != 1:
                raise ValueError(
                    f"Invalid category token: {category_token}"
                )

            # Reserve one position for the category token.
            prompt_ids = prompt_ids[:max_length - 1]

            input_ids = prompt_ids + category_ids

            # Train only on the category prediction.
            labels = (
                [-100] * len(prompt_ids)
                + category_ids
            )

            self.records.append({
                "input_ids": input_ids,
                "labels": labels
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class FeedbackCollator:
    """Pad training batches while masking padded labels."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(
            len(item["input_ids"])
            for item in features
        )

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            ids = item["input_ids"]
            target = item["labels"]

            padding = max_len - len(ids)

            input_ids.append(
                ids + [self.pad_token_id] * padding
            )

            attention_masks.append(
                [1] * len(ids) + [0] * padding
            )

            labels.append(
                target + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long
            )
        }
