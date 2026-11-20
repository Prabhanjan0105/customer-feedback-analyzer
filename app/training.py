"""Dataset and batch collator for GPT-2 category-token training."""
import torch
from torch.utils.data import Dataset


class FeedbackDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        if max_length < 16:
            raise ValueError("max_length must be at least 16")
        self.records = []
        for example in examples:
            prompt = f"Feedback: {example['text']}\nCategory:"
            category = f"<{example['label']}>"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            category_ids = tokenizer.encode(category, add_special_tokens=False)
            if len(category_ids) != 1:
                raise ValueError(f"Invalid category token: {category}")
            # Preserve the end of the comment and the 'Category:' cue when truncating.
            prompt_ids = prompt_ids[-(max_length - 1):]
            input_ids = prompt_ids + category_ids
            labels = [-100] * len(prompt_ids) + category_ids
            self.records.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class FeedbackCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        longest = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for example in features:
            n = longest - len(example["input_ids"])
            ids.append(example["input_ids"] + [self.pad_token_id] * n)
            masks.append([1] * len(example["input_ids"]) + [0] * n)
            labels.append(example["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
