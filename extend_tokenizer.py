from app.core import milestone_2

if __name__ == "__main__":
    print("MILESTONE 2 - EXTEND GPT-2 TOKENIZER")
    result = milestone_2()
    print("Base model:", result["model_name"])
    print("Original vocabulary:", result["original_vocab_size"])
    print("New vocabulary:", result["new_vocab_size"])
    print("Added tokens:", result["added_tokens"])
    for token, token_id in result["token_ids"].items():
        print(f"{token}: {token_id}")
    print("Saved to:", result["output_dir"])
