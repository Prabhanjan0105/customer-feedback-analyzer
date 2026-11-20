
from app.core import milestone_2

print("=" * 60)
print("CUSTOMER FEEDBACK ANALYZER")
print("MILESTONE 2 - EXTEND GPT-2 TOKENIZER")
print("=" * 60)

result = milestone_2(
    model_name="sshleifer/tiny-gpt2",
    output_dir="models/feedback_gpt2"
)

print("\nMODEL INFORMATION")
print("-" * 60)

print("Base model:", result["model_name"])

print(
    "Original vocabulary:",
    result["original_vocab_size"]
)

print(
    "Extended vocabulary:",
    result["new_vocab_size"]
)

print(
    "New tokens added:",
    result["added_tokens"]
)

print("\nCATEGORY TOKEN IDS")

for token, token_id in result["token_ids"].items():
    print(f"{token}: {token_id}")

print("\nEmbedding size:", result["embedding_size"])
print("Saved model:", result["output_dir"])

print("\n" + "=" * 60)
print("MILESTONE 2 COMPLETED")
print("=" * 60)

