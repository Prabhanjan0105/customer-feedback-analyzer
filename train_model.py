
from app.core import milestone_3

print("=" * 60)
print("CUSTOMER FEEDBACK ANALYZER")
print("MILESTONE 3 - GPT-2 FINE-TUNING")
print("=" * 60)

result = milestone_3(
    model_dir="models/feedback_gpt2",
    output_dir="models/feedback_finetuned",
    epochs=5,
    batch_size=4,
    learning_rate=0.0001,
    max_length=128
)

print("\nTRAINING SUMMARY")
print("-" * 60)

print(
    "Training examples:",
    result["training_examples"]
)

print("Epochs:", result["epochs"])
print("Batch size:", result["batch_size"])

print(
    "Average training loss:",
    round(result["training_loss"], 4)
)

print("\nCategory tokens:")
for token in result["category_tokens"]:
    print(" ", token)

print("\nSaved model:", result["output_dir"])

print("=" * 60)
print("MILESTONE 3 COMPLETED")
print("=" * 60)

