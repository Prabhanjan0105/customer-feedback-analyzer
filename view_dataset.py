
from app.core import milestone_1

result = milestone_1()

print("=" * 60)
print("CUSTOMER FEEDBACK ANALYZER - DATASET")
print("=" * 60)

print("\nTotal examples:", result["total_examples"])

print("\nFEEDBACK CATEGORIES")
for category in result["categories"]:
    count = result["category_counts"][category]
    print(f"{category}: {count} examples")

print("\nLABELED CUSTOMER COMMENTS")

for item in result["dataset"]:
    print("-" * 60)
    print("ID:", item["id"])
    print("Comment:", item["text"])
    print("Category:", item["label"])

print("\nSAMPLE GPT-2 TRAINING EXAMPLE")
print("=" * 60)
print(result["training_examples"][0]["text"])

