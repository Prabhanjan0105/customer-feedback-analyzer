from app.core import milestone_1

result = milestone_1()
print("=" * 60)
print("CUSTOMER FEEDBACK ANALYZER - DATASET")
print("=" * 60)
print("Total examples:", result["total_examples"])
for category, count in result["category_counts"].items():
    print(f"{category}: {count} examples")
for item in result["dataset"]:
    print(f"{item['id']:02d} | {item['label']} | {item['text']}")
print("\nSample GPT-2 training example:")
print(result["training_examples"][0]["text"])
