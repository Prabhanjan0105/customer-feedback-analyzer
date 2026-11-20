from app.core import milestone_4
from app.inference import load_feedback_model

SAMPLES = [
    "The headphones stopped working after one day.",
    "The staff were extremely friendly and helpful.",
    "My delivery arrived four days late.",
    "The product is too expensive.",
    "The store was clean and well organized.",
]


def show(comment, model, tokenizer):
    result = milestone_4(comment, model=model, tokenizer=tokenizer)
    print("\nFeedback:", result["comment"])
    print("Prediction:", result["predicted_category"])
    print("Relative confidence:", f"{result['confidence'] * 100:.2f}%")
    for category, score in result["category_scores"].items():
        print(f"  {category}: {score * 100:.2f}%")


if __name__ == "__main__":
    print("MILESTONE 4 - CUSTOMER FEEDBACK CLASSIFIER")
    model, tokenizer = load_feedback_model()
    for comment in SAMPLES:
        show(comment, model, tokenizer)
    while True:
        comment = input("\nEnter feedback (or 'exit'): ").strip()
        if comment.lower() == "exit":
            break
        if not comment:
            print("Please enter feedback.")
            continue
        show(comment, model, tokenizer)
