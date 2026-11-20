
from app.core import milestone_4
from app.inference import load_feedback_model


def main():
    print("=" * 60)
    print("CUSTOMER FEEDBACK ANALYZER")
    print("MILESTONE 4 - FEEDBACK CLASSIFICATION")
    print("=" * 60)

    print("\nLoading fine-tuned GPT-2 model...")

    model, tokenizer = load_feedback_model(
        "models/feedback_finetuned"
    )

    print("Model loaded successfully!")

    sample_comments = [
        "The headphones stopped working after one day.",
        "The staff were extremely friendly and helpful.",
        "My delivery arrived four days late.",
        "The product is too expensive.",
        "The store was clean and well organized."
    ]

    for comment in sample_comments:
        result = milestone_4(
            comment=comment,
            model=model,
            tokenizer=tokenizer
        )

        print("\n" + "-" * 60)
        print("Customer feedback:", comment)
        print(
            "Predicted category:",
            result["predicted_category"]
        )
        print(
            "Relative confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )

        print("Category scores:")

        for category, score in (
            result["category_scores"].items()
        ):
            print(
                f"  {category}: {score * 100:.2f}%"
            )

    print("\n" + "=" * 60)
    print("INTERACTIVE FEEDBACK CLASSIFICATION")
    print("=" * 60)

    while True:
        comment = input(
            "\nEnter customer feedback "
            "(or type 'exit'): "
        ).strip()

        if comment.lower() == "exit":
            print("Exiting feedback analyzer.")
            break

        if not comment:
            print("Please enter a comment.")
            continue

        result = milestone_4(
            comment=comment,
            model=model,
            tokenizer=tokenizer
        )

        print(
            "\nPredicted category:",
            result["predicted_category"]
        )
        print(
            "Relative confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
