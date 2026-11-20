from app.core import milestone_3

if __name__ == "__main__":
    print("MILESTONE 3 - GPT-2 FINE-TUNING")
    result = milestone_3(
        model_dir="models/feedback_gpt2",
        output_dir="models/feedback_finetuned",
        epochs=5, batch_size=4, learning_rate=0.0001, max_length=128,
    )
    print("Training examples:", result["training_examples"])
    print("Epochs:", result["epochs"])
    print("Average training loss:", round(result["training_loss"], 4))
    print("Saved model:", result["output_dir"])
