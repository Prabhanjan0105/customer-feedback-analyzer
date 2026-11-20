
"""Streamlit application for Customer Feedback Analyzer."""

import streamlit as st

from app.core import milestone_1, milestone_5
from app.inference import load_feedback_model


st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="💬",
    layout="wide"
)


@st.cache_resource
def get_model():
    """Load the fine-tuned GPT-2 model once."""
    return load_feedback_model(
        "models/feedback_finetuned"
    )


def main():
    st.title("💬 Customer Feedback Analyzer")

    st.write(
        "Automatically categorize retail customer "
        "feedback using a fine-tuned GPT-2 model."
    )

    st.info(
        "This is an educational model trained on a "
        "small dataset. Predictions may be inaccurate."
    )

    # Sidebar
    st.sidebar.title("⚙️ Model Information")
    st.sidebar.write("Model: GPT-2")
    st.sidebar.write("Framework: Hugging Face")
    st.sidebar.write("Task: Feedback Classification")

    dataset = milestone_1()

    st.sidebar.metric(
        "Training Examples",
        dataset["total_examples"]
    )

    st.sidebar.metric(
        "Feedback Categories",
        len(dataset["categories"])
    )

    # Load trained model
    try:
        model, tokenizer = get_model()
        st.sidebar.success("Fine-tuned model loaded")

    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Unable to load the model: {error}")
        st.warning(
            "Train the model before starting the app."
        )
        st.code("python train_model.py")
        st.stop()

    tab1, tab2 = st.tabs([
        "🔍 Analyze Feedback",
        "📚 Training Dataset"
    ])

    with tab1:
        st.subheader("Analyze Customer Feedback")

        comment = st.text_area(
            "Enter customer feedback",
            placeholder=(
                "Example: My package arrived late "
                "and the box was damaged."
            ),
            height=130
        )

        if st.button(
            "Analyze Feedback",
            type="primary"
        ):
            if not comment.strip():
                st.warning(
                    "Please enter feedback before analyzing."
                )
            else:
                with st.spinner(
                    "Analyzing customer feedback..."
                ):
                    try:
                        result = milestone_5(
                            comment=comment,
                            model=model,
                            tokenizer=tokenizer
                        )

                        st.session_state[
                            "feedback_result"
                        ] = result

                    except (ValueError, RuntimeError) as error:
                        st.error(
                            f"Classification failed: {error}"
                        )

        if "feedback_result" in st.session_state:
            result = st.session_state[
                "feedback_result"
            ]

            st.divider()
            st.subheader("📊 Classification Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Category",
                    result["display_category"]
                )

            with col2:
                st.metric(
                    "Relative Confidence",
                    f"{result['confidence_percentage']:.2f}%"
                )

            st.subheader("Category Comparison")

            # Scores are relative probabilities among
            # the five allowed category tokens.
            chart_data = {
                category.replace("_", " ").title(): score
                for category, score in (
                    result["category_scores"].items()
                )
            }

            st.bar_chart(
                chart_data,
                horizontal=True
            )

            with st.expander(
                "View detailed category scores"
            ):
                for category, score in (
                    result["category_scores"].items()
                ):
                    st.write(
                        f"{category}: {score * 100:.2f}%"
                    )

            st.caption(
                "Relative scores are not calibrated "
                "accuracy probabilities."
            )

    with tab2:
        st.subheader("Training Feedback Dataset")

        st.write(
            "Explore the labeled examples used "
            "for GPT-2 fine-tuning."
        )

        selected_category = st.selectbox(
            "Filter by category",
            ["All"] + dataset["categories"]
        )

        filtered = [
            item
            for item in dataset["dataset"]
            if (
                selected_category == "All"
                or item["label"] == selected_category
            )
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing {len(filtered)} "
            "training examples."
        )

    st.divider()

    st.caption(
        "Customer Feedback Analyzer | "
        "GPT-2 Fine-Tuning Project"
    )


if __name__ == "__main__":
    main()
