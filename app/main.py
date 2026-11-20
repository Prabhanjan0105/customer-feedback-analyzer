"""Streamlit app: run with python -m streamlit run app/main.py."""
import streamlit as st
from app.core import milestone_1, milestone_5
from app.inference import load_feedback_model

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="💬", layout="wide")


@st.cache_resource
def get_model():
    return load_feedback_model("models/feedback_finetuned")


def main():
    st.title("💬 Customer Feedback Analyzer")
    st.write("Automatically categorize retail customer feedback using fine-tuned GPT-2.")
    st.info("Educational demo trained on 30 examples; predictions may be inaccurate.")
    data = milestone_1()
    with st.sidebar:
        st.header("Model Information")
        st.write("Model: GPT-2 (tiny pretrained checkpoint)")
        st.write("Framework: Hugging Face Transformers")
        st.metric("Training examples", data["total_examples"])
        st.metric("Categories", len(data["categories"]))
    try:
        model, tokenizer = get_model()
        st.sidebar.success("Fine-tuned model loaded")
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Cannot load fine-tuned model: {exc}")
        st.code("python extend_tokenizer.py\npython train_model.py", language="powershell")
        st.stop()

    analyze, dataset_tab = st.tabs(["Analyze Feedback", "Training Dataset"])
    with analyze:
        st.subheader("Analyze customer feedback")
        comment = st.text_area(
            "Enter customer feedback",
            placeholder="Example: My package arrived late and the box was damaged.",
            height=130,
        )
        if st.button("Analyze Feedback", type="primary"):
            if not comment.strip():
                st.warning("Please enter feedback before analyzing.")
                st.session_state.pop("feedback_result", None)
            else:
                with st.spinner("Analyzing..."):
                    try:
                        st.session_state["feedback_result"] = milestone_5(
                            comment, model=model, tokenizer=tokenizer
                        )
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Classification failed: {exc}")
                        st.session_state.pop("feedback_result", None)
        result = st.session_state.get("feedback_result")
        if result:
            st.divider()
            st.subheader("Classification Result")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Category", result["display_category"])
            col2.metric("Relative Confidence", f"{result['confidence_percentage']:.2f}%")
            st.subheader("Category Comparison")
            chart_data = {
                category.replace("_", " ").title(): score
                for category, score in result["category_scores"].items()
            }
            st.bar_chart(chart_data, horizontal=True)
            with st.expander("Detailed category scores"):
                for category, score in result["category_scores"].items():
                    st.write(f"{category}: {score * 100:.2f}%")
            st.caption("Scores are relative among the five labels, not calibrated accuracy estimates.")
    with dataset_tab:
        st.subheader("Labeled training dataset")
        selected = st.selectbox("Filter by category", ["All"] + data["categories"])
        rows = [
            row for row in data["dataset"]
            if selected == "All" or row["label"] == selected
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(rows)} training examples.")
    st.divider()
    st.caption("Customer Feedback Analyzer | GPT-2 Fine-Tuning Project")


if __name__ == "__main__":
    main()
