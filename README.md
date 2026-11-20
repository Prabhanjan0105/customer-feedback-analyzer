# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/bar3ffrqa5/badge.svg)](https://ask.thili.ai/b/bar3ffrqa5)

This Customer Feedback Analyzer helps retail managers quickly understand and prioritize customer concerns by automatically categorizing feedback comments. It leverages a fine-tuned GPT-2 model, similar to tagging legal clauses, to classify comments into categories like 'product issue' or 'staff praise' by generating special label tokens.

## What you are building

**Goal.** You will build a tool that automatically categorizes customer feedback comments using a fine-tuned GPT-2 model, demonstrating the fine-tuning logic and inference pipeline to classify new customer comments.

**Done when.** You will have a Streamlit application where retail managers can input customer feedback comments and receive automatic classifications into predefined categories, proving the model's ability to categorize new feedback.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Identify common themes in customer feedback to define your categories and then manually label a small set of example comments.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Refer to the drill's approach for extending the tokenizer and model with new special tokens for your categories.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer class, as shown in the drill, to manage the fine-tuning process.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Adapt the inference pipeline from the drill to process new customer comments and generate category predictions.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Use Streamlit widgets to create an input text area and display the classification results.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the model to assign multiple categories to a single comment if applicable.
- Implement a feature to display the model's confidence score for each classification.
- Add basic aggregation to show the count of comments per category after processing multiple inputs.
