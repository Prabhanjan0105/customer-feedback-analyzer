# Run Customer Feedback Analyzer on Windows

Use Python 3.12 in VS Code PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python view_dataset.py
python extend_tokenizer.py
python train_model.py
python classify_feedback.py
python -m pytest -v
python -m streamlit run app/main.py
```

If `py` is unavailable, use the installed Python 3.12 executable instead.

Training with `sshleifer/tiny-gpt2` is a demonstration. The 30-item dataset is not enough to validate production accuracy. The model weights in `models/` are generated locally and ignored by Git.
