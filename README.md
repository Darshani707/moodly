# SentimentAI – Intelligent Text Sentiment Analyzer

A modern **Streamlit** web application powered by your existing trained **PyTorch `SentimentLSTM`** model for binary sentiment analysis of English text and movie reviews.

## Key Features

- **Exact Trained Model Integration**: Recreates `SentimentLSTM` (`Embedding(52692, 100, padding_idx=0)` → `LSTM(100, 128, batch_first=True)` with `pack_padded_sequence` → `Linear(128, 1)`) and loads your saved `model.pth` state dictionary with `model.eval()` and `torch.no_grad()`.
- **Training-Exact Preprocessing**: Applies lowercase conversion, `<br>` tag removal, `[a-z\s]` filtering, whitespace normalization, `<PAD>` (`0`) / `<UNK>` (`1`) vocabulary encoding, 300-token truncation, and length packing.
- **Interactive Prediction & Confidence UI**: Displays Positive/Negative prediction cards, horizontal probability bars, Plotly confidence charts, live word/character counters, and clickable example reviews.
- **Technical Diagnostics**: Expandable panel inspecting raw text, cleaned text, token counts, `<UNK>` counts, and truncated sequence length.
- **Session Prediction History & CSV Export**: Tracks session-scoped predictions in `st.session_state` with one-click CSV export and history reset (nothing is saved permanently to disk).

---

## Project Structure

```text
SentimentAI/
│
├── app.py                  # Streamlit web interface & visualizations
├── model.py                # SentimentLSTM architecture, preprocessing & inference
├── requirements.txt        # Python dependencies
├── README.md               # Setup, local execution & cloud deployment guide
│
├── .streamlit/
│   └── config.toml         # Light theme & Streamlit server settings
│
├── models/
│   ├── model.pth           # <-- PLACE YOUR TRAINED PYTORCH STATE_DICT HERE
│   └── vocab.json          # <-- PLACE YOUR VOCABULARY JSON HERE
│
└── assets/
    └── logo.png            # SentimentAI brand mark
```

---

## Step-by-Step Setup & Local Execution

### 1. Place Your Trained Model Files
Copy the two artifacts exported from your Google Colab training session into `SentimentAI/models/`:

- `SentimentAI/models/model.pth` — Your saved PyTorch `state_dict` (`torch.save(model.state_dict(), "model.pth")`).
- `SentimentAI/models/vocab.json` — Your word-to-index dictionary (`52,692` tokens with `"<PAD>": 0` and `"<UNK>": 1`).

### 2. Create a Virtual Environment & Install Dependencies
```bash
cd SentimentAI
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Streamlit Application Locally
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (typically `http://localhost:8501`).

---

## Deploying to Streamlit Community Cloud

1. Push the `SentimentAI/` directory to a GitHub repository.
   - Ensure `models/model.pth` and `models/vocab.json` are committed (if `model.pth` exceeds 100 MB, track it with **Git LFS**).
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **New app**, select your repository and branch, and set **Main file path** to `app.py` (or `SentimentAI/app.py` if located in a subfolder).
4. Click **Deploy!** — Streamlit Community Cloud automatically installs `requirements.txt` and launches the app.

---

## Model Specifications

| Attribute | Value |
| :--- | :--- |
| Model | LSTM (Long Short-Term Memory) |
| Framework | PyTorch |
| Dataset | IMDb Movie Reviews (25,000 train / 25,000 test) |
| Embedding Dimension | 100 |
| Hidden Dimension | 128 |
| Maximum Sequence Length | 300 |
| Vocabulary Size | 52,692 |
| Training Epochs | 10 (Adam, lr=0.001, BCEWithLogitsLoss) |
| Reported Test Accuracy | ~84.09% |
| Classes | Positive (`>= 0.5`), Negative (`< 0.5`) |

> **Note**: This model was trained by the project owner using Google Colab and has not been independently validated beyond the reported IMDb test split results.
