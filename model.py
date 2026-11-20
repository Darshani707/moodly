"""
SentimentAI – PyTorch LSTM Model, Preprocessing, and Inference Pipeline.

This module strictly mirrors the original Google Colab IMDb training
implementation:
- Architecture: SentimentLSTM (Embedding(52692, 100, padding_idx=0) ->
  LSTM(100, 128, batch_first=True) with pack_padded_sequence -> Linear(128, 1))
- Preprocessing: lowercase, remove HTML <br> tags, retain [a-z\\s], collapse
  whitespace, whitespace tokenization, pad/truncate to 300 tokens.
- Inference: Sigmoid activation over raw logits with a 0.5 decision threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model & Preprocessing Constants (Matching Original Training Configuration)
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
VOCAB_SIZE = 52692
MAX_SEQ_LEN = 300
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncodedReview:
    """Encoded token sequence and preprocessing statistics for a single text."""

    padded_indices: list[int]
    actual_length: int
    original_text: str
    cleaned_text: str
    word_count: int
    char_count: int
    preprocessed_token_count: int
    unknown_word_count: int
    used_token_count: int
    truncated_tokens: list[str]


@dataclass(frozen=True)
class PredictionResult:
    """Complete sentiment prediction output and technical diagnostics."""

    sentiment: str
    positive_probability: float
    negative_probability: float
    raw_logit: float
    device: str
    encoded: EncodedReview


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class SentimentLSTM(nn.Module):
    """Exact PyTorch LSTM architecture used during IMDb training."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        padding_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting padded token IDs and pre-padding sequence lengths.

        Args:
            text: LongTensor of shape (batch_size, max_seq_len).
            lengths: LongTensor of shape (batch_size,) with actual token counts.

        Returns:
            Raw logit tensor of shape (batch_size,).
        """
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed_embedded)
        output = self.fc(hidden[-1])
        return output.squeeze(1)


# ---------------------------------------------------------------------------
# Exact Text Preprocessing & Tokenization
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean raw review text using the exact training regex pipeline."""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text by whitespace splitting."""
    return clean_text(text).split()


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_seq_len: int = MAX_SEQ_LEN,
) -> EncodedReview:
    """
    Encode input text into padded vocabulary indices and record actual length.

    Rules enforced:
    - Index 0 reserved for <PAD>, Index 1 reserved for <UNK>.
    - Truncate sequences to max_seq_len (300) tokens.
    - If empty after preprocessing, encode as a single <UNK> token (length=1).
    - Pad remaining positions up to max_seq_len with <PAD>.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()

    pad_idx = int(vocab.get(PAD_TOKEN, PAD_IDX))
    unk_idx = int(vocab.get(UNK_TOKEN, UNK_IDX))

    word_count = len(text.split())
    char_count = len(text)
    preprocessed_token_count = len(tokens)

    if preprocessed_token_count == 0:
        truncated_tokens: list[str] = []
        encoded_indices = [unk_idx]
        unknown_word_count = 1
        actual_length = 1
    else:
        truncated_tokens = tokens[:max_seq_len]
        encoded_indices = [int(vocab.get(token, unk_idx)) for token in truncated_tokens]
        unknown_word_count = sum(1 for token in truncated_tokens if token not in vocab)
        actual_length = len(encoded_indices)

    pad_count = max_seq_len - len(encoded_indices)
    padded_indices = encoded_indices + [pad_idx] * pad_count

    return EncodedReview(
        padded_indices=padded_indices,
        actual_length=actual_length,
        original_text=text,
        cleaned_text=cleaned,
        word_count=word_count,
        char_count=char_count,
        preprocessed_token_count=preprocessed_token_count,
        unknown_word_count=unknown_word_count,
        used_token_count=actual_length,
        truncated_tokens=truncated_tokens,
    )


# ---------------------------------------------------------------------------
# Artifact Loading & Inference
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Select CUDA GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocabulary(vocab_path: str | Path) -> dict[str, int]:
    """Load and validate the word-to-index vocabulary JSON dictionary."""
    path = Path(vocab_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found at '{path}'. "
            "Place your exported 'vocab.json' inside the 'models/' directory."
        )

    with path.open("r", encoding="utf-8") as file:
        vocab = json.load(file)

    if not isinstance(vocab, dict) or len(vocab) == 0:
        raise ValueError(
            f"Invalid vocabulary format in '{path}'. Expected a non-empty JSON object."
        )

    return {str(token): int(index) for token, index in vocab.items()}


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Normalize raw state_dict or wrapped checkpoint dictionaries."""
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(
            checkpoint["model_state_dict"], dict
        ):
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format in model.pth. "
            "Expected a PyTorch state_dict dictionary."
        )

    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    return state_dict


def load_trained_model(
    model_path: str | Path,
    vocab_size: int = VOCAB_SIZE,
    device: torch.device | None = None,
) -> tuple[SentimentLSTM, torch.device]:
    """
    Instantiate SentimentLSTM and load the saved state_dict on CPU or GPU.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights file not found at '{path}'. "
            "Place your trained 'model.pth' inside the 'models/' directory."
        )

    target_device = device if device is not None else get_device()

    try:
        checkpoint = torch.load(path, map_location=target_device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=target_device)

    state_dict = _extract_state_dict(checkpoint)

    if "embedding.weight" in state_dict:
        checkpoint_vocab_size, checkpoint_emb_dim = state_dict["embedding.weight"].shape
        if checkpoint_emb_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Checkpoint embedding dimension ({checkpoint_emb_dim}) does not match "
                f"expected architecture ({EMBEDDING_DIM})."
            )
        if checkpoint_vocab_size != vocab_size:
            raise ValueError(
                f"Checkpoint vocabulary size ({checkpoint_vocab_size:,}) does not match "
                f"expected vocabulary size ({vocab_size:,})."
            )

    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=PAD_IDX,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device)
    model.eval()

    return model, target_device


def predict_sentiment(
    text: str,
    model: SentimentLSTM,
    vocab: dict[str, int],
    device: torch.device,
) -> PredictionResult:
    """
    Run end-to-end preprocessing, encoding, and PyTorch LSTM inference.
    """
    if not text or not text.strip():
        raise ValueError("Please enter non-empty English text before running analysis.")

    encoded = encode_text(text=text, vocab=vocab, max_seq_len=MAX_SEQ_LEN)

    input_tensor = torch.tensor(
        [encoded.padded_indices],
        dtype=torch.long,
        device=device,
    )
    length_tensor = torch.tensor(
        [encoded.actual_length],
        dtype=torch.long,
    )

    model.eval()
    with torch.no_grad():
        output = model(input_tensor, length_tensor)
        raw_logit = float(output.item())
        probability = float(torch.sigmoid(output).item())

    negative_probability = 1.0 - probability
    sentiment = "Positive" if probability >= DECISION_THRESHOLD else "Negative"

    return PredictionResult(
        sentiment=sentiment,
        positive_probability=probability,
        negative_probability=negative_probability,
        raw_logit=raw_logit,
        device=str(device),
        encoded=encoded,
    )
