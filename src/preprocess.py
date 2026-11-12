"""Load, clean and normalise the airline tweet dataset."""
import re

import pandas as pd

REQUIRED_COLUMNS = ["text", "airline_sentiment", "airline"]
LABELS = ["negative", "neutral", "positive"]

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    """Lowercase, remove URLs and @mentions, keep hashtag words, strip punctuation and emojis."""
    text = _URL.sub(" ", text.lower())
    text = _MENTION.sub(" ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_clean(path: str):
    """Return (clean DataFrame, number of raw rows)."""
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    n_raw = len(raw)
    df = raw.copy()

    # Duplicates: prefer tweet_id, otherwise same text from the same airline
    if "tweet_id" in df.columns:
        df = df.drop_duplicates(subset=["tweet_id"])
    else:
        df = df.drop_duplicates(subset=["text", "airline"])

    # Missing values: drop rows without text, sentiment or airline
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["airline_sentiment"] = df["airline_sentiment"].astype(str).str.strip().str.lower()
    df = df[df["airline_sentiment"].isin(LABELS)].copy()

    df["clean_text"] = df["text"].astype(str).map(clean_text)
    df = df[df["clean_text"].str.len() > 2].copy()

    # Reason is only meaningful for negative tweets
    reason = df["negativereason"] if "negativereason" in df.columns else pd.Series(None, index=df.index, dtype=object)
    df["negativereason"] = reason.fillna("Unspecified").where(df["airline_sentiment"].eq("negative"), "Not negative")

    rt = df["retweet_count"] if "retweet_count" in df.columns else pd.Series(0, index=df.index)
    df["retweet_count"] = pd.to_numeric(rt, errors="coerce").fillna(0)

    df = df.reset_index(drop=True)
    df["id"] = df["tweet_id"].astype(str) if "tweet_id" in df.columns else df.index.astype(str)
    return df, n_raw
