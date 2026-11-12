"""Run the whole analysis and write public/data/results.json for the dashboard.

Usage (from the project root):
    python -m src.pipeline
    python -m src.pipeline --data data/Tweets.csv --out public/data/results.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (CAPACITY_STEPS, DATA_PATH, ISSUE_KEYWORDS, ISSUE_MINUTES,
                        ISSUE_WEIGHT, MAX_QUEUE_ITEMS, OUT_PATH)
from src.preprocess import LABELS, load_and_clean
from src.prioritize import build_complaints, select_tickets
from src.sentiment import evaluate, out_of_fold_probabilities
from src.stats import add_issue_flags, issue_vs_dissatisfaction, reason_by_airline_test


def issue_rates(df: pd.DataFrame) -> list:
    is_neg = df["airline_sentiment"].eq("negative")
    rows = []
    for issue in ISSUE_KEYWORDS:
        f = df[f"issue__{issue}"]
        rows.append({
            "issue": issue,
            "mentions": int(f.sum()),
            "rate_mentioned": float(is_neg[f].mean()) if f.any() else None,
            "rate_not_mentioned": float(is_neg[~f].mean()) if (~f).any() else None,
        })
    return sorted(rows, key=lambda r: -(r["rate_mentioned"] or 0))


def priority_grid(df: pd.DataFrame) -> dict:
    complaints = build_complaints(df)
    total_severity = float(complaints["severity"].sum())
    grid = []
    for capacity in CAPACITY_STEPS:
        chosen = complaints[select_tickets(complaints, capacity)].sort_values("severity", ascending=False)
        covered = float(chosen["severity"].sum())
        grid.append({
            "capacity_minutes": capacity,
            "tickets": int(len(chosen)),
            "minutes_used": float(chosen["minutes"].sum()),
            "severity_covered": round(covered, 3),
            "share_of_total_severity": round(covered / total_severity, 4) if total_severity else 0.0,
            "items": chosen.head(MAX_QUEUE_ITEMS).round({"severity": 3}).to_dict("records"),
        })
    return {"open_complaints": int(len(complaints)), "grid": grid}


def summarize(df: pd.DataFrame) -> dict:
    counts = df["airline_sentiment"].value_counts().reindex(LABELS, fill_value=0)
    reasons = df.loc[df["airline_sentiment"].eq("negative"), "negativereason"].value_counts().head(10)
    return {
        "tweets": int(len(df)),
        "sentiment_counts": {k: int(v) for k, v in counts.items()},
        "negative_reasons": [{"reason": r, "count": int(c)} for r, c in reasons.items()],
        "issues": issue_rates(df),
        "priority": priority_grid(df),
    }


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def main():
    ap = argparse.ArgumentParser(description="Airline customer feedback analysis")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    df, n_raw = load_and_clean(args.data)
    print(f"Rows: {n_raw:,} raw -> {len(df):,} after cleaning")

    print("Sentiment model: 5-fold out-of-fold predictions ...")
    proba = out_of_fold_probabilities(df)
    df["p_negative"] = proba[:, 0]
    model = evaluate(df["airline_sentiment"].to_numpy(), proba)
    print(f"  accuracy={model['accuracy']:.3f}  macro-F1={model['macro_f1']:.3f}  "
          f"Brier(neg)={model['calibration']['brier_negative']:.3f}")

    df = add_issue_flags(df)
    issue_tests = issue_vs_dissatisfaction(df)
    reason_test = reason_by_airline_test(df)

    scopes = {"All": summarize(df)}
    for airline in sorted(df["airline"].unique()):
        scopes[airline] = summarize(df[df["airline"] == airline])

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"source": Path(args.data).name, "raw_rows": n_raw, "rows": int(len(df)),
                    "airlines": int(df["airline"].nunique())},
        "model": {"name": "TF-IDF + Logistic Regression, sigmoid calibration", "cv_folds": 5, **model},
        "stats": {"issue_vs_dissatisfaction": issue_tests, "reason_by_airline": reason_test},
        "assumptions": {"issue_weight": ISSUE_WEIGHT, "issue_minutes": ISSUE_MINUTES,
                        "capacity_steps": CAPACITY_STEPS},
        "scopes": scopes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
