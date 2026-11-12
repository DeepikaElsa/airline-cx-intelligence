"""Complaint prioritisation: choose the best complaints for a fixed support capacity.

Formulated as a 0/1 knapsack:
    maximise   sum(severity_i * x_i)
    subject to sum(minutes_i * x_i) <= capacity,   x_i in {0, 1}
Solved exactly with SciPy's MILP solver (HiGHS).
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.config import (DEFAULT_MINUTES, DEFAULT_WEIGHT, ISSUE_KEYWORDS,
                        ISSUE_MINUTES, ISSUE_WEIGHT)


def build_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Tweets the model predicts as negative (P >= 0.5), with issue, severity and handling time."""
    c = df[df["p_negative"] >= 0.5].copy()

    assigned = pd.Series(None, index=c.index, dtype=object)
    for issue in ISSUE_KEYWORDS:
        hit = c[f"issue__{issue}"] & assigned.isna()
        assigned[hit] = issue
    c["issue"] = assigned.fillna("Other")

    weight = c["issue"].map(ISSUE_WEIGHT).fillna(DEFAULT_WEIGHT)
    # severity = model confidence x issue weight x reach (retweets)
    c["severity"] = c["p_negative"] * weight * (1 + np.log1p(c["retweet_count"]))
    c["minutes"] = c["issue"].map(ISSUE_MINUTES).fillna(DEFAULT_MINUTES).astype(float)
    c["short_text"] = c["text"].astype(str).str.slice(0, 140)

    return c.reset_index(drop=True)[["id", "airline", "issue", "severity", "minutes", "short_text"]]


def select_tickets(complaints: pd.DataFrame, capacity: float) -> np.ndarray:
    """Boolean mask of the complaints to handle within the capacity (agent-minutes)."""
    if complaints.empty:
        return np.zeros(0, dtype=bool)

    severity = complaints["severity"].to_numpy(dtype=float)
    minutes = complaints["minutes"].to_numpy(dtype=float)

    res = milp(
        c=-severity,
        integrality=np.ones(len(severity)),
        bounds=Bounds(0, 1),
        constraints=[LinearConstraint(minutes.reshape(1, -1), 0, capacity)],
        options={"time_limit": 30},
    )
    if res.x is not None:
        return res.x > 0.5

    # Fallback if the solver stops early: greedy by severity per minute
    chosen = np.zeros(len(severity), dtype=bool)
    used = 0.0
    for i in np.argsort(-severity / minutes):
        if used + minutes[i] <= capacity:
            chosen[i] = True
            used += minutes[i]
    return chosen
