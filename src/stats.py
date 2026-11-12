"""Chi-square and correlation analysis: issue categories vs dissatisfaction."""
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

from src.config import ISSUE_KEYWORDS


def add_issue_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per issue category: True if the tweet mentions it."""
    df = df.copy()
    for issue, keywords in ISSUE_KEYWORDS.items():
        pattern = r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b"
        df[f"issue__{issue}"] = df["clean_text"].str.contains(pattern, regex=True)
    return df


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.values.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0


def issue_vs_dissatisfaction(df: pd.DataFrame) -> list:
    """For each issue: 2x2 chi-square (mentioned? x negative?), Cramer's V and point-biserial correlation."""
    is_neg = df["airline_sentiment"].eq("negative")
    results = []
    for issue in ISSUE_KEYWORDS:
        flag = df[f"issue__{issue}"]
        row = {"issue": issue, "mentions": int(flag.sum())}
        if 5 <= flag.sum() < len(df) - 5:
            table = pd.crosstab(flag, is_neg).reindex(index=[False, True], columns=[False, True], fill_value=0)
            chi2, p, _, _ = chi2_contingency(table.values, correction=True)
            r, _ = pointbiserialr(flag.astype(int), is_neg.astype(int))
            row.update({
                "negative_rate_when_mentioned": float(is_neg[flag].mean()),
                "negative_rate_otherwise": float(is_neg[~flag].mean()),
                "chi2": float(chi2),
                "p_value": float(p),
                "cramers_v": cramers_v(table, chi2),
                "correlation": float(r),
                "significant": bool(p < 0.05),
            })
        results.append(row)
    return sorted(results, key=lambda r: -r.get("negative_rate_when_mentioned", 0))


def reason_by_airline_test(df: pd.DataFrame) -> dict:
    """Is the mix of negative reasons different across airlines? (chi-square on negative tweets)"""
    neg = df[df["airline_sentiment"].eq("negative")]
    table = pd.crosstab(neg["negativereason"], neg["airline"])
    chi2, p, dof, _ = chi2_contingency(table.values)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "cramers_v": cramers_v(table, chi2), "negative_tweets": int(len(neg))}
