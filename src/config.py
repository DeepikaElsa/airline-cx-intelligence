"""All settings in one place. Change assumptions here, not inside the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Tweets.csv"
OUT_PATH = ROOT / "public" / "data" / "results.json"

RANDOM_STATE = 42
CV_FOLDS = 5

# Issue categories found from tweet text with keyword rules.
# Order matters: the first match wins when assigning one category per complaint.
ISSUE_KEYWORDS = {
    "Baggage": ["luggage", "baggage", "bag", "bags", "suitcase"],
    "Cancellations": ["cancel", "cancelled", "cancellation", "rebook", "rebooked"],
    "Booking and refunds": ["refund", "refunds", "booking", "ticket", "tickets", "fare", "fee", "charged"],
    "Delays": ["delay", "delayed", "late", "hours", "waiting"],
    "Seats and comfort": ["seat", "seats", "legroom", "comfort"],
    "Food and drink": ["food", "meal", "drink", "snack"],
    "Wi-Fi and entertainment": ["wifi", "wi fi", "internet", "entertainment", "screen"],
    "Customer service": ["service", "rude", "agent", "agents", "staff", "hold", "hung"],
}

# ASSUMPTIONS for prioritisation (not from the dataset - adjust to your support team):
# weight = how much one unresolved complaint hurts the business
# minutes = average support-agent minutes needed to handle one complaint
ISSUE_WEIGHT = {
    "Customer service": 1.5,
    "Baggage": 1.3,
    "Cancellations": 1.2,
    "Booking and refunds": 1.1,
    "Delays": 1.0,
    "Seats and comfort": 0.9,
    "Food and drink": 0.7,
    "Wi-Fi and entertainment": 0.7,
    "Other": 0.8,
}
ISSUE_MINUTES = {
    "Customer service": 15,
    "Baggage": 25,
    "Cancellations": 20,
    "Booking and refunds": 20,
    "Delays": 10,
    "Seats and comfort": 8,
    "Food and drink": 5,
    "Wi-Fi and entertainment": 5,
    "Other": 8,
}
DEFAULT_WEIGHT = 1.0
DEFAULT_MINUTES = 10

# Support capacity levels (agent-minutes) that the dashboard slider can choose.
CAPACITY_STEPS = [30, 60, 120, 180, 240, 300, 480, 600]
MAX_QUEUE_ITEMS = 50  # tickets stored per capacity level for the dashboard list
