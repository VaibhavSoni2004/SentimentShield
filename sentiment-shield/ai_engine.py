import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure vader_lexicon is safely downloaded on server start
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()


# -----------------------------
# Urgency Keywords
# -----------------------------

high_urgency_keywords = [
    "hacked",
    "stolen",
    "fraud",
    "scam",
    "account frozen",
    "account blocked",
    "money stolen",
    "unauthorized transaction",
    "identity theft",
]

medium_urgency_keywords = [
    "failed",
    "debited",
    "crashed",
    "not working",
    "blocked",
    "frozen",
    "hijacked",
]

low_urgency_keywords = [
    "slow",
    "suggestion",
    "feedback",
    "nice",
    "great",
]


def analyze_text(text: str):
    scores = sia.polarity_scores(text)

    compound = scores["compound"]
    text_lower = text.lower()

    # -----------------------------
    # Check urgency keywords
    # -----------------------------

    high_matches = [
        word for word in high_urgency_keywords
        if word in text_lower
    ]

    medium_matches = [
        word for word in medium_urgency_keywords
        if word in text_lower
    ]

    low_matches = [
        word for word in low_urgency_keywords
        if word in text_lower
    ]

    # -----------------------------
    # Base urgency from sentiment
    # -----------------------------

    if compound < 0:
        base_urgency = 50 + (abs(compound) * 45)
    else:
        base_urgency = max(5, 30 - (compound * 25))

    # -----------------------------
    # Determine urgency
    # -----------------------------

    # HIGH priority keywords always make the issue High
    if high_matches:
        urgency = "High"
        urgency_percentage = max(85, base_urgency)

    # MEDIUM keywords make the issue Medium
    elif medium_matches:
        urgency = "Medium"
        urgency_percentage = max(50, base_urgency)

    # LOW keywords keep the issue Low
    elif low_matches:
        urgency = "Low"
        urgency_percentage = min(35, base_urgency)

    # No urgency keyword → rely on sentiment
    else:
        urgency_percentage = base_urgency

        if urgency_percentage >= 70:
            urgency = "High"
        elif urgency_percentage >= 40:
            urgency = "Medium"
        else:
            urgency = "Low"

    # Keep percentage between 5 and 99
    urgency_percentage = min(
        99,
        max(5, round(urgency_percentage))
    )

    # -----------------------------
    # Determine Sentiment
    # -----------------------------

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # -----------------------------
    # Return result
    # -----------------------------

    return {
        "sentiment": sentiment,
        "polarity_score": round(compound, 2),
        "urgency": urgency,
        "urgency_score": urgency_percentage
    }