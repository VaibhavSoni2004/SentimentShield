import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_text(text: str):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    text_lower = text.lower()
    urgent_keywords = ["urgent", "immediately", "asap", "failed", "debited", "stolen", "hacked", "now"]
    has_urgent_word = any(word in text_lower for word in urgent_keywords)
    
    if compound <= -0.4 or (has_urgent_word and compound < 0.2):
        urgency = "High"
    elif compound < 0 or has_urgent_word:
        urgency = "Medium"
    else:
        urgency = "Low"
        
    return {
        "sentiment": sentiment,
        "polarity_score": round(compound, 2),
        "urgency": urgency
    }