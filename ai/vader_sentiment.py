from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any

class VaderSentimentAnalyzer:
    """Wrapper for VADER sentiment analysis."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_headlines(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyze a list of headlines and return the average compound score.
        Range: -1 (Most Negative) to +1 (Most Positive)
        """
        if not headlines:
            return {"score": 0.0, "label": "Neutral", "count": 0}
            
        total_score = 0.0
        for headline in headlines:
            vs = self.analyzer.polarity_scores(headline)
            total_score += vs['compound']
            
        avg_score = total_score / len(headlines)
        
        if avg_score >= 0.05:
            label = "Positive"
        elif avg_score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return {
            "score": avg_score,
            "label": label,
            "count": len(headlines)
        }
