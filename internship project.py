import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def sentiment_analysis(keyword):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(keyword)
    
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Example usage
keyword = input("Enter a keyword: ")
result = sentiment_analysis(keyword)
print("Sentiment Analysis Result:", result)

