from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Παίρνει μια πρόταση (string) και επιστρέφει:
    - Το συναίσθημα (label: POSITIVE ή NEGATIVE)
    - Το σκορ εμπιστοσύνης του μοντέλου
    """

    result = sentiment_pipeline(text)[0]
    return {
        "label" : result["label"],
        "score" : round(result["score"], 2)
                    }
                    