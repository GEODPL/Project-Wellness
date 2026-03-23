import pandas as pd

# Φόρτωση λεξικού
df = pd.read_csv("app/greek_sentiment_lexicon.tsv", sep="\t")
df = df[["Term", "Polarity1"]].dropna()
df.columns = ["word", "polarity"]

def analyze_lexicon_sentiment(text: str) -> dict:
    words = text.lower().split()
    pos, neg = 0, 0

    for word in words:
        match = df[df["word"] == word]
        if not match.empty:
            polarity = match["polarity"].values[0]
            if polarity.lower() == "positive":
                pos += 1
            elif polarity.lower() == "negative":
                neg += 1

    if pos > neg:
        label = "positive"
    elif neg > pos:
        label = "negative"
    else:
        label = "neutral"

    return {
        "positive": pos,
        "negative": neg,
        "score": pos - neg,
        "label": label
    }