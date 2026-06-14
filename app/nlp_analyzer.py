import spacy
import re

# Φορτώνουμε το μεσαίο (md) ελληνικό μοντέλο
try:
    nlp = spacy.load("el_core_news_md")
except OSError:
    # Fallback σε περίπτωση που τρέχεις τοπικά και δεν έχει κατέβει
    import spacy.cli
    spacy.cli.download("el_core_news_md")
    nlp = spacy.load("el_core_news_md")

def extract_linguistic_features(text: str) -> dict:
    """
    Αναλύει το κείμενο του χρήστη και εξάγει γλωσσολογικά/μορφολογικά metrics.
    """
    if not text.strip():
        return {"word_count": 0, "absolutes_count": 0, "negations_count": 0, "verbs_count": 0, "has_urgency": False}

    doc = nlp(text)
    
    # 1. Απόλυτες λέξεις (Γνωστική Διαστρέβλωση: "Όλα ή τίποτα")
    absolute_words = {"πάντα", "ποτέ", "όλοι", "κανείς", "κανένας", "τίποτα", "συνέχεια", "απολύτως"}
    absolutes_count = sum(1 for token in doc if token.lemma_.lower() in absolute_words)
    
    # 2. Αρνητικά Μόρια
    negations_count = sum(1 for token in doc if token.text.lower() in {"δεν", "μην", "μη"})
    
    # 3. Ρήματα (Δείκτης ενεργητικότητας)
    verbs_count = sum(1 for token in doc if token.pos_ == "VERB")
    
    # 4. Λεκτικό Επείγοντος (RegEx για άμεσο εντοπισμό κινδύνου/άγχους)
    urgency_patterns = [r"δεν αντέχω", r"καταρρέω", r"δεν μπορώ άλλο", r"έχω πελαγώσει", r"πνίγομαι", r"κρίση πανικού"]
    has_urgency = any(re.search(p, text.lower()) for p in urgency_patterns)

    # 5. Αριθμός λέξεων
    word_count = len([token for token in doc if not token.is_punct])

    return {
        "word_count": word_count,
        "absolutes_count": absolutes_count,
        "negations_count": negations_count,
        "verbs_count": verbs_count,
        "has_urgency": has_urgency
    }