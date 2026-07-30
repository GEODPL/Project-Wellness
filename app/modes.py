"""
modes.py
========
"Dynamic Mode Blending": ανιχνεύει από το κείμενο του χρήστη (και από το
check-in mood/sleep/water) ποιο θεματικό πλαίσιο ταιριάζει περισσότερο
—φοιτητικό άγχος, εργασιακό στρες, ύπνος, ή σχέσεις— ώστε το chat να
προσαρμόσει ανάλογα το ύφος/εστίασή του.

Δεν εξαρτάται από Streamlit ή Supabase, μόνο από απλό κείμενο και νούμερα,
οπότε είναι εύκολο να δοκιμαστεί ξεχωριστά (unit tests) αν θες αργότερα.
"""

from typing import Any, Dict, List, Optional


MODES: Dict[str, Dict[str, Any]] = {
    "student": {
        "label": "Φοιτητική ευεξία",
        "keywords": [
            "σχολή", "σχολη", "πανεπιστ", "διάβασ", "διαβασ", "μάθημα", "μαθημα",
            "εξετασ", "προθεσμ", "deadline", "παράδοση", "παραδοση",
            "δεν προλαβαίνω", "δεν προλαβαινω", "είμαι πίσω", "ειμαι πισω",
            "αναβλητικ", "κοπωση", "γνωστικ", "διάσπαση", "συγκέντρ", "συγκεντρ",
        ],
        "linguistic_patterns": ["δεν προλαβαινω", "είμαι πίσω", "ειμαι πισω"],
    },
    "work": {
        "label": "Εργασιακό στρες",
        "keywords": [
            "δουλει", "εργασια", "job", "office", "γραφειο", "αφεντικ",
            "burnout", "καμμένος", "καμενος", "τρέχω", "τρεχω", "δεν φτάνω", "δεν φτανω",
            "όρια", "ορια", "υπερωρ", "πίεση", "πιεση",
        ],
        "weight_sleep_water": True,
    },
    "sleep": {
        "label": "Ύπνος & αποκατάσταση",
        "keywords": [
            "ύπν", "υπν", "αϋπν", "αυπν", "ξυπν", "ξύπν",
            "σκέψεις", "σκεψεις", "υπερένταση", "υπερενταση",
            "δεν κοιμ", "δεν μπορω να κοιμ", "δεν μπορώ να κοιμ",
        ],
        "low_threshold": True,
    },
    "relationships": {
        "label": "Σχέσεις & οικογένεια",
        "keywords": [
            "μαμα", "μπαμπ", "γονει", "αδελφ", "οικογεν",
            "σχέση", "σχεση", "αγόρι", "αγορι", "κοπέλα", "κοπελα",
            "σύντρο", "συντρο", "χωρισ", "ζήλια", "ζηλια",
            "τσακω", "σύγκρου", "συγκρου", "παρεξηγ", "ένταση", "ενταση",
            "απόρριψη", "απορριψη", "μοναξ", "δεν με καταλαβαίνουν", "δεν με καταλαβαινουν",
            "όρια", "ορια",
        ],
        "relational": True,
    },
}


def detect_sleep_difficulty(sleep_hours: int, text: str) -> bool:
    """True αν οι ώρες ύπνου είναι λίγες Ή αν το κείμενο αναφέρει δυσκολία ύπνου."""
    try:
        s = int(sleep_hours)
    except (TypeError, ValueError):
        s = 0

    t = (text or "").lower()
    poor_sleep_hours = s <= 6
    sleep_keywords = [
        "αϋπν", "αυπν",
        "δεν κοιμ", "δεν μπορω να κοιμ", "δεν μπορώ να κοιμ",
        "ξυπν", "ξύπν",
        "δεν κοιμηθ", "λίγο ύπνο", "λιγο υπνο",
        "σκέψεις πριν τον ύπνο", "σκεψεις πριν τον υπνο",
        "υπερένταση", "υπερενταση",
    ]
    mentions_sleep = any(w in t for w in sleep_keywords)
    return poor_sleep_hours or mentions_sleep


def score_keywords(text: str, keywords: List[str]) -> int:
    """Μετράει πόσες λέξεις-κλειδιά από τη λίστα εμφανίζονται στο κείμενο."""
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw in t)


def detect_mode_weights(text: str, mood: int, sleep: int, water: int) -> Dict[str, float]:
    """
    Επιστρέφει ένα dict {mode_name: weight} που δείχνει πόσο "ταιριάζει"
    κάθε mode (student/work/sleep/relationships) με το μήνυμα + το check-in.
    Τα weights αθροίζουν σε ~1.0 (ενισχύοντας τα 2 κορυφαία).
    """
    t = (text or "").lower()

    scores = {
        "student": score_keywords(t, MODES["student"]["keywords"]),
        "work": score_keywords(t, MODES["work"]["keywords"]),
        "sleep": score_keywords(t, MODES["sleep"]["keywords"]),
        "relationships": score_keywords(t, MODES["relationships"]["keywords"]),
    }

    # boosts
    for p in MODES["student"].get("linguistic_patterns", []):
        if p in t:
            scores["student"] += 2

    if (sleep <= 6 or water <= 4) and scores["work"] > 0:
        scores["work"] += 2

    if detect_sleep_difficulty(sleep, text):
        scores["sleep"] += 2

    relational_boost = any(
        x in t for x in ["δεν με καταλαβ", "δεν με καταλαβα", "απόρριψ", "απορριψ", "μοναξ", "τσακω", "συγκρου"]
    )
    if relational_boost:
        scores["relationships"] += 2

    if all(v == 0 for v in scores.values()):
        if sleep <= 6:
            scores["sleep"] = 2
        elif mood <= 5:
            scores["student"] = 1
            scores["work"] = 1
        else:
            scores["student"] = 1

    total = sum(scores.values())
    weights = {k: (v / total) for k, v in scores.items()} if total else {k: 0.0 for k in scores}

    sorted_keys = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    top = sorted_keys[:2]
    for k in weights:
        if k not in top:
            weights[k] *= 0.6
    ren = sum(weights.values())
    if ren > 0:
        weights = {k: v / ren for k, v in weights.items()}

    return weights


def choose_active_mode_from_weights(weights: Dict[str, float]) -> Optional[str]:
    """Επιστρέφει το επικρατέστερο mode, μόνο αν έχει αρκετά υψηλό weight (>=0.34)."""
    if not weights:
        return None
    best = max(weights, key=lambda k: weights[k])
    return best if float(weights.get(best, 0.0)) >= 0.34 else None


def format_mode_blend_text(weights: Dict[str, float]) -> str:
    """Μορφοποιεί τα top-2 weights σε αναγνώσιμο κείμενο, για το explainable layer."""
    order = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return ", ".join([f"{k}={v:.2f}" for k, v in order])