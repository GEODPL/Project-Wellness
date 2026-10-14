"""
emotional_map.py
================
Συναισθηματικός χάρτης ημέρας (tags) από check-in + ελεύθερο κείμενο.

Κλίμακες που δέχεται (όπως τις δίνει το app σήμερα):
  - mood:  1-10
  - sleep: ώρες ύπνου (1-10)
  - water: ποτήρια (1-15)

Δύο διορθώσεις σε σχέση με την προηγούμενη έκδοση:
1) Η ανίχνευση λέξεων-κλειδιών γίνεται πλέον χωρίς να μετράνε οι τόνοι και
   τα κεφαλαία/πεζά ("ΠΙΕΣΗ", "πιεση", "πίεση", "πιέζομαι" ταιριάζουν όλα).
2) Το HTML φέρνει μαζί του τα στυλ του (inline), ώστε να μην εξαρτάται από
   κλάσεις που δεν υπάρχουν στο style.css. Χρησιμοποιεί τις μεταβλητές της
   παλέτας (var(--twilight-*)) με fallback χρώματα.
"""

import unicodedata
from typing import Any, List, Tuple


def _as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _fold(text: str) -> str:
    """Πεζά + αφαίρεση τόνων/διαλυτικών, π.χ. 'ΠΊΕΣΗ' -> 'πιεση'."""
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


# (στελέχη χωρίς τόνους, emoji, ετικέτα) -- ίδια λογική με πριν, σε ενιαία λίστα
_TEXT_RULES: List[Tuple[Tuple[str, ...], str, str]] = [
    (("αγχ",), "😟", "Άγχος"),
    (("πιεσ", "πιεζ", "πολλα"), "🟠", "Πίεση"),
    (("κουρασ", "εξαντ"), "💤", "Κούραση"),
    (("ελπι",), "💛", "Ελπίδα"),
    (("μοναξ",), "🤍", "Μοναξιά"),
]


def extract_emotional_tags(mood: int, sleep: int, water: int, text: str) -> List[Tuple[str, str]]:
    t = _fold(text)
    mood = _as_int(mood, 5)
    sleep = _as_int(sleep, 7)
    water = _as_int(water, 6)

    tags: List[Tuple[str, str]] = []

    # Όρια διάθεσης: τα παλιά 20/40/60/80 (κλίμακα 0-100) διαιρεμένα με 10
    if mood <= 2:
        tags.append(("💙", "Θλίψη"))
    elif mood <= 4:
        tags.append(("😕", "Αναστάτωση"))
    elif mood <= 6:
        tags.append(("😐", "Αβεβαιότητα"))
    elif mood <= 8:
        tags.append(("🙂", "Ήπια ηρεμία"))
    else:
        tags.append(("😄", "Θετική διάθεση"))

    for stems, emoji, label in _TEXT_RULES:
        if any(stem in t for stem in stems):
            tags.append((emoji, label))

    # Παλιά διαστήματα "0–2" και "3–5" ώρες ύπνου -> sleep <= 5
    if sleep <= 5:
        tags.append(("😴", "Ανάγκη για ξεκούραση"))
    # Παλιά διαστήματα "0" και "1–3" ποτήρια -> water <= 3
    if water <= 3:
        tags.append(("💧", "Ανάγκη για φροντίδα σώματος"))

    seen = set()
    uniq: List[Tuple[str, str]] = []
    for emoji, label in tags:
        if label not in seen:
            uniq.append((emoji, label))
            seen.add(label)

    return uniq[:4]


_CARD_STYLE = (
    "margin:0;padding:0;"
)
_TITLE_STYLE = (
    "font-family:var(--font-display,Georgia,serif);font-weight:600;"
    "font-size:0.95rem;margin-bottom:0.5rem;color:var(--twilight-deep,#2E2440);"
)
_TAGS_STYLE = "display:flex;flex-wrap:wrap;gap:0.4rem;"
_PILL_STYLE = (
    "display:inline-block;padding:0.28rem 0.75rem;border-radius:999px;"
    "font-size:0.84rem;font-weight:600;white-space:nowrap;"
    "color:var(--twilight-deep,#2E2440);"
    "background:rgba(183,157,242,0.18);"
    "border:1px solid rgba(183,157,242,0.35);"
)


def render_emotional_map(mood: int, sleep: int, water: int, text: str) -> str:
    tags = extract_emotional_tags(mood, sleep, water, text)
    if not tags:
        return ""

    pills_html = "".join(
        f"<span class='emotion-pill' style='{_PILL_STYLE}'>{emoji} {label}</span>"
        for emoji, label in tags
    )

    # Σκόπιμα σε μία γραμμή, χωρίς εσοχές/κενές γραμμές, ώστε ο Markdown
    # parser του st.markdown να μην το διαβάσει ποτέ ως code block.
    return (
        f"<div class='emotion-map-card' style='{_CARD_STYLE}'>"
        f"<div class='emotion-map-title' style='{_TITLE_STYLE}'>🧠 Συναίσθημα ημέρας:</div>"
        f"<div class='emotion-map-tags' style='{_TAGS_STYLE}'>{pills_html}</div>"
        f"</div>"
    )
