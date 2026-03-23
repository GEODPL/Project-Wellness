def extract_emotional_tags(mood: int, sleep: str, water: str, text: str):
    t = text.lower()
    tags: list[tuple[str, str]] = []

    if mood <= 20:
        tags.append(("💙", "Θλίψη"))
    elif mood <= 40:
        tags.append(("😕", "Αναστάτωση"))
    elif mood <= 60:
        tags.append(("😐", "Αβεβαιότητα"))
    elif mood <= 80:
        tags.append(("🙂", "Ήπια ηρεμία"))
    else:
        tags.append(("😄", "Θετική διάθεση"))

    if "άγχ" in t or "αγχος" in t:
        tags.append(("😟", "Άγχος"))
    if "πίεσ" in t or "πιεζ" in t or "πολλά" in t or "πολλα" in t:
        tags.append(("🟠", "Πίεση"))
    if "κουρασ" in t or "κουράσ" in t or "εξαντ" in t:
        tags.append(("💤", "Κούραση"))
    if "ελπί" in t or "ελπι" in t:
        tags.append(("💛", "Ελπίδα"))
    if "μοναξ" in t:
        tags.append(("🤍", "Μοναξιά"))

    if sleep in ["0–2", "3–5"]:
        tags.append(("💛", "Ανάγκη για ξεκούραση"))
    if water in ["0", "1–3"]:
        tags.append(("💧", "Ανάγκη για φροντίδα σώματος"))

    seen = set()
    uniq = []
    for emoji, label in tags:
        if label not in seen:
            uniq.append((emoji, label))
            seen.add(label)

    return uniq[:4]


def render_emotional_map(mood: int, sleep: str, water: str, text: str) -> str:
    tags = extract_emotional_tags(mood, sleep, water, text)
    if not tags:
        return ""

    pills_html = " ".join(
        f"<span class='emotion-pill'>{emoji} {label}</span>"
        for emoji, label in tags
    )

    return f"""
    <div class="emotion-map-card">
        <div class="emotion-map-title">🧠 Συναίσθημα ημέρας:</div>
        <div class="emotion-map-tags">
            {pills_html}
        </div>
    </div>
    """
