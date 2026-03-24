import os
import re
import html
import socket
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
import pandas as pd
import qrcode
from PIL import Image

from llm import llm_therapeutic_reply, llm_exercise_followup
from rules import (
    personal_reply,
    fallback_therapeutic_reply,
    exercise_suggestion,
    is_emergency,
    emergency_message,
)
from emotional_map import render_emotional_map
from components import (
    render_message,
    render_exercise_card,
    render_emergency_block,
    render_action_plan_card,
)
from data_logger import log_user_data
# Η νέα, 100% ιδιωτική διαχείριση του προφίλ στη μνήμη του browser
def load_profile():
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    return st.session_state.user_profile

def save_profile(new_profile):
    st.session_state.user_profile = new_profile


# ============================================================
# CSS
# ============================================================

def load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================
# MOBILE URL + QR
# ============================================================

def get_local_url(port: int = 8501) -> str:
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    return f"http://{local_ip}:{port}"


def generate_qr_image(url: str) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        box_size=4,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    if hasattr(img, "get_image"):
        img = img.get_image()
    return img


# ============================================================
# SMALL HELPERS
# ============================================================

# ============================================================
# NARRATIVE CONTINUITY + MODE BLENDING + EXPLAINABLE LAYER
# ============================================================

def _safe_trim(text: str, max_chars: int = 1200) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


def remember_user_message(user_text: str, max_keep: int = 30) -> None:
    t = (user_text or "").strip()
    if not t:
        return
    st.session_state.last_user_messages.append(t)
    st.session_state.last_user_messages = st.session_state.last_user_messages[-max_keep:]


def remember_bot_output(bot_text: str, max_keep: int = 20) -> None:
    t = (bot_text or "").strip()
    if not t:
        return
    st.session_state.last_bot_outputs.append(t)
    st.session_state.last_bot_outputs = st.session_state.last_bot_outputs[-max_keep:]


def get_last_bot_outputs(n: int = 8) -> List[str]:
    return (st.session_state.get("last_bot_outputs") or [])[-n:]


def build_compact_context(max_pairs: int = 10) -> str:
    """
    Compact context για narrative continuity:
    παίρνει τα τελευταία N ζεύγη user/bot από st.session_state.messages
    (όχι maps/exercises/plans/emergency).
    """
    pairs = []
    count = 0
    for sender, content in reversed(st.session_state.get("messages", [])):
        if sender not in ("user", "bot"):
            continue
        role = "Χρήστης" if sender == "user" else "Βοηθός"
        pairs.append(f"{role}: {str(content).strip()}")
        count += 1
        if count >= max_pairs * 2:
            break
    pairs.reverse()
    return "\n".join(pairs).strip()


def detect_phase_and_closure(text: str) -> Tuple[str, bool, List[str]]:
    """
    Explainable phase detector.
    Επιστρέφει: (phase, closing_now, signals)
    phase: opening | exploration | deepening | closing
    """
    t = (text or "").strip().lower()
    signals = []

    if len(st.session_state.get("messages", [])) <= 1:
        signals.append("Πρώτο turn → opening")
        return "opening", False, signals

    # closure heuristic 
    closing_phrases = [
        "ευχαρισ", "οκ", "ok", "εντάξει", "τελ", "τέλος", "αυτά", "τα λεμε", "τα λέμε",
        "καληνυχ", "bye", "αντίο", "αντιο"
    ]
    is_short = len(t) <= 25
    has_close = any(p in t for p in closing_phrases)
    closing_now = bool(is_short and has_close)
    if closing_now:
        signals.append("Σύντομο μήνυμα + closing token → closing")
        return "closing", True, signals

    # deepening heuristics (αυτοαναφορά + έντονα συναισθήματα + μοτίβα)
    deep_tokens = ["φοβά", "ντρέπ", "ενοχ", "ενοχή", "νιωθ", "πονά", "δεν αντέχ", "πανικ", "άγχ"]
    if any(tok in t for tok in deep_tokens) and len(t) > 80:
        signals.append("Μεγάλο μήνυμα + affect tokens → deepening")
        return "deepening", False, signals

    # exploration default
    signals.append("Default → exploration")
    return "exploration", False, signals


def compute_mode_weights(text: str, mood: int, sleep: int, water: int) -> Tuple[Dict[str, float], List[str]]:
    """
    Dynamic Mode Blending:
    Παράγει weights για student/work/sleep/relationships, κανονικοποιημένα.
    Επιστρέφει weights + explain signals.
    """
    t = (text or "").lower()
    signals = []

    raw = {
        "student": _score_keywords(t, MODES["student"]["keywords"]),
        "work": _score_keywords(t, MODES["work"]["keywords"]),
        "sleep": _score_keywords(t, MODES["sleep"]["keywords"]),
        "relationships": _score_keywords(t, MODES["relationships"]["keywords"]),
    }

    # boosts 
    for p in MODES["student"].get("linguistic_patterns", []):
        if p in t:
            raw["student"] += 2
            signals.append(f"Student boost: pattern '{p}'")

    if detect_sleep_difficulty(sleep, text):
        raw["sleep"] += 2
        signals.append("Sleep boost: sleep difficulty signal")

    if (sleep <= 6 or water <= 4) and raw["work"] > 0:
        raw["work"] += 2
        signals.append("Work boost: low sleep/water + work keywords")

    rel_boost = any(x in t for x in ["δεν με καταλαβ", "απόρριψ", "μοναξ", "τσακω", "συγκρου"])
    if rel_boost:
        raw["relationships"] += 2
        signals.append("Relationships boost: relational distress tokens")

    # baseline from check-in if strained
    if looks_a_bit_strained(mood, sleep, water):
        raw["sleep"] += 1
        signals.append("Baseline: strained check-in → +sleep weight")

    # if everything is zero, keep neutral distribution but slightly favor sleep (safety)
    total = sum(raw.values())
    if total <= 0:
        signals.append("No strong keywords → fallback weights")
        return {"student": 0.15, "work": 0.15, "sleep": 0.45, "relationships": 0.25}, signals

    weights = {k: v / total for k, v in raw.items()}
    return weights, signals


def pick_primary_mode(weights: Dict[str, float]) -> Optional[str]:
    if not weights:
        return None
    best = max(weights, key=lambda k: weights[k])
    return best if weights[best] >= 0.34 else None


def blend_mode_instructions(weights: Dict[str, float]) -> str:
    """
    Μετατρέπει τα weights σε blended coaching instructions που περνάνε στο LLM.
    """
    if not weights:
        return ""


    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:2]
    top_modes = [k for k, _ in top]

    chunks = []
    if "student" in top_modes:
        chunks.append("Πλαίσιο σπουδών: μικρά, ρεαλιστικά βήματα, αποφυγή τελειομανίας, ήπια δομή.")
    if "work" in top_modes:
        chunks.append("Πλαίσιο εργασίας: όρια, αποφόρτιση, ιεράρχηση, αποφυγή υπερ-ευθύνης.")
    if "sleep" in top_modes:
        chunks.append("Πλαίσιο ύπνου/αποκατάστασης: πολύ ήπιος τόνος, μικρές παρεμβάσεις, μείωση διέγερσης.")
    if "relationships" in top_modes:
        chunks.append("Πλαίσιο σχέσεων: συναίσθημα/ανάγκες/όρια, κατανόηση χωρίς επίθεση, μικρές πράξεις σύνδεσης.")

    wtxt = ", ".join([f"{k}={weights[k]:.2f}" for k in top_modes])
    return f"Dynamic mode blending (top): {wtxt}. " + " ".join(chunks)


def build_dialogue_llm_user_prompt(latest_user_text: str, profile: Dict[str, Any], weights: Dict[str, float], phase: str) -> str:
    """
    Narrative continuity prompt:
    - Inject compact context
    - Inject blended instructions
    - Enforce: no wrap-up bundles unless phase==closing
    - Anti-repeat: avoid last bot outputs
    """
    ctx = build_compact_context(max_pairs=10)
    blended = blend_mode_instructions(weights)
    avoid = get_last_bot_outputs(6)

    avoid_block = "\n".join([f"- {a}" for a in avoid]) if avoid else "- (κανένα)"

    rules = (
        "Στόχος: φυσικός διάλογος με συνοχή.\n"
        "Κανόνες:\n"
        "- Μη δίνεις διάγνωση ή ιατρικές οδηγίες.\n"
        "- Μην εμφανίζεις σύνοψη/χάρτη/άσκηση εκτός αν σου ζητηθεί ρητά ή αν είμαστε σε closing phase.\n"
        "- Μην επαναλάβεις πρόσφατες φράσεις (δες avoid list).\n"
        "- Κράτα 1–2 ερωτήσεις μόνο αν είναι απαραίτητο.\n"
    )

    phase_rule = "closing" if phase == "closing" else "dialogue"
    return (
        f"{rules}\n"
        f"Τρέχουσα φάση: {phase_rule}\n"
        f"{('Blending οδηγίες: ' + blended) if blended else ''}\n\n"
        f"Πλαίσιο προφίλ (σύντομα): name={profile.get('name','')}, context={profile.get('context','')}, tone={profile.get('preferred_tone','')}\n\n"
        f"Ιστορικό (compact):\n{ctx}\n\n"
        f"Avoid επαναλήψεων:\n{avoid_block}\n\n"
        f"Νέο μήνυμα χρήστη:\n{latest_user_text}"
    )


def infer_age_range(age: Any) -> str:
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return ""
    if age_int <= 0:
        return ""
    if 18 <= age_int <= 24:
        return "18–24"
    if 25 <= age_int <= 34:
        return "25–34"
    if 35 <= age_int <= 44:
        return "35–44"
    return "45+"


def user_says_feels_ok(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        "ειμαι καλα", "είμαι καλά",
        "ειμαι οκ", "είμαι οκ",
        "ειμαι μια χαρα", "είμαι μια χαρά",
        "ολα καλα", "όλα καλά",
        "μια χαρα ειμαι", "μια χαρά είμαι",
    ]
    return any(p in t for p in patterns)


def looks_a_bit_strained(mood: int, sleep: int, water: int) -> bool:
    try:
        m = int(mood)
        s = int(sleep)
        w = int(water)
    except (TypeError, ValueError):
        return False
    return (m <= 6) or (s <= 5) or (w <= 4)


def detect_study_anxiety(text: str) -> bool:
    t = (text or "").lower()
    anxiety_words = ["άγχ", "αγχος", "στρες", "stress", "πανικ", "πίεσ", "πιεσ"]
    study_words = [
        "σπουδ", "σχολή", "σχολη", "πανεπιστ", "εξετάσ", "εξετασ",
        "εργασία", "εργασια", "μαθημα", "μάθημα", "διάβασ", "διαβασ",
        "προθεσμ", "deadline", "παράδοσ", "παραδοσ",
    ]
    has_anxiety = any(w in t for w in anxiety_words)
    has_study = any(w in t for w in study_words)
    return has_anxiety and has_study


def detect_sleep_difficulty(sleep_hours: int, text: str) -> bool:
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

from llm import micro_prompt_with_fallback

def get_chat_tail_from_state(max_n: int = 6) -> list[str]:
    tail = []
    for sender, content in st.session_state.get("messages", []):
        if sender == "user":
            tail.append(str(content or "").strip())
    return [t for t in tail if t][-max_n:]

def get_chat_tail(messages, max_items: int = 10):
    """
    Επιστρέφει τα τελευταία max_items ως καθαρό κείμενο για context injection.
    messages: list[tuple(sender, content)]
    """
    tail = []
    for sender, content in messages[-max_items:]:
        if sender in ("user", "bot"):
            tail.append(f"{sender.upper()}: {content}")
    return "\n".join(tail).strip()

def update_narrative_memory(
    profile: dict,
    active_mode: str,
    user_text: str,
    bot_text: str,
):
    """
    Ενημερώνει:
    - conversation_summary (rolling)
    - open_threads (εκκρεμή θέματα)
    - facts_memory (προαιρετικά)
    """

    from llm import llm_update_memory  

    prev_summary = st.session_state.conversation_summary
    prev_threads = st.session_state.open_threads
    prev_facts = st.session_state.facts_memory

    out = llm_update_memory(
        profile=profile,
        active_mode=active_mode,
        prev_summary=prev_summary,
        prev_threads=prev_threads,
        prev_facts=prev_facts,
        user_text=user_text,
        bot_text=bot_text,
    )

    if out is None:
        
        return

    st.session_state.conversation_summary = out.get("summary", prev_summary) or prev_summary
    st.session_state.open_threads = out.get("threads", prev_threads) or prev_threads
    st.session_state.facts_memory = out.get("facts", prev_facts) or prev_facts

def detect_dialogue_closure(text: str) -> bool:
    """
    Heuristic κλείσιμο διαλόγου.
    Σήματα: ευχαριστώ/οκ/τέλος/καληνύχτα/αυτά ήταν/εντάξει/νομίζω οκ κτλ.
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    closing_phrases = [
        "ευχαρισ", "thanks", "οκ", "ok", "εντάξει", "τελ", "τέλος",
        "αυτά", "αυτα ηταν", "νομίζω είμαι οκ", "νομιζω ειμαι οκ",
        "καληνυχ", "τα λέμε", "τα λεμε", "bye", "αντίο", "αντιο",
        "όλα καλά", "ολα καλα", "that's all", "αρκεί", "αρκεi"
    ]

    
    short = len(t) <= 25
    has_close_word = any(p in t for p in closing_phrases)
    return has_close_word and short


def run_wrapup_bundle(
    mood_value: int,
    sleep: int,
    water: int,
    last_user_text: str,
    profile: dict,
    active_mode: str,
    decision_trace: list,
) -> None:
    """
    Εκτελεί το “τέλος”: συμπεράσματα + χάρτης + άσκηση.
    Όλα μπαίνουν στο chat ως messages.
    """
    
    wrap_prompt = f"""
Θέλω να κλείσεις τον διάλογο με ήρεμο τρόπο.
Δώσε:
(α) 4–7 προτάσεις με βασικά συμπεράσματα/μοτίβα που εμφανίστηκαν,
(β) 2 πολύ μικρά, εφαρμόσιμα βήματα για τις επόμενες 24 ώρες,
(γ) 1 φράση κλεισίματος.
Χωρίς διάγνωση, χωρίς “πρέπει”.
Mode: {active_mode}
""".strip()

    enriched = build_llm_user_prompt(
        raw_user_text=wrap_prompt,
        mood_value=mood_value,
        sleep=sleep,
        water=water,
        active_mode=active_mode,
    )

    closing = llm_therapeutic_reply(
        mood=mood_value,
        sleep=sleep,
        water=water,
        user_text=enriched,
        profile=profile,
        active_mode=active_mode,
    )

    if closing:
        closing = _safe_trim(closing, 1400)
        st.session_state.messages.append(("bot", closing))
        remember_bot_output(closing)
        decision_trace.append("Wrap-up: παρήχθη σύνοψη/συμπεράσματα μέσω LLM.")
    else:
        # fallback κλείσιμο 
        closing_rb = "Σε ευχαριστώ που το μοιράστηκες. Κράτα ένα μικρό, πρακτικό βήμα για σήμερα και πήγαινε απαλά."
        st.session_state.messages.append(("bot", closing_rb))
        remember_bot_output(closing_rb)
        decision_trace.append("Wrap-up: LLM δεν επέστρεψε αποτέλεσμα → απλό fallback κλείσιμο.")

    # 2) Συναισθηματικός χάρτης
    map_html = render_emotional_map(mood_value, sleep, water, last_user_text)
    st.session_state.messages.append(("map", map_html))
    decision_trace.append("Wrap-up: προστέθηκε συναισθηματικός χάρτης.")

    # 3) Άσκηση ημέρας
    ex = exercise_suggestion(mood_value, sleep, water, last_user_text)
    st.session_state.messages.append(("exercise", ex))
    decision_trace.append("Wrap-up: προστέθηκε μικρή άσκηση ημέρας.")




# ============================================================
# MODES (Contextual Coaches)
# ============================================================

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


def _score_keywords(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    score = 0
    for kw in keywords:
        if kw in t:
            score += 1
    return score


def detect_active_mode(text: str, mood: int, sleep: int, water: int) -> Optional[str]:
    """
    Επιστρέφει mode key ή None.
    Heuristic scoring: keywords + context signals.
    """
    t = (text or "").lower()

    scores: Dict[str, int] = {
        "student": _score_keywords(t, MODES["student"]["keywords"]),
        "work": _score_keywords(t, MODES["work"]["keywords"]),
        "sleep": _score_keywords(t, MODES["sleep"]["keywords"]),
        "relationships": _score_keywords(t, MODES["relationships"]["keywords"]),
    }

    # Boost rules
    # Student
    for p in MODES["student"]["linguistic_patterns"]:
        if p in t:
            scores["student"] += 2

    # Work 
    if (sleep <= 6 or water <= 4) and scores["work"] > 0:
        scores["work"] += 2

    # Sleep
    if detect_sleep_difficulty(sleep, text):
        scores["sleep"] += 2

    # Relationships
    relational_boost = any(x in t for x in ["δεν με καταλαβ", "δεν με καταλαβα", "απόρριψ", "απορριψ", "μοναξ", "τσακω", "συγκρου"])
    if relational_boost:
        scores["relationships"] += 2

    # Choose best
    best_mode = max(scores, key=lambda k: scores[k])
    best_score = scores[best_mode]

    # Minimum thresholds per mode
    if best_mode == "sleep":
        return best_mode if best_score >= 2 else None
    return best_mode if best_score >= 3 else None


# ============================================================
# MODE-SPECIFIC PLANS 
# ============================================================

def build_student_week_plan(subfocus: str = "") -> str:
    header = "Μικρό πλάνο εβδομάδας (σπουδές) – πολύ ήπιο και εφαρμόσιμο:\n\n"
    core = (
        "1) **Δύο μικρά blocks την ημέρα (25' + 10')**\n"
        "   Ένα 25' διάβασμα + 10' «κλείσιμο» (σημείωσε τι έμεινε).\n\n"
        "2) **Μία «ελάχιστη νίκη»**\n"
        "   Κάθε μέρα: διάλεξε 1 σελίδα ή 1 άσκηση. Μόνο αυτό αρκεί.\n\n"
        "3) **Reset πριν ξεκινήσεις (4–2–6, 5 κύκλοι)**\n"
        "   Για να πέσει ο τόνος του άγχους πριν ανοίξεις βιβλία.\n\n"
        "Αν είσαι «πίσω», δεν σημαίνει αποτυχία. Συνήθως σημαίνει ότι χρειάζεσαι καλύτερη δομή και λιγότερη πίεση."
    )
    if subfocus:
        return header + f"Εστίαση: **{subfocus}**\n\n" + core
    return header + core


def build_work_stress_plan() -> str:
    return (
        "Μικρό πλάνο αποφόρτισης (εργασιακό στρες):\n\n"
        "1) **Grounding 30–45''**\n"
        "   Κοίτα 3 πράγματα γύρω σου, άκου 2 ήχους, νιώσε 1 σημείο του σώματος να ακουμπά (καρέκλα/πάτωμα).\n\n"
        "2) **Μικρό όριο (μία πρόταση)**\n"
        "   «Μπορώ να το δω, αλλά θα επανέλθω σε αυτό στις __:__». Δεν είναι αγένεια, είναι ρυθμός.\n\n"
        "3) **Αποφόρτιση σώματος**\n"
        "   2 λεπτά τέντωμα ώμων/αυχένα ή μικρή βόλτα στο σπίτι.\n\n"
        "Στόχος δεν είναι περισσότερη παραγωγικότητα. Στόχος είναι να μην «καίγεσαι»."
    )


def build_sleep_restore_plan() -> str:
    return (
        "Μικρό πλάνο ύπνου (ήπιο, μη-ενεργοποιητικό):\n\n"
        "1) **5 λεπτά «κλείσιμο»**\n"
        "   2 αργές αναπνοές + 2 τεντώματα + χαμήλωμα φωτός.\n\n"
        "2) **Αν τρέχουν σκέψεις**\n"
        "   Γράψε 1 πρόταση: «Αυτό που γυρίζει στο μυαλό μου είναι…». Τέλος.\n\n"
        "3) **Ήπια φράση**\n"
        "   «Απόψε κάνω χώρο για ξεκούραση, όχι τελειότητα».\n\n"
        "Αν ο ύπνος σε επηρεάζει έντονα και επίμονα, αξίζει να το συζητήσεις και με επαγγελματία υγείας."
    )


def build_relationships_plan() -> str:
    return (
        "Μικρό relational πλάνο (σχέσεις/οικογένεια):\n\n"
        "1) **Τι χρειάζομαι πραγματικά; (1 πρόταση)**\n"
        "   «Αυτό που χρειάζομαι τώρα είναι…» (π.χ. κατανόηση, χώρο, σαφήνεια).\n\n"
        "2) **Όριο χωρίς επίθεση**\n"
        "   «Σε ακούω, αλλά δεν μπορώ να το συνεχίσω έτσι. Ας το ξαναπιάσουμε πιο ήρεμα αργότερα.»\n\n"
        "3) **Μικρή πράξη σύνδεσης**\n"
        "   Ένα μήνυμα/μια κίνηση που λέει «είμαι εδώ», χωρίς να λύσει τα πάντα.\n\n"
        "Αν νιώθεις απόρριψη ή ότι «δεν σε καταλαβαίνουν», δεν σημαίνει ότι είσαι υπερβολικός/ή. Σημαίνει ότι πονάς."
    )


def mode_relational_exercise(text: str) -> str:
    t = (text or "").strip()
    return (
        "Μικρή relational άσκηση (60''):\n\n"
        "• Αν αυτή η κατάσταση ήταν τίτλος, ποιος θα ήταν;\n"
        "• Τι θα ήθελες να καταλάβει ο άλλος άνθρωπος σε 1 πρόταση;\n"
        "• Ποιο είναι ένα μικρό όριο που θα σε προστάτευε σήμερα;\n\n"
        f"Αν θες, γράψε μόνο την 1 πρόταση: «Θέλω να καταλάβεις ότι…»\n\n"
        f"Σημείωση: {html.escape(t)[:0]}"
    )


# ============================================================
# THOUGHT REFRAME (Rule-based)
# ============================================================

def generate_reframe(thought: str, pattern: str) -> str:
    t = thought.strip()
    if not t:
        return ""
    base_intro = "Μια πιο τρυφερή, ρεαλιστική ματιά πάνω σε αυτή τη σκέψη θα μπορούσε να είναι:\n\n"

    if pattern == "Καταστροφολογία (τα βλέπω όλα χάλια)":
        return (
            base_intro
            + "«Αυτή τη στιγμή ο νους μου πάει στο χειρότερο σενάριο. "
              "Στην πραγματικότητα, υπάρχουν και ενδιάμεσες εκδοχές. "
              "Μπορεί να είναι δύσκολο, αλλά δεν σημαίνει ότι όλα θα καταρρεύσουν.»"
        )
    if pattern == "Όλα ή τίποτα (είτε τέλειο είτε αποτυχία)":
        return (
            base_intro
            + "«Δεν χρειάζεται να είναι όλα τέλεια για να έχουν αξία. "
              "Μπορώ να αναγνωρίσω τα λάθη μου χωρίς να ακυρώνω πλήρως τον εαυτό μου.»"
        )
    if pattern == "Διάβασμα σκέψης (υποθέτω τι σκέφτονται οι άλλοι)":
        return (
            base_intro
            + "«Δεν μπορώ να ξέρω με βεβαιότητα τι σκέφτονται οι άλλοι. "
              "Μπορώ να εξετάσω και την πιθανότητα ότι κάποιοι με βλέπουν πιο ζεστά απ’ όσο φαντάζομαι.»"
        )
    if pattern == "Υπεργενίκευση (μια εμπειρία = πάντα έτσι)":
        return (
            base_intro
            + "«Αυτή η εμπειρία ήταν δύσκολη, αλλά δεν σημαίνει ότι θα είναι πάντα έτσι. "
              "Έχω δικαίωμα να δώσω στον εαυτό μου καινούριες ευκαιρίες.»"
        )
    return (
        base_intro
        + "«Αυτή η σκέψη φαίνεται πολύ βαριά αυτή τη στιγμή. "
          "Μπορώ να την αντιμετωπίσω σαν μια άποψη του μυαλού μου, όχι σαν απόλυτη αλήθεια. "
          "Θα προσπαθήσω να είμαι λίγο πιο επιεικής με τον εαυτό μου καθώς τη σκέφτομαι.»"
    )


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Project Wellness",
    page_icon="🧠",
    layout="wide",
)
load_css()


# ============================================================
# SESSION STATE INIT
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "messages" not in st.session_state:
    st.session_state.messages = []
if "exercise_followup" not in st.session_state:
    st.session_state.exercise_followup = False
if "last_decision_trace" not in st.session_state:
    st.session_state.last_decision_trace = []
if "last_checkin" not in st.session_state:
    st.session_state.last_checkin = {}

# Conversation coherence 
if "conversation_phase" not in st.session_state:
    st.session_state.conversation_phase = "dialogue"  

if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

if "checkin_ctx" not in st.session_state:
    st.session_state.checkin_ctx = None

if "pending_reflection" not in st.session_state:
    st.session_state.pending_reflection = False

if "active_mode" not in st.session_state:
    st.session_state.active_mode = None

if "study_anxiety_count" not in st.session_state:
    st.session_state.study_anxiety_count = 0
if "study_anxiety_plan_given" not in st.session_state:
    st.session_state.study_anxiety_plan_given = False

if "sleep_plan_count" not in st.session_state:
    st.session_state.sleep_plan_count = 0
if "sleep_plan_given" not in st.session_state:
    st.session_state.sleep_plan_given = False

if "reframes_history" not in st.session_state:
    st.session_state.reframes_history = []

if "active_mode" not in st.session_state:
    st.session_state.active_mode = None
if "mode_plan_given" not in st.session_state:
    st.session_state.mode_plan_given = {}

if "microcoach_history" not in st.session_state:
    st.session_state.microcoach_history = {}  

# ============================================================
# NARRATIVE CONTINUITY STATE
# ============================================================
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = "" 

if "open_threads" not in st.session_state:
    st.session_state.open_threads = []  

if "facts_memory" not in st.session_state:
    st.session_state.facts_memory = []  

if "last_bot_outputs" not in st.session_state:
    st.session_state.last_bot_outputs = []  

if "last_user_messages" not in st.session_state:
    st.session_state.last_user_messages = []  


# ============================================================
# LOGIN GATE
# ============================================================

if not st.session_state.logged_in:
    st.markdown(
        """
        <div class="main-wrapper">
          <div class="chat-card">
            <h2 style="margin-top:0;">🧠 Project Wellness – Είσοδος</h2>
            <p style="color:#6C5A9E; font-size:0.96rem;">
              Πριν συνεχίσουμε, γράψε ένα όνομα χρήστη και το Gmail σου.
              Έτσι μπορώ να ξεχωρίζω τα δεδομένα σου στο ίδιο περιβάλλον.
            </p>
        """,
        unsafe_allow_html=True,
    )

    login_name = st.text_input("Όνομα χρήστη", value=st.session_state.user_name)
    login_email = st.text_input("Gmail", value=st.session_state.user_email)

    col_lg1, col_lg2 = st.columns([1, 2])
    with col_lg1:
        if st.button("Είσοδος"):
            email = (login_email or "").strip().lower()
            name = (login_name or "").strip()
            if not name:
                st.error("Γράψε ένα όνομα χρήστη 🙂")
            elif not email or "@gmail.com" not in email:
                st.error("Χρειάζομαι ένα έγκυρο Gmail (π.χ. example@gmail.com).")
            else:
                st.session_state.user_name = name
                st.session_state.user_email = email
                st.session_state.logged_in = True
                st.success(f"Καλώς ήρθες, {name}!")
                st.rerun()

    with col_lg2:
        st.caption(
            "Το email χρησιμοποιείται μόνο για να ξεχωρίζουν τα δεδομένα κάθε χρήστη "
            "στο ίδιο περιβάλλον. Δεν στέλνω πουθενά μηνύματα."
        )

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# APP HEADER + TABS
# ============================================================

user_name = st.session_state.get("user_name", "")
subtitle = (
    f"Καλωσήρθες, {user_name} · μικρά συναισθηματικά check-ins και μικρά αλλά σταθερά βήματα φροντίδας."
    if user_name
    else "Μικρά συναισθηματικά check-ins, μικρά βήματα φροντίδας."
)

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-title-left">🧠 Project Wellness</div>
      <div class="app-subtitle-right">{subtitle}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Chat",
        "Ιστορικό",
        "Στατιστικά",
        "Ασκήσεις",
        "Thought Reframe Studio",
        "Προφίλ",
        "Σύνδεση από κινητό",
        "Σχετικά & Ασφάλεια",
    ]
)

(
    tab_chat,
    tab_history,
    tab_stats,
    tab_ex,
    tab_reframe,
    tab_profile,
    tab_mobile,
    tab_info,
) = tabs

# ============================================================
# CHAT TAB 
# ============================================================

with tab_chat:
    import json
    import hashlib
    from datetime import datetime

    profile = load_profile()

    # --- βασικό προφίλ ---
    has_basic_profile = bool(
        (profile.get("name") or "").strip() or (profile.get("context") or "").strip()
    )

    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>Wellness Edition</h1>
          <p class='page-subtitle'>
            Μίλησέ μου για τη μέρα σου – θα κρατήσουμε συνοχή
            και στο τέλος θα κάνουμε ήρεμο wrap-up.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_basic_profile:
        st.markdown(
            """
            <div class="main-wrapper">
              <div class="chat-card">
                <h4>🔒 Το Chat είναι κλειδωμένο</h4>
                <p style="font-size:1.05rem; font-weight:600; color:#4A3D73;">
                  Για να προχωρήσεις, συμπλήρωσε πρώτα το προφίλ σου.
                </p>
                <p style="color:#6C5A9E;">
                  Πήγαινε στην καρτέλα <strong>«Προφίλ»</strong> και συμπλήρωσε τουλάχιστον:
                </p>
                <ul style="color:#6C5A9E;">
                  <li>ένα όνομα ή ψευδώνυμο</li>
                  <li>ή ένα πλαίσιο ζωής (π.χ. φοιτητής, εργαζόμενος)</li>
                </ul>
                <p style="margin-top:10px; color:#6C5A9E;">
                  Μετά το “Αποθήκευση προφίλ”, το Chat ξεκλειδώνει.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        def _user_key() -> str:
            email = (st.session_state.get("user_email") or "").strip().lower()
            if not email:
                email = (st.session_state.get("user_name") or "anon").strip().lower()
            return email or "anon"

        def _user_store_path() -> str:
            base_dir = os.path.dirname(__file__)
            key = _user_key()
            h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            return os.path.join(base_dir, f"chat_state_{h}.json")

        def _load_persisted_chat_state() -> dict:
            path = _user_store_path()
            if not os.path.exists(path):
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}

        def _save_persisted_chat_state(state: dict) -> None:
            path = _user_store_path()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # ============================================================
        # Core helpers
        # ============================================================
        def _safe_trim(text: str, max_chars: int = 1400) -> str:
            t = (text or "").strip()
            if len(t) <= max_chars:
                return t
            return t[:max_chars].rstrip() + "…"

        def _clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
            try:
                v = int(round(float(x)))
            except Exception:
                v = 0
            return max(lo, min(hi, v))

        def remember_bot_output(text: str) -> None:
            t = (text or "").strip()
            if not t:
                return
            st.session_state.last_bot_outputs.append(t)
            st.session_state.last_bot_outputs = st.session_state.last_bot_outputs[-8:]

        def _get_dialogue_context_text(max_pairs: int = 10) -> str:
            items = []
            count = 0
            for sender, content in st.session_state.messages[::-1]:
                if sender in ("user", "bot"):
                    role = "Χρήστης" if sender == "user" else "Βοηθός"
                    items.append(f"{role}: {str(content).strip()}")
                    count += 1
                if count >= max_pairs * 2:
                    break
            items = list(reversed(items))
            return "\n".join(items).strip()

        def _detect_dialogue_closure(text: str) -> bool:
            t = (text or "").strip().lower()
            if not t:
                return False
            closures = [
                "ευχαριστώ", "ευχαριστω", "οκ", "ok", "εντάξει", "ενταξει",
                "τέλος", "τελος", "τα λέμε", "τα λεμε", "καληνύχτα", "καληνυχτα",
                "αυτά", "αυτα", "νομίζω οκ", "νομιζω οκ", "bye", "αντίο", "αντιο"
            ]
            shortish = len(t) <= 35
            return shortish and any(c in t for c in closures)

        def _score_keywords(text: str, keywords: list[str]) -> int:
            t = (text or "").lower()
            return sum(1 for kw in keywords if kw in t)

        def detect_mode_weights(text: str, mood: int, sleep: int, water: int) -> dict:
            """
            Dynamic Mode Blending: weights για 4 modes.
            Χρησιμοποιεί το MODES dict που έχεις ήδη ορίσει στο app.py.
            """
            t = (text or "").lower()

            scores = {
                "student": _score_keywords(t, MODES["student"]["keywords"]),
                "work": _score_keywords(t, MODES["work"]["keywords"]),
                "sleep": _score_keywords(t, MODES["sleep"]["keywords"]),
                "relationships": _score_keywords(t, MODES["relationships"]["keywords"]),
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

            # baseline όταν όλα 0
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

        def choose_active_mode_from_weights(weights: dict) -> Optional[str]:
            if not weights:
                return None
            best = max(weights, key=lambda k: weights[k])
            return best if float(weights.get(best, 0.0)) >= 0.34 else None

        def _format_mode_blend_text(weights: dict) -> str:
            order = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
            return ", ".join([f"{k}={v:.2f}" for k, v in order])

        def build_llm_user_prompt(
            raw_user_text: str,
            mood_value: int,
            sleep: int,
            water: int,
            active_mode: Optional[str],
            mode_weights: dict,
        ) -> str:
            ctx = _get_dialogue_context_text(max_pairs=10)
            summary = (st.session_state.conversation_summary or "").strip()
            threads = st.session_state.open_threads or []
            facts = st.session_state.facts_memory or []

            # profile snippet
            profile_snip_parts = []
            for k in ["context", "main_goals", "main_struggles", "helpful_things", "preferred_tone", "triggers", "soothing_things"]:
                v = (profile.get(k) or "").strip()
                if v:
                    profile_snip_parts.append(f"- {k}: {v}")
            profile_snip = "\n".join(profile_snip_parts).strip()

            # top-2 modes
            mode_instructions = []
            top2 = sorted(mode_weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
            for mode_key, w in top2:
                label = MODES.get(mode_key, {}).get("label", mode_key)
                mode_instructions.append(f"- {label} (weight={w:.2f})")
            mode_text = "\n".join(mode_instructions) if mode_instructions else "- None"

            avoid = st.session_state.last_bot_outputs[-6:] if st.session_state.last_bot_outputs else []

            prompt = f"""
ΣΥΣΤΗΜΑ ΣΥΝΟΧΗΣ (Narrative Continuity Layer):
- Διατήρησε συνοχή με βάση το ιστορικό και τη rolling memory.
- Στόχος: φυσικός διάλογος, όχι κάθε φορά σύνοψη/χάρτης/άσκηση.
- Μην κάνεις διάγνωση, μην δίνεις ιατρικές οδηγίες.

CHECK-IN:
- mood: {mood_value}/10
- sleep: {sleep}
- water: {water}

DYNAMIC MODE BLEND (Top-2):
{mode_text}
Active mode: {active_mode or "NONE"}

PROFILE SNIPPET:
{profile_snip if profile_snip else "- (no extra profile context)"}

ROLLING SUMMARY:
{summary if summary else "- (empty)"}

OPEN THREADS:
{("- " + "\n- ".join(threads)) if threads else "- (none)"}

FACTS MEMORY:
{("- " + "\n- ".join(facts)) if facts else "- (none)"}

ΤΕΛΕΥΤΑΙΟ CONTEXT:
{ctx if ctx else "- (no prior turns)"}

ANTI-REPEAT:
{("- " + "\n- ".join(avoid)) if avoid else "- (none)"}

ΟΔΗΓΙΕΣ:
- Ελληνικά.
- 4–10 προτάσεις, φυσικές.
- 0–2 ερωτήσεις μόνο αν βοηθούν.
- Όχι bullets.
- Μην εμφανίσεις reasoning μέσα στο κυρίως μήνυμα.

ΜΗΝΥΜΑ ΧΡΗΣΤΗ:
{raw_user_text}
""".strip()
            return prompt

        def update_narrative_memory_llm(
            profile: dict,
            active_mode: Optional[str],
            mode_weights: dict,
            user_text: str,
            bot_text: str,
            decision_trace: list[str],
        ) -> None:
            """
            Προαιρετικό: αν έχεις llm_update_memory στο llm.py, θα ενημερώνει summary/threads/facts.
            Αν δεν υπάρχει, δεν σπάει τίποτα.
            """
            try:
                from llm import llm_update_memory
            except Exception:
                decision_trace.append("Narrative memory: llm_update_memory δεν βρέθηκε → skip.")
                return

            prev_summary = st.session_state.conversation_summary
            prev_threads = st.session_state.open_threads
            prev_facts = st.session_state.facts_memory

            out = llm_update_memory(
                profile=profile,
                active_mode=active_mode or "NONE",
                mode_weights=mode_weights,
                prev_summary=prev_summary,
                prev_threads=prev_threads,
                prev_facts=prev_facts,
                user_text=user_text,
                bot_text=bot_text,
            )
            if not out:
                decision_trace.append("Narrative memory: update failed/None → keep previous.")
                return

            st.session_state.conversation_summary = (out.get("summary") or prev_summary or "").strip()
            st.session_state.open_threads = out.get("threads") or prev_threads or []
            st.session_state.facts_memory = out.get("facts") or prev_facts or []
            decision_trace.append("Narrative memory: ενημερώθηκαν summary/threads/facts.")

        # --- Discreet professional-support indicator ((((non-diagnostic)))) ---
        def compute_support_need_score(
            text: str,
            mood: int,
            sleep: int,
            water: int,
            mode_weights: Optional[dict] = None,
            conversation_summary: str = "",
            open_threads: Optional[list] = None,
        ) -> tuple[int, list[str]]:
            t = (text or "").lower()
            reasons: list[str] = []
            score = 0.0

            if mood <= 3:
                score += 18; reasons.append("πολύ χαμηλή διάθεση (≤3/10)")
            elif mood <= 5:
                score += 10; reasons.append("χαμηλή/μέτρια διάθεση (≤5/10)")

            if sleep <= 4:
                score += 16; reasons.append("πολύ λίγος ύπνος (≤4 ώρες)")
            elif sleep <= 6:
                score += 9; reasons.append("λίγος ύπνος (≤6 ώρες)")

            if water <= 3:
                score += 4; reasons.append("χαμηλή ενυδάτωση (≤3 ποτήρια)")

            high_burden = [
                "δεν αντεχω", "δεν αντέχω", "κουραστηκα", "κουράστηκα",
                "δεν παει αλλο", "δεν πάει άλλο", "ειμαι χαλια", "είμαι χάλια",
                "απελπισ", "απόγνωση", "πανικ", "πανικό", "φοβαμαι", "φοβάμαι",
                "μονος", "μόνος", "μοναξ", "καταρρε", "καταρρέ",
                "δεν μπορω", "δεν μπορώ", "δεν βγαινει", "δεν βγαίνει",
            ]
            hit_burden = any(p in t for p in high_burden)
            if hit_burden:
                score += 14; reasons.append("γλωσσικά σήματα έντονου βάρους/δυσκολίας")

            chronic = ["κάθε μέρα", "καθε μερα", "εδώ και", "εδω και", "μήνες", "μηνες", "εβδομάδες", "εβδομαδες", "πάντα", "παντα"]
            if any(p in t for p in chronic):
                score += 8; reasons.append("πιθανή διάρκεια/επιμονή")

            impairment = [
                "δεν μπορω να λειτουργησω", "δεν μπορώ να λειτουργήσω",
                "δεν μπορω να σηκωθω", "δεν μπορώ να σηκωθώ",
                "δεν μπορω να διαβασω", "δεν μπορώ να διαβάσω",
                "δεν μπορω να δουλεψω", "δεν μπορώ να δουλέψω",
                "δεν τρωω", "δεν τρώω", "δεν μπορω να φαω", "δεν μπορώ να φάω",
            ]
            if any(p in t for p in impairment):
                score += 12; reasons.append("σήματα δυσκολίας λειτουργικότητας")

            if mode_weights:
                sleep_w = float(mode_weights.get("sleep", 0.0))
                work_w = float(mode_weights.get("work", 0.0))
                if sleep_w >= 0.45 and sleep <= 6:
                    score += 6; reasons.append("υψηλό βάρος ‘Ύπνος’ + χαμηλός ύπνος")
                if work_w >= 0.45 and (sleep <= 6 or mood <= 5):
                    score += 5; reasons.append("υψηλό βάρος ‘Εργασία’ + κόπωση/πίεση")

            threads = open_threads or []
            if len(threads) >= 3:
                score += 6; reasons.append("πολλά ανοιχτά θέματα (threads)")
            if conversation_summary and len(conversation_summary) > 260:
                score += 3; reasons.append("πλούσιο summary → πιο σύνθετο φορτίο")

            if len((text or "").strip()) <= 12 and not hit_burden:
                score -= 6

            return _clamp_int(score, 0, 100), reasons

        def support_need_label(score: int) -> tuple[str, str]:
            if score <= 24:
                return (
                    "Χαμηλή ένδειξη",
                    "Αν σε βοηθά, συνέχισε με μικρά βήματα φροντίδας. Αν επιμείνει ή δυσκολεύει την καθημερινότητα, μια συζήτηση με ειδικό μπορεί να είναι υποστηρικτική."
                )
            if score <= 49:
                return (
                    "Μέτρια ένδειξη",
                    "Αν αυτό συνεχίζεται ή σε δυσκολεύει λειτουργικά, θα μπορούσε να βοηθήσει μια συζήτηση με ψυχολόγο, έστω για 1–2 συνεδρίες διερεύνησης."
                )
            if score <= 69:
                return (
                    "Αυξημένη ένδειξη",
                    "Με βάση αυτά που γράφεις και το check-in, φαίνεται ότι θα άξιζε επαγγελματική υποστήριξη σύντομα — όχι επειδή “κάτι πάει λάθος”, αλλά για να μην το σηκώνεις μόνος/η."
                )
            return (
                "Υψηλή ένδειξη",
                "Αυτό ακούγεται αρκετά βαρύ. Αν μπορείς, θα ήταν καλό να μιλήσεις με επαγγελματία μέσα στις επόμενες μέρες. Αν νιώσεις κίνδυνο για την ασφάλειά σου, χρησιμοποίησε άμεσα τις γραμμές βοήθειας."
            )

        def run_wrapup_bundle(
            mood_value: int,
            sleep: int,
            water: int,
            last_user_text: str,
            profile: dict,
            active_mode: Optional[str],
            mode_weights: dict,
            decision_trace: list[str],
        ) -> None:
            wrap_raw = (
                "Κλείσε τον διάλογο ήρεμα.\n"
                "Δώσε:\n"
                "1) 4–7 προτάσεις με συμπεράσματα/μοτίβα που εμφανίστηκαν,\n"
                "2) 2 πολύ μικρά, εφαρμόσιμα βήματα για 24 ώρες,\n"
                "3) 1 φράση κλεισίματος.\n"
                "Χωρίς διάγνωση, χωρίς «πρέπει»."
            )

            wrap_prompt = build_llm_user_prompt(
                raw_user_text=wrap_raw,
                mood_value=mood_value,
                sleep=sleep,
                water=water,
                active_mode=active_mode,
                mode_weights=mode_weights,
            )

            closing = llm_therapeutic_reply(
                mood=mood_value,
                sleep=sleep,
                water=water,
                user_text=wrap_prompt,
                profile=profile,
                active_mode=active_mode,
            )

            if closing:
                closing = _safe_trim(closing, 1500)
                st.session_state.messages.append(("bot", closing))
                remember_bot_output(closing)
                decision_trace.append("Wrap-up: LLM συμπεράσματα/βήματα/κλείσιμο.")
            else:
                rb = "Σε ευχαριστώ που το μοιράστηκες. Κράτα ένα μικρό βήμα φροντίδας για σήμερα και πήγαινε απαλά."
                st.session_state.messages.append(("bot", rb))
                remember_bot_output(rb)
                decision_trace.append("Wrap-up: fallback κλείσιμο (LLM None).")

            map_html = render_emotional_map(mood_value, sleep, water, last_user_text)
            st.session_state.messages.append(("map", map_html))
            decision_trace.append("Wrap-up: συναισθηματικός χάρτης.")

            ex = exercise_suggestion(mood_value, sleep, water, last_user_text)
            st.session_state.messages.append(("exercise", ex))
            decision_trace.append("Wrap-up: άσκηση ημέρας.")

            ex_low = (ex or "").lower()
            if "διάλεξε" in ex_low and "λέξη" in ex_low:
                st.session_state.exercise_followup = True
                decision_trace.append("Wrap-up: άσκηση λέξης → follow-up ενεργό.")

        # ============================================================
        # Session state init + load persisted
        # ============================================================
        if "chat_loaded_for_user" not in st.session_state:
            st.session_state.chat_loaded_for_user = {}

        ukey = _user_key()
        if not st.session_state.chat_loaded_for_user.get(ukey, False):
            persisted = _load_persisted_chat_state()

            st.session_state.messages = persisted.get("messages", st.session_state.get("messages", [])) or []
            st.session_state.dialogue_active = persisted.get("dialogue_active", False)
            st.session_state.dialogue_turns = int(persisted.get("dialogue_turns", 0) or 0)
            st.session_state.wrapup_done = persisted.get("wrapup_done", False)

            st.session_state.conversation_summary = persisted.get("conversation_summary", "") or ""
            st.session_state.open_threads = persisted.get("open_threads", []) or []
            st.session_state.facts_memory = persisted.get("facts_memory", []) or []

            st.session_state.last_bot_outputs = persisted.get("last_bot_outputs", []) or []
            st.session_state.last_checkin = persisted.get("last_checkin", {}) or {}
            st.session_state.exercise_followup = persisted.get("exercise_followup", False)

            st.session_state.support_indicator_history = persisted.get("support_indicator_history", []) or []

            st.session_state.chat_loaded_for_user[ukey] = True

        # Defaults if missing
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "exercise_followup" not in st.session_state:
            st.session_state.exercise_followup = False
        if "last_decision_trace" not in st.session_state:
            st.session_state.last_decision_trace = []
        if "last_checkin" not in st.session_state:
            st.session_state.last_checkin = {}
        if "dialogue_active" not in st.session_state:
            st.session_state.dialogue_active = False
        if "dialogue_turns" not in st.session_state:
            st.session_state.dialogue_turns = 0
        if "wrapup_done" not in st.session_state:
            st.session_state.wrapup_done = False

        if "conversation_summary" not in st.session_state:
            st.session_state.conversation_summary = ""
        if "open_threads" not in st.session_state:
            st.session_state.open_threads = []
        if "facts_memory" not in st.session_state:
            st.session_state.facts_memory = []
        if "last_bot_outputs" not in st.session_state:
            st.session_state.last_bot_outputs = []

        if "support_indicator_history" not in st.session_state:
            st.session_state.support_indicator_history = []

        # ============================================================
        # UI
        # ============================================================
        st.markdown(
            """
            <style>
            .pink-metrics-row{display:flex; gap:14px; flex-wrap:wrap; margin: 8px 0 14px 0;}
            .pink-metric{
                width: 118px; height: 118px; border-radius: 999px;
                background: rgba(255, 105, 180, 0.14);
                border: 1px solid rgba(255, 105, 180, 0.28);
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                box-shadow: 0 6px 20px rgba(255,105,180,0.08);
            }
            .pink-metric .v{font-size: 26px; font-weight: 800; color:#4A3D73; line-height: 1;}
            .pink-metric .t{font-size: 12.5px; font-weight:700; color:#6C5A9E; margin-top:6px;}
            .pink-metric .p{font-size: 12px; color:#6C5A9E; margin-top:2px;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="main-wrapper"><div class="chat-card">', unsafe_allow_html=True)

        st.markdown("#### 😊 Πώς είσαι σήμερα;")

        colA, colB, colC = st.columns(3)
        with colA:
            mood_value = st.slider("Διάθεση (1–10)", 1, 10, 5, 1, key="chat_mood")
        with colB:
            sleep = st.slider("Ύπνος (ώρες, 1–10)", 1, 10, 7, 1, key="chat_sleep")
        with colC:
            water = st.slider("Νερό (ποτήρια, 1–15)", 1, 15, 6, 1, key="chat_water")

        mood_pct = int(round((mood_value / 10.0) * 100))
        sleep_pct = int(round((sleep / 10.0) * 100))
        water_pct = int(round((water / 15.0) * 100))

        st.markdown(
            f"""
            <div class="pink-metrics-row">
              <div class="pink-metric">
                <div class="v">{mood_value}/10</div>
                <div class="t">Διάθεση</div>
                <div class="p">{mood_pct}%</div>
              </div>
              <div class="pink-metric">
                <div class="v">{sleep}h</div>
                <div class="t">Ύπνος</div>
                <div class="p">{sleep_pct}%</div>
              </div>
              <div class="pink-metric">
                <div class="v">{water}</div>
                <div class="t">Νερό</div>
                <div class="p">{water_pct}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_text = st.text_area("📝 Γράψε μου ό,τι θέλεις για τη μέρα σου:", height=120, key="chat_text")

        # ============================================================
        # SEND
        # ============================================================
        if st.button("Αποστολή", key="chat_send"):
            text = (user_text or "").strip()
            if not text:
                st.warning("Γράψε κάτι μικρό πριν πατήσεις αποστολή ")
            else:
                decision_trace: list[str] = []
                decision_trace.append(f"Check-in: mood={mood_value}/10 ({mood_pct}%), sleep={sleep} ({sleep_pct}%), water={water} ({water_pct}%).")

                # 1) Emergency gate 
                if is_emergency(text):
                    decision_trace.append("Emergency gate: ενεργοποίηση ειδικού μηνύματος.")
                    emergency_html = emergency_message()
                    st.session_state.messages.append(("emergency", emergency_html))

                    user_email = st.session_state.get("user_email", "")
                    log_user_data("EMERGENCY", "-", "-", text, email=user_email)

                    st.session_state.last_decision_trace = decision_trace

                    # persist
                    _save_persisted_chat_state({
                        "messages": st.session_state.messages,
                        "dialogue_active": st.session_state.dialogue_active,
                        "dialogue_turns": st.session_state.dialogue_turns,
                        "wrapup_done": st.session_state.wrapup_done,
                        "conversation_summary": st.session_state.conversation_summary,
                        "open_threads": st.session_state.open_threads,
                        "facts_memory": st.session_state.facts_memory,
                        "last_bot_outputs": st.session_state.last_bot_outputs,
                        "last_checkin": st.session_state.last_checkin,
                        "exercise_followup": st.session_state.exercise_followup,
                        "support_indicator_history": st.session_state.support_indicator_history,
                    })
                    st.rerun()

                # 2) Append user message
                st.session_state.messages.append(("user", text))

                # 3) Mode blending
                mode_weights = detect_mode_weights(text, mood_value, sleep, water)
                active_mode = choose_active_mode_from_weights(mode_weights)
                decision_trace.append(f"Mode blending: {_format_mode_blend_text(mode_weights)}.")
                decision_trace.append(f"Active mode: {active_mode or 'NONE'}.")

                # 4) Follow-up 
                if st.session_state.get("exercise_followup", False):
                    decision_trace.append("Exercise follow-up: λέξη.")
                    ctx = st.session_state.get("last_checkin", {})

                    followup = llm_exercise_followup(
                        chosen_word=text,
                        mood_value=ctx.get("mood", mood_value),
                        sleep=ctx.get("sleep", sleep),
                        water=ctx.get("water", water),
                        last_text=ctx.get("text", ""),
                        profile=profile,
                        active_mode=active_mode,
                    )
                    if followup:
                        followup = _safe_trim(followup, 900)
                        st.session_state.messages.append(("bot", followup))
                        remember_bot_output(followup)
                        decision_trace.append("Exercise follow-up: LLM reply.")
                    else:
                        rb = "Σε ακούω. Θες να μου πεις λίγα παραπάνω για το τι σημαίνει αυτή η λέξη για σένα;"
                        st.session_state.messages.append(("bot", rb))
                        remember_bot_output(rb)
                        decision_trace.append("Exercise follow-up: fallback (LLM None).")

                    st.session_state.exercise_followup = False
                    st.session_state.last_decision_trace = decision_trace

                    _save_persisted_chat_state({
                        "messages": st.session_state.messages,
                        "dialogue_active": st.session_state.dialogue_active,
                        "dialogue_turns": st.session_state.dialogue_turns,
                        "wrapup_done": st.session_state.wrapup_done,
                        "conversation_summary": st.session_state.conversation_summary,
                        "open_threads": st.session_state.open_threads,
                        "facts_memory": st.session_state.facts_memory,
                        "last_bot_outputs": st.session_state.last_bot_outputs,
                        "last_checkin": st.session_state.last_checkin,
                        "exercise_followup": st.session_state.exercise_followup,
                        "support_indicator_history": st.session_state.support_indicator_history,
                    })
                    st.rerun()

                # 5) Rule-based opening μόνο στην αρχή νέου διαλόγου IMPORTANT
                if (not st.session_state.dialogue_active) or (st.session_state.dialogue_turns == 0):
                    st.session_state.dialogue_active = True
                    st.session_state.wrapup_done = False
                    st.session_state.dialogue_turns = 0

                    rb_open = personal_reply(mood_value, sleep, water)
                    st.session_state.messages.append(("bot", rb_open))
                    remember_bot_output(rb_open)
                    decision_trace.append("Open: personal_reply (μόνο 1η φορά).")

                    if user_says_feels_ok(text) and looks_a_bit_strained(mood_value, sleep, water):
                        gentle = (
                            "Σημειώνω ότι λες πως είσαι καλά, και αυτό μετράει.\n\n"
                            f"Ταυτόχρονα (διάθεση {mood_value}/10, ύπνος {sleep}, νερό {water}) μοιάζει να υπάρχει λίγη κούραση από κάτω. "
                            "Ας το πάμε ήπια."
                        )
                        st.session_state.messages.append(("bot", gentle))
                        remember_bot_output(gentle)
                        decision_trace.append("Open: gentle mismatch note (ok + strained).")

                # 6) LLM dialogue reply 
                prompt = build_llm_user_prompt(
                    raw_user_text=text,
                    mood_value=mood_value,
                    sleep=sleep,
                    water=water,
                    active_mode=active_mode,
                    mode_weights=mode_weights,
                )

                llm_out = llm_therapeutic_reply(
                    mood=mood_value,
                    sleep=sleep,
                    water=water,
                    user_text=prompt,
                    profile=profile,
                    active_mode=active_mode,
                )

                if llm_out:
                    llm_out = _safe_trim(llm_out, 1400)
                    st.session_state.messages.append(("bot", llm_out))
                    remember_bot_output(llm_out)
                    decision_trace.append("Dialogue: LLM reply (continuity+blending).")

                    update_narrative_memory_llm(
                        profile=profile,
                        active_mode=active_mode,
                        mode_weights=mode_weights,
                        user_text=text,
                        bot_text=llm_out,
                        decision_trace=decision_trace,
                    )
                else:
                    rb = fallback_therapeutic_reply(mood_value, sleep, water, text)
                    st.session_state.messages.append(("bot", rb))
                    remember_bot_output(rb)
                    decision_trace.append("Dialogue: fallback rule-based (LLM None).")

                st.session_state.dialogue_turns += 1

                # 7) Discreet “support” indicator (non-diagnostic)
                score, score_reasons = compute_support_need_score(
                    text=text,
                    mood=mood_value,
                    sleep=sleep,
                    water=water,
                    mode_weights=mode_weights,
                    conversation_summary=st.session_state.get("conversation_summary", ""),
                    open_threads=st.session_state.get("open_threads", []),
                )
                label, suggestion = support_need_label(score)

                if score >= 35:
                    hint = (
                        f"**Δείκτης προτεινόμενης υποστήριξης:** {score}/100 · *{label}*\n\n"
                        f"{suggestion}"
                    )
                    st.session_state.messages.append(("support_hint", hint))
                    decision_trace.append(f"Support indicator: {score}/100 ({label}).")
                    if score_reasons:
                        decision_trace.append("Support reasons: " + ", ".join(score_reasons[:4]) + ("…" if len(score_reasons) > 4 else ""))
                else:
                    decision_trace.append(f"Support indicator: {score}/100 (no display; below threshold).")

                st.session_state.support_indicator_history.append(
                    {"score": score, "label": label, "mood": mood_value, "sleep": sleep, "water": water, "ts": datetime.now().isoformat(timespec="seconds")}
                )
                st.session_state.support_indicator_history = st.session_state.support_indicator_history[-30:]

                # 8) Closure → Wrap-up bundle (μία φορά)
                closing_now = _detect_dialogue_closure(text)
                if closing_now and not st.session_state.wrapup_done:
                    decision_trace.append("Closure detected: run wrap-up bundle.")
                    run_wrapup_bundle(
                        mood_value=mood_value,
                        sleep=sleep,
                        water=water,
                        last_user_text=text,
                        profile=profile,
                        active_mode=active_mode,
                        mode_weights=mode_weights,
                        decision_trace=decision_trace,
                    )
                    st.session_state.wrapup_done = True

                # 9) Logging (CSV)
                user_email = st.session_state.get("user_email", "")
                log_user_data(mood_value, sleep, water, text, email=user_email)
                decision_trace.append("Logging: saved to CSV.")

                # 10) last_checkin
                st.session_state.last_checkin = {
                    "mood": mood_value,
                    "sleep": sleep,
                    "water": water,
                    "text": text,
                    "active_mode": active_mode,
                    "mode_weights": mode_weights,
                }

                st.session_state.last_decision_trace = decision_trace

                # 11) Persist per-user state
                _save_persisted_chat_state({
                    "messages": st.session_state.messages,
                    "dialogue_active": st.session_state.dialogue_active,
                    "dialogue_turns": st.session_state.dialogue_turns,
                    "wrapup_done": st.session_state.wrapup_done,
                    "conversation_summary": st.session_state.conversation_summary,
                    "open_threads": st.session_state.open_threads,
                    "facts_memory": st.session_state.facts_memory,
                    "last_bot_outputs": st.session_state.last_bot_outputs,
                    "last_checkin": st.session_state.last_checkin,
                    "exercise_followup": st.session_state.exercise_followup,
                    "support_indicator_history": st.session_state.support_indicator_history,
                })

                st.rerun()

        # ============================================================
        # RENDER CHAT HISTORY
        # ============================================================
        for sender, content in st.session_state.messages:
            if sender == "user":
                render_message("user", content)
            elif sender == "bot":
                render_message("bot", content)
            elif sender == "exercise":
                render_exercise_card(content)
            elif sender == "map":
                st.markdown(f"<div class='emotional-map-card'>{content}</div>", unsafe_allow_html=True)
            elif sender == "emergency":
                render_emergency_block(content)
            elif sender == "plan":
                render_action_plan_card(content)
            elif sender == "support_hint":
                
                render_message("bot", content)

        # ============================================================
        # Explainable layer 
        # ============================================================
        if st.session_state.get("last_decision_trace"):
            with st.expander("ℹ️ Explainable layer (τι έγινε σε αυτό το turn)"):
                for item in st.session_state.last_decision_trace:
                    st.markdown(f"- {item}")

        st.markdown(
            """
            <p class="footer-disclaimer">
              Το Project Wellness είναι εργαλείο αυτοβοήθειας και ψυχοεκπαιδευτικού χαρακτήρα.
              Δεν αντικαθιστά ψυχολόγο, ψυχίατρο ή υπηρεσίες έκτακτης ανάγκης.
              Αν βρίσκεσαι σε κίνδυνο, κάλεσε το 112 ή τη Γραμμή Παρέμβασης 1018.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div></div>", unsafe_allow_html=True)  # chat-card + main-wrapper


# ============================================================
# ΙΣΤΟΡΙΚΟ TAB 
# ============================================================

with tab_history:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>📁 Ιστορικό Καταγραφών</h1>
          <p class='page-subtitle'>Μια ήρεμη, καθαρή ματιά στις προηγούμενες καταγραφές σου.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    # 1) Εντοπίζουμε το CSV 
    csv_path = os.path.join(os.path.dirname(__file__), "..", "user_data.csv")

    # μικρό debug panel
    with st.expander("🛠️ Debug (αν είναι άδειο)", expanded=False):
        st.write("CSV path:", csv_path)
        st.write("Υπάρχει αρχείο;", os.path.exists(csv_path))
        st.write("Τρέχον email:", st.session_state.get("user_email", ""))

    if not os.path.exists(csv_path):
        st.info("Δεν υπάρχουν ακόμη καταγραφές. Κάνε πρώτα ένα check-in στην καρτέλα «Chat».")
    else:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")

        if df is None or df.empty:
            st.info("Το αρχείο καταγραφών είναι άδειο. Κάνε ένα πρώτο check-in στην καρτέλα «Chat».")
        else:
            # 3) Κανονικοποίηση στηλών 
            colmap = {c.lower().strip(): c for c in df.columns}

            def _col(name: str) -> str | None:
                return colmap.get(name)

            if _col("message") is None:
                if _col("text") is not None:
                    df.rename(columns={_col("text"): "message"}, inplace=True)
                elif _col("user_text") is not None:
                    df.rename(columns={_col("user_text"): "message"}, inplace=True)

            if _col("timestamp") is None:
                if _col("time") is not None:
                    df.rename(columns={_col("time"): "timestamp"}, inplace=True)
                elif _col("date") is not None:
                    df.rename(columns={_col("date"): "timestamp"}, inplace=True)

            if _col("email") is None and _col("user_email") is not None:
                df.rename(columns={_col("user_email"): "email"}, inplace=True)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            current_email = (st.session_state.get("user_email", "") or "").strip().lower()
            df_all = df.copy()

            if "email" in df.columns and current_email:
                df["email"] = df["email"].astype(str).str.strip().str.lower()
                df_user = df[df["email"] == current_email].copy()
            else:
                df_user = df.copy()

            if df_user.empty:
                st.warning(
                    "Δεν βρέθηκαν εγγραφές για αυτόν τον λογαριασμό. "
                    "Αυτό συνήθως σημαίνει ότι το email δεν γράφτηκε σωστά στο CSV ή άλλαξε format."
                )
                show_all = st.checkbox("Δείξε όλες τις εγγραφές (χωρίς φιλτράρισμα)", value=False)
                if show_all:
                    df_user = df_all.copy()

            if not df_user.empty:
                st.markdown("### 🪪 Πρόσφατες καταγραφές")
                st.caption("Κάρτες με διάθεση, ύπνο, νερό και το μήνυμά σου κάθε φορά.")

                if "timestamp" in df_user.columns and not df_user["timestamp"].isna().all():
                    df_user = df_user.sort_values("timestamp", ascending=False).head(60)
                else:
                    df_user = df_user.iloc[::-1].head(60)

                for _, row in df_user.iterrows():
                    ts = row.get("timestamp", "")
                    if pd.notna(ts) and str(ts).strip():
                        try:
                            ts_str = pd.to_datetime(ts).strftime("%d/%m/%Y, %H:%M")
                        except Exception:
                            ts_str = str(ts)
                    else:
                        ts_str = "-"

                    mood = row.get("mood", "")
                    sleep = row.get("sleep", "")
                    water = row.get("water", "")
                    raw_msg = str(row.get("message", "") or "")

                    msg_no_tags = re.sub(r"<[^>]+>", "", raw_msg)
                    msg_clean = html.unescape(msg_no_tags).strip()
                    if len(msg_clean) > 260:
                        msg_clean = msg_clean[:260].rstrip() + "…"
                    short_msg_html = html.escape(msg_clean)

                    try:
                        mood_val = float(mood)
                    except (TypeError, ValueError):
                        mood_val = None

                    if mood_val is None:
                        mood_class = "mood-neutral"
                    elif mood_val <= 4:
                        mood_class = "mood-low"
                    elif mood_val <= 7:
                        mood_class = "mood-mid"
                    else:
                        mood_class = "mood-high"

                    card_html = f"""
                    <div class="history-card">
                      <div class="history-card-header">
                        <span class="history-date">🕒 {ts_str}</span>
                        <span class="history-mood-pill {mood_class}">Διάθεση: {mood}/10</span>
                      </div>
                      <div class="history-meta-row">
                        <span class="history-chip">😴 Ύπνος: {sleep} ώρες</span>
                        <span class="history-chip">💧 Νερό: {water} ποτήρια</span>
                      </div>
                      <div class="history-message">«{short_msg_html}»</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                with st.expander("📌 Στήλες που βρέθηκαν στο CSV", expanded=False):
                    st.write(list(df_all.columns))
                    st.write("Σύνολο εγγραφών:", len(df_all))
                    if "email" in df_all.columns and current_email:
                        st.write("Εγγραφές για το email σου:", int((df_all["email"].astype(str).str.lower().str.strip() == current_email).sum()))

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΣΤΑΤΙΣΤΙΚΑ TAB
# ============================================================

with tab_stats:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>📊 Στατιστικά Ευεξίας</h1>
          <p class='page-subtitle'>Μια ματιά στις τελευταίες καταγραφές σου – διάθεση, ύπνος και νερό.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "user_data.csv")

    if not os.path.exists(csv_path):
        st.info("Δεν βρέθηκε ακόμη αρχείο καταγραφών. Κάνε πρώτα μερικά check-ins στην καρτέλα «Chat».")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        df = pd.read_csv(csv_path)
        if df.empty:
            st.info("Το αρχείο καταγραφών είναι άδειο. Κάνε ένα πρώτο check-in στην καρτέλα «Chat».")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            current_email = st.session_state.get("user_email", "")
            if "email" in df.columns and current_email:
                df = df[df["email"] == current_email]

            if df.empty:
                st.info("Δεν βρέθηκαν καταγραφές για αυτόν τον λογαριασμό. Δοκίμασε ένα νέο check-in.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp")

                df_last = df.tail(60).copy()

                for col in ["mood", "sleep", "water"]:
                    df_last[col] = pd.to_numeric(df_last[col], errors="coerce")

                avg_mood = df_last["mood"].mean()
                avg_sleep = df_last["sleep"].mean()
                avg_water = df_last["water"].mean()

                last_row = df_last.iloc[-1]
                last_mood = float(last_row.get("mood", 0) or 0)
                last_sleep = float(last_row.get("sleep", 0) or 0)
                last_water = float(last_row.get("water", 0) or 0)

                st.markdown("### 🧾 Σύνοψη τελευταίων καταγραφών")
                st.caption("Συνοπτική εικόνα από τις πρόσφατες εγγραφές σου.")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(
                        f"""
                        <div class="stats-card stats-card-mood">
                          <div class="stats-card-label">Διάθεση</div>
                          <div class="stats-card-main">{last_mood:.1f}/10</div>
                          <div class="stats-card-sub">Μ.Ο.: {avg_mood:.1f}/10</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(
                        f"""
                        <div class="stats-card stats-card-sleep">
                          <div class="stats-card-label">Ύπνος</div>
                          <div class="stats-card-main">{last_sleep:.1f} ώρες</div>
                          <div class="stats-card-sub">Μ.Ο.: {avg_sleep:.1f} ώρες</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_c:
                    st.markdown(
                        f"""
                        <div class="stats-card stats-card-water">
                          <div class="stats-card-label">Νερό</div>
                          <div class="stats-card-main">{last_water:.1f} ποτήρια</div>
                          <div class="stats-card-sub">Μ.Ο.: {avg_water:.1f} ποτήρια</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                if "timestamp" not in df_last.columns or df_last["timestamp"].isna().all():
                    st.info("Δεν υπάρχουν έγκυρες ημερομηνίες στις καταγραφές, οπότε δεν μπορώ να δείξω διαγράμματα ακόμη.")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    df_last = df_last.dropna(subset=["timestamp"])
                    dates = df_last["timestamp"]

                    st.markdown("### Διάθεση με τον χρόνο")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(dates, df_last["mood"], marker="o")
                    ax1.set_ylabel("Διάθεση (1–10)")
                    ax1.set_xlabel("Ημερομηνία")
                    ax1.set_ylim(0, 10.5)
                    ax1.grid(True, alpha=0.25)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                    fig1.autofmt_xdate()
                    st.pyplot(fig1)

                    st.markdown("###  Ύπνος με τον χρόνο")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(dates, df_last["sleep"], marker="o")
                    ax2.set_ylabel("Ύπνος (ώρες)")
                    ax2.set_xlabel("Ημερομηνία")
                    ax2.grid(True, alpha=0.25)
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                    fig2.autofmt_xdate()
                    st.pyplot(fig2)

                    st.markdown("###  Νερό με τον χρόνο")
                    fig3, ax3 = plt.subplots()
                    ax3.plot(dates, df_last["water"], marker="o")
                    ax3.set_ylabel("Νερό (ποτήρια)")
                    ax3.set_xlabel("Ημερομηνία")
                    ax3.grid(True, alpha=0.25)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                    fig3.autofmt_xdate()
                    st.pyplot(fig3)

                    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΑΣΚΗΣΕΙΣ TAB
# ============================================================

with tab_ex:
    if "support1" not in st.session_state:
        st.session_state.support1 = ""
    if "support2" not in st.session_state:
        st.session_state.support2 = ""
    if "support3" not in st.session_state:
        st.session_state.support3 = ""

    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>🧘 Μικρή Βιβλιοθήκη Ασκήσεων</h1>
          <p class='page-subtitle'>
            Μικρές ασκήσεις φροντίδας και ήρεμα διαδραστικά κουμπάκια,
            προσαρμοσμένα σε εσένα.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 1. Άσκηση αναπνοής 4–2–6
    # --------------------------------------------------------
    st.markdown("### 1. Άσκηση αναπνοής 4–2–6")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(
            """
            <div class="breathing-container">
              <div class="breathing-circle"></div>
              <p class="breathing-hint">4″ εισπνοή • 2″ κράτημα • 6″ εκπνοή</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ξεκίνα 5 κύκλους αναπνοής", key="breath_start"):
            st.info(
                "Πάρε 4″ εισπνοή από τη μύτη, κράτησε 2″ και εκπνοή από το στόμα για 6″. "
                "Επανάλαβε 5 φορές, με όσο πιο ήρεμο ρυθμό μπορείς."
            )

    with col_right:
        st.write(
            "Χρησιμεύει όταν υπάρχει έντονο άγχος ή σωματική ένταση.\n\n"
            "Μπορείς να το δεις σαν ένα μικρό reset του νευρικού συστήματος."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # 2. Αποφόρτιση σκέψεων
    # --------------------------------------------------------
    st.markdown("### 2. Μικρή άσκηση αποφόρτισης σκέψεων")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        heavy_thought = st.text_area(
            "📝 Συμπλήρωσε: «Αυτό που με βαραίνει περισσότερο είναι…»",
            height=80,
            key="heavy_thought",
        )

    with col_b:
        if st.button("Αποφόρτιση", key="discharge_btn", use_container_width=True):
            if (heavy_thought or "").strip():
                st.success(
                    "Το έγραψες — άρα δεν το κρατάς μόνο μέσα σου. "
                    "Δεν χρειάζεται να λυθεί τώρα."
                )
            else:
                st.warning("Γράψε πρώτα μία μικρή πρόταση 🙂")

    st.markdown("---")

    # --------------------------------------------------------
    # 3. Θλίψη / μοναξιά
    # --------------------------------------------------------
    st.markdown("### 3. Άσκηση ηρεμίας για θλίψη / μοναξιά")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            "Όταν νιώθεις βαρύς/ιά ή μόνος/η:\n\n"
            "1. Βάλε το χέρι στο στήθος.\n"
            "2. Πάρε μία αργή ανάσα.\n"
            "3. Πες από μέσα σου:\n\n"
            "_«Είναι εντάξει να νιώθω έτσι. Δεν είμαι μόνος/η.»_"
        )

    with col2:
        if st.button("Θέλω μια μικρή φράση στήριξης", key="lonely_btn"):
            profile = load_profile()
            chat_tail = get_chat_tail_from_state()
            active_mode = st.session_state.get("active_mode", "NONE")

            key = "support_phrase"
            history = st.session_state.microcoach_history.get(key, [])

            msg = micro_prompt_with_fallback(
                exercise_key=key,
                profile=profile,
                chat_tail=chat_tail,
                active_mode=active_mode,
                avoid_texts=history,
            )

            history.append(msg)
            st.session_state.microcoach_history[key] = history[-8:]
            st.info(msg)

    st.markdown("---")

    # --------------------------------------------------------
    # 4. Νερό
    # --------------------------------------------------------
    st.markdown("### 4. Μικρή άσκηση φροντίδας σώματος")
    col_w1, col_w2 = st.columns([2, 1])

    with col_w1:
        st.write(
            "Ένα ποτήρι νερό είναι μικρή αλλά ουσιαστική πράξη φροντίδας.\n\n"
            "Δοκίμασε να το συνδυάσεις με 3 αργές αναπνοές."
        )

    with col_w2:
        if st.button("💧 Μικρή υπενθύμιση νερού", key="water_nudge"):
            profile = load_profile()
            chat_tail = get_chat_tail_from_state()
            active_mode = st.session_state.get("active_mode", "NONE")

            key = "water_nudge"
            history = st.session_state.microcoach_history.get(key, [])

            msg = micro_prompt_with_fallback(
                exercise_key=key,
                profile=profile,
                chat_tail=chat_tail,
                active_mode=active_mode,
                avoid_texts=history,
            )

            history.append(msg)
            st.session_state.microcoach_history[key] = history[-8:]
            st.success(msg)

    st.markdown("---")

    # --------------------------------------------------------
    # 5. 3 μικρά στηρίγματα
    # --------------------------------------------------------
    st.markdown("### 5. Άσκηση «3 μικρά στηρίγματα της ημέρας»")
    st.write(
        "Σημείωσε τρία μικρά πράγματα που σε στήριξαν σήμερα "
        "(όσο μικρά κι αν φαίνονται)."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        support1 = st.text_input("Στήριγμα 1", key="support1")
    with col_s2:
        support2 = st.text_input("Στήριγμα 2", key="support2")
    with col_s3:
        support3 = st.text_input("Στήριγμα 3", key="support3")

    if support1 or support2 or support3:
        st.caption("Αυτά τα μικρά στηρίγματα μετράνε περισσότερο απ’ όσο φαίνεται.")

    st.markdown("---")

    # --------------------------------------------------------
    # 6. Body scan
    # --------------------------------------------------------
    st.markdown("### 6. Μικρή σωματική σάρωση (body scan 1′)")
    col_bs1, col_bs2 = st.columns([2, 1])

    with col_bs1:
        st.write(
            "Πάρε 1 λεπτό για να περάσεις προσοχή στο σώμα:\n\n"
            "• μέτωπο & μάτια\n"
            "• ώμοι & αυχένας\n"
            "• στήθος & κοιλιά\n"
            "• παλάμες & πέλματα\n\n"
            "Μόνο παρατήρηση — όχι διόρθωση."
        )

    with col_bs2:
        if st.button("Ξεκίνα 1′ body scan", key="bodyscan_btn"):
            st.info(
                "Κλείσε τα μάτια και πέρασε ήρεμα την προσοχή σου στο σώμα, "
                "χωρίς να αλλάξεις τίποτα."
            )

    st.markdown("---")

    # --------------------------------------------------------
    # 7. Μικρό παιχνίδι – φροντιστική απάντηση
    # --------------------------------------------------------
    st.markdown("### 🧩 Μικρό παιχνίδι: «Απάντηση φροντίδας σε δύσκολη σκέψη»")

    difficult_thoughts = [
        "Αποτυγχάνω σε όλα.",
        "Δεν αξίζω την αγάπη των άλλων.",
        "Οι άλλοι θα με θεωρήσουν αδύναμο/η.",
        "Δεν θα αλλάξει ποτέ τίποτα.",
    ]

    chosen_label = st.selectbox(
        "Διάλεξε μία σκέψη:",
        difficult_thoughts,
        key="mh_game_select",
    )

    if st.button("Δείξε μου μια απάντηση φροντίδας", key="mh_game_button"):
        profile = load_profile()
        chat_tail = get_chat_tail_from_state() + [f"Δύσκολη σκέψη: {chosen_label}"]
        active_mode = st.session_state.get("active_mode", "NONE")

        key = "care_reply_game"
        history = st.session_state.microcoach_history.get(key, [])

        msg = micro_prompt_with_fallback(
            exercise_key=key,
            profile=profile,
            chat_tail=chat_tail,
            active_mode=active_mode,
            avoid_texts=history,
        )

        history.append(msg)
        st.session_state.microcoach_history[key] = history[-8:]
        st.info(msg)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# THOUGHT REFRAME STUDIO TAB
# ============================================================

with tab_reframe:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>🧩 Thought Reframe Studio</h1>
          <p class='page-subtitle'>
            Ένας μικρός χώρος για να μετασχηματίζεις δύσκολες σκέψεις σε πιο τρυφερές, ρεαλιστικές.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    st.markdown(
        """
        Σημείωσε μια σκέψη που σε ζορίζει και επίλεξε τον τρόπο σκέψης που της ταιριάζει περισσότερο.
        Ο στόχος δεν είναι να ακυρώσουμε το συναίσθημα, αλλά να βρούμε μια φράση πιο επιεική με εσένα.
        """,
        unsafe_allow_html=True,
    )

    pattern = st.selectbox(
        "Τι μοιάζει περισσότερο με αυτή τη σκέψη;",
        [
            "Καταστροφολογία (τα βλέπω όλα χάλια)",
            "Όλα ή τίποτα (είτε τέλειο είτε αποτυχία)",
            "Διάβασμα σκέψης (υποθέτω τι σκέφτονται οι άλλοι)",
            "Υπεργενίκευση (μια εμπειρία = πάντα έτσι)",
            "Άλλο / δεν είμαι σίγουρος",
        ],
    )

    pattern_explanations = {
        "Καταστροφολογία (τα βλέπω όλα χάλια)": (
            "Ο νους πηγαίνει στο χειρότερο σενάριο, σαν να είναι βέβαιο ότι όλα θα πάνε στραβά."
        ),
        "Όλα ή τίποτα (είτε τέλειο είτε αποτυχία)": (
            "Άσπρο–μαύρο: ή τέλειο ή αποτυχία, χωρίς τις ενδιάμεσες αποχρώσεις."
        ),
        "Διάβασμα σκέψης (υποθέτω τι σκέφτονται οι άλλοι)": (
            "Υποθέτεις αρνητική κρίση των άλλων χωρίς πραγματικές αποδείξεις."
        ),
        "Υπεργενίκευση (μια εμπειρία = πάντα έτσι)": (
            "Μια εμπειρία γίνεται «απόδειξη» ότι πάντα έτσι θα είναι."
        ),
        "Άλλο / δεν είμαι σίγουρος": (
            "Δεν ταιριάζει καθαρά σε ένα μοτίβο. Είναι ΟΚ."
        ),
    }

    with st.expander("🔍 Μικρή εξήγηση για αυτό το μοτίβο"):
        st.write(pattern_explanations.get(pattern, ""))

    intensity = st.slider("Πόσο σε βαραίνει αυτή η σκέψη αυτή τη στιγμή;", min_value=1, max_value=10, value=7)

    thought_input = st.text_area(
        "✏️ Δύσκολη σκέψη (όπως ακριβώς σου έρχεται στο μυαλό):",
        height=90,
        key="reframe_thought",
    )

    if st.button("Μετασχηματισμός σκέψης ✨", key="reframe_button"):
        if not (thought_input or "").strip():
            st.warning("Γράψε πρώτα τη σκέψη που σε δυσκολεύει 📝")
        else:
            reframed = generate_reframe(thought_input, pattern)
            full_reframe = (
                f"Βλέπω ότι αυτή η σκέψη σε βαραίνει περίπου **{intensity}/10** αυτή τη στιγμή.\n\n"
                + reframed
            )

            st.session_state.reframes_history.append(
                {
                    "pattern": pattern,
                    "thought": thought_input,
                    "reframe": full_reframe,
                    "intensity": intensity,
                }
            )

            st.markdown(
                """
                <div class="reframe-card">
                  <div class="reframe-original-title">Αρχική σκέψη</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='reframe-original-body'>{html.escape(thought_input)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='reframe-new-title'>Πιο φροντιστική εκδοχή</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='reframe-new-body'>{full_reframe}</div></div>",
                unsafe_allow_html=True,
            )

    if st.session_state.reframes_history:
        st.markdown("---")
        st.markdown("### 📚 Μικρό ιστορικό μετασχηματισμών")

        for item in reversed(st.session_state.reframes_history[-5:]):
            orig = html.escape(item["thought"])
            pat = item["pattern"]
            inten = item["intensity"]
            ref = item["reframe"]

            st.markdown(
                f"""
                <div class="history-card">
                  <div class="history-card-header">
                    <span class="history-date">Μοτίβο: {pat}</span>
                    <span class="history-mood-pill mood-mid">Ένταση: {inten}/10</span>
                  </div>
                  <div class="history-message"><strong>Σκέψη:</strong> «{orig}»</div>
                  <div class="history-message" style="margin-top:0.3rem;"><strong>Νέα εκδοχή:</strong> {ref}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΠΡΟΦΙΛ TAB
# ============================================================

with tab_profile:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>🌿 Προφίλ Φροντίδας</h1>
          <p class='page-subtitle'>Μερικές πληροφορίες που βοηθούν να καταλαβαίνω καλύτερα το πλαίσιο σου.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = load_profile()

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)
    st.markdown("### 📝 Βασικά στοιχεία")

    name = st.text_input("Όνομα ή ψευδώνυμο", value=profile.get("name", ""))

    stored_age = profile.get("age", 0)
    try:
        stored_age = int(stored_age)
    except (TypeError, ValueError):
        stored_age = 0

    age = st.number_input("Ηλικία (προαιρετικό)", min_value=0, max_value=120, value=stored_age, step=1)

    computed_age_range = infer_age_range(age)
    if computed_age_range:
        age_range = computed_age_range
        st.caption(f"Ηλικιακή ομάδα (υπολογισμένη): **{age_range}**")
    else:
        age_range = profile.get("age_range", "")
        if age_range:
            st.caption(f"Αποθηκευμένη ηλικιακή ομάδα: **{age_range}**")
        else:
            st.caption("Αν συμπληρώσεις ηλικία, η εφαρμογή υπολογίζει ηλικιακή ομάδα (18–24, 25–34, 35–44, 45+).")

    context = st.text_input(
        "Πλαίσιο ζωής / ρόλος (π.χ. φοιτητής, εργαζόμενος)",
        value=profile.get("context", ""),
    )

    st.markdown("---")
    st.markdown("### 🎯 Στόχοι")
    main_goals = st.text_area(
        "Ποιοι είναι οι βασικοί σου στόχοι ευεξίας αυτή την περίοδο;",
        value=profile.get("main_goals", ""),
        height=100,
    )

    st.markdown("---")
    st.markdown("### Δυσκολίες & ανάγκες")
    main_struggles = st.text_area(
        "Τι σε δυσκολεύει περισσότερο τον τελευταίο καιρό;",
        value=profile.get("main_struggles", ""),
        height=110,
    )
    helpful_things = st.text_area(
        "Τι σε βοηθά συνήθως (ακόμη κι αν είναι μικρό);",
        value=profile.get("helpful_things", ""),
        height=90,
    )

    st.markdown("---")
    st.markdown("### 💜 Συναισθηματικές προτιμήσεις")
    preferred_tone = st.text_area(
        "Πώς θα ήθελες να σου μιλάει η εφαρμογή;",
        value=profile.get("preferred_tone", ""),
        height=80,
    )
    triggers = st.text_area(
        "Υπάρχουν θέματα ή λέξεις που θα προτιμούσες να αποφεύγονται;",
        value=profile.get("triggers", ""),
        height=80,
    )
    soothing_things = st.text_area(
        "Ποια πράγματα σε ηρεμούν συνήθως όταν ζορίζεσαι;",
        value=profile.get("soothing_things", ""),
        height=80,
    )

    st.markdown("---")

    if st.button("💾 Αποθήκευση προφίλ"):
        new_profile = {
            "name": name,
            "age": int(age) if age else 0,
            "age_range": age_range,
            "context": context,
            "main_goals": main_goals,
            "main_struggles": main_struggles,
            "helpful_things": helpful_things,
            "preferred_tone": preferred_tone,
            "triggers": triggers,
            "soothing_things": soothing_things,
        }
        save_profile(new_profile)
        st.success("Το προφίλ σου αποθηκεύτηκε επιτυχώς 🙂")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΣΥΝΔΕΣΗ ΑΠΟ ΚΙΝΗΤΟ TAB
# ============================================================

with tab_mobile:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>📱 Σύνδεση από κινητό</h1>
          <p class='page-subtitle'>
            Άνοιξε την εφαρμογή του Project Wellness και στο κινητό σου, όσο είσαι στο ίδιο Wi-Fi με τον υπολογιστή.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    local_url = get_local_url(port=8501)
    qr_img = generate_qr_image(local_url)

    st.markdown("<div class='qr-page-wrapper'>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown(
            """
            <div class="qr-main-card">
              <div class="qr-badge">Local only</div>
              <div class="qr-title">Σκάναρε το QR από το κινητό σου</div>
              <p class="qr-subtitle">
                Άνοιξε την κάμερα στο κινητό και στόχευσε το QR.
                Αν δεν ανοίξει αυτόματα, χρησιμοποίησε app ανάγνωσης QR.
              </p>
            """,
            unsafe_allow_html=True,
        )
        st.image(qr_img, width=170)
        st.markdown(
            f"""
              <div class="qr-url-chip">{local_url}</div>
              <p class="qr-note">
                Αν δεν δουλέψει το QR, γράψε χειροκίνητα αυτή τη διεύθυνση στον browser του κινητού.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="qr-side-card">
              <h3 class="qr-side-title">Πώς λειτουργεί;</h3>
              <ol class="qr-steps">
                <li>Υπολογιστής και κινητό στο ίδιο Wi-Fi.</li>
                <li>Άφησε ανοιχτή την εφαρμογή στον υπολογιστή.</li>
                <li>Σκάναρε το QR ή γράψε τη διεύθυνση χειροκίνητα.</li>
              </ol>
              <p class="qr-side-note">
                Η εφαρμογή τρέχει τοπικά. Τα δεδομένα δεν ανεβαίνουν σε εξωτερικό server.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΣΧΕΤΙΚΑ & ΑΣΦΑΛΕΙΑ TAB
# ============================================================

with tab_info:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>ℹ️ Σχετικά &amp; Ασφάλεια</h1>
          <p class='page-subtitle'>Πώς λειτουργεί το Project Wellness και πώς σε προστατεύει.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
          <h3 class="info-title">🔐 Μέτρα Ασφαλείας</h3>
          <p>
            Το Project Wellness αναγνωρίζει λέξεις/φράσεις που μπορεί να δείχνουν έντονη κρίση ή άμεσο κίνδυνο.
          </p>
          <p>Σε αυτές τις περιπτώσεις:</p>
          <ul>
            <li>σταματά η κανονική ροή</li>
            <li>εμφανίζεται ειδικό μήνυμα με οδηγίες και πηγές βοήθειας</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
          <h3 class="info-title">ℹ️ Πώς λειτουργεί</h3>
          <p>
            Συνδυάζει απλούς κανόνες (διάθεση/ύπνος/νερό + λέξεις-κλειδιά) με υποστηρικτική γλώσσα.
          </p>
          <p>
            Δεν κάνει διάγνωση, δεν δίνει ιατρικές οδηγίες και δεν αντικαθιστά επαγγελματία ψυχικής υγείας.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
          <h3 class="info-title">☎️ Γραμμές Βοήθειας (Ελλάδα)</h3>
          <ul>
            <li><strong>112</strong> – Άμεση ανάγκη</li>
            <li><strong>1018</strong> – Γραμμή Παρέμβασης για την Αυτοκτονία (24/7)</li>
            <li><strong>10306</strong> – Γραμμή Ψυχοκοινωνικής Υποστήριξης</li>
            <li><strong>1056</strong> – Για ανηλίκους (Το Χαμόγελο του Παιδιού)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
          <h3 class="info-title"> Σημαντικό</h3>
          <p>
            Το Project Wellness είναι εργαλείο αυτοβοήθειας και ψυχοεκπαίδευσης.
            Δεν αντικαθιστά ψυχολόγο/ψυχίατρο/υπηρεσίες έκτακτης ανάγκης.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)