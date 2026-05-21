import os
import re
import html
import socket
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
import streamlit.components.v1 as components

import streamlit as st
import pandas as pd
import qrcode
from data_logger import log_user_data
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
import json
import hashlib
from datetime import datetime

#============================================================
# ΠΡΟΣΩΠΙΚΟ ΠΡΟΦΙΛ ΧΡΗΣΤΗ 
# ===========================================================
def _get_profile_path():
    # Βρίσκουμε ποιος χρήστης είναι συνδεδεμένος
    email = (st.session_state.get("user_email") or "").strip().lower()
    if not email:
        email = (st.session_state.get("user_name") or "anon").strip().lower()
    key = email or "anon"
    
    # Φτιάχνουμε ένα μοναδικό, κρυφό όνομα αρχείου (π.χ. profile_a7b9c.json)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, f"profile_{h}.json")

def load_profile():
    # Διαβάζει το προφίλ από το προσωπικό αρχείο του χρήστη (αν υπάρχει)
    path = _get_profile_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}

def save_profile(new_profile):
    # Αποθηκεύει το προφίλ μόνιμα στο προσωπικό αρχείο
    path = _get_profile_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_profile, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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

            profile_snip_parts = []
            for k in ["context", "main_goals", "main_struggles", "helpful_things", "preferred_tone", "triggers", "soothing_things"]:
                v = (profile.get(k) or "").strip()
                if v:
                    profile_snip_parts.append(f"- {k}: {v}")
            profile_snip = "\n".join(profile_snip_parts).strip()

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

        col_title, col_btn = st.columns([3, 1])
        with col_title:
            st.markdown("#### 😊 Πώς είσαι σήμερα;")
        with col_btn:
            if st.button("🔄 Νέα Συνομιλία", use_container_width=True):
                # Καθαρισμός προσωρινής μνήμης
                st.session_state.messages = []
                st.session_state.dialogue_active = False
                st.session_state.dialogue_turns = 0
                st.session_state.wrapup_done = False
                st.session_state.conversation_summary = ""
                st.session_state.open_threads = []
                st.session_state.facts_memory = []
                st.session_state.last_bot_outputs = []
                st.session_state.last_checkin = {}
                st.session_state.exercise_followup = False
                st.session_state.last_decision_trace = []

                # Καθαρισμός αποθηκευμένης μνήμης χρήστη
                _save_persisted_chat_state({
                    "messages": [],
                    "dialogue_active": False,
                    "dialogue_turns": 0,
                    "wrapup_done": False,
                    "conversation_summary": "",
                    "open_threads": [],
                    "facts_memory": [],
                    "last_bot_outputs": [],
                    "last_checkin": {},
                    "exercise_followup": False,
                    "support_indicator_history": st.session_state.get("support_indicator_history", [])
                })
                
                st.rerun()

        # ΜΕΤΑΦΟΡΑ SLIDERS ΜΕΣΑ ΣΕ EXPANDER
        with st.expander("📊 Σημερινό Check-in (Διάθεση, Ύπνος, Νερό)", expanded=True):
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

        # RENDER CHAT HISTORY ΕΔΩ ΠΑΝΩ! 
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

        if st.session_state.get("last_decision_trace"):
            with st.expander("ℹ️ Explainable layer (τι έγινε σε αυτό το turn)"):
                for item in st.session_state.last_decision_trace:
                    st.markdown(f"- {item}")


        # ST.CHAT_INPUT ΜΟΝΙΜΑ ΣΤΟ ΚΑΤΩ ΜΕΡΟΣ
        user_text = st.chat_input("📝 Γράψε μου ό,τι θέλεις για τη μέρα σου:")

        # ============================================================
        # SEND
        # ============================================================
        if user_text:
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

                # 5) Rule-based opening 
                if (not st.session_state.dialogue_active) or (st.session_state.dialogue_turns == 0):
                    st.session_state.dialogue_active = True
                    st.session_state.wrapup_done = False
                    st.session_state.dialogue_turns = 0

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

                # 7) Discreet “support” indicator 
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

                # 8) Closure → Wrap-up bundle 
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

                # 9) Logging
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

        st.markdown("</div></div>", unsafe_allow_html=True) 


# ============================================================
# ΙΣΤΟΡΙΚΟ TAB 
# ============================================================

# ============================================================
# ΙΣΤΟΡΙΚΟ TAB (Interactive Glass Timeline)
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

    # 1) Εντοπίζουμε το CSV 
    csv_path = os.path.join(os.path.dirname(__file__), "..", "user_data.csv")

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
            
            if "email" in df.columns and current_email:
                df["email"] = df["email"].astype(str).str.strip().str.lower()
                df_user = df[df["email"] == current_email].copy()
            else:
                df_user = df.copy()

            if df_user.empty:
                st.warning("Δεν βρέθηκαν εγγραφές για αυτόν τον λογαριασμό.")
            else:
                st.markdown("### 🪪 Το Χρονολόγιό σου")
                
                # Ταξινόμηση ώστε τα πιο πρόσφατα να είναι πάνω
                if "timestamp" in df_user.columns and not df_user["timestamp"].isna().all():
                    df_user = df_user.sort_values("timestamp", ascending=False).head(60)
                else:
                    df_user = df_user.iloc[::-1].head(60)

                # Ετοιμάζουμε τα δεδομένα για να τα περάσουμε στη Javascript
                timeline_data = []
                for _, row in df_user.iterrows():
                    ts = row.get("timestamp", "")
                    if pd.notna(ts) and str(ts).strip():
                        try:
                            # Μορφοποίηση ημερομηνίας πιο κομψά (π.χ. 15 May 2026 • 20:30)
                            ts_str = pd.to_datetime(ts).strftime("%d %b %Y • %H:%M")
                        except Exception:
                            ts_str = str(ts)
                    else:
                        ts_str = "-"
                        
                    mood = row.get("mood", 5)
                    sleep = row.get("sleep", 0)
                    water = row.get("water", 0)
                    raw_msg = str(row.get("message", "") or "")
                    
                    # Καθαρισμός HTML tags για ασφάλεια
                    msg_no_tags = re.sub(r"<[^>]+>", "", raw_msg)
                    msg_clean = html.unescape(msg_no_tags).strip()
                    
                    try:
                        mood_val = float(mood)
                    except:
                        mood_val = 5.0
                        
                    # Καθορισμός χρωμάτων και emojis βάσει της διάθεσης
                    if mood_val <= 4:
                        mood_emoji = "🌧️"
                        mood_color = "#ff8fa3" # Κόκκινο/Ροζ
                    elif mood_val <= 7:
                        mood_emoji = "⛅"
                        mood_color = "#ffd08a" # Πορτοκαλί/Κίτρινο
                    else:
                        mood_emoji = "☀️"
                        mood_color = "#baf7c3" # Απαλό Πράσινο
                        
                    timeline_data.append({
                        "date": ts_str,
                        "mood": mood,
                        "sleep": sleep,
                        "water": water,
                        "message": msg_clean,
                        "emoji": mood_emoji,
                        "color": mood_color
                    })
                    
                # Μετατροπή της λίστας σε JSON string για να το διαβάσει η JS
                timeline_json = json.dumps(timeline_data, ensure_ascii=False)
                
                # Το υπερ-ρεαλιστικό Timeline (HTML / CSS / JavaScript)
                components.html(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap" rel="stylesheet">
                    <style>
                        body {{
                            margin: 0; padding: 20px 10px; 
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            background: transparent;
                        }}
                        /* Η κεντρική γραμμή του Timeline */
                        .timeline {{
                            position: relative;
                            max-width: 850px;
                            margin: 0 auto;
                            padding-left: 50px; /* Χώρος για τη γραμμή και τα εικονίδια */
                        }}
                        .timeline::before {{
                            content: '';
                            position: absolute;
                            left: 19px;
                            top: 10px;
                            bottom: 0;
                            width: 3px;
                            background: rgba(212, 190, 250, 0.4); /* Λιλά γραμμή */
                            border-radius: 4px;
                        }}
                        /* Κάθε κάρτα/καταγραφή */
                        .item {{
                            position: relative;
                            margin-bottom: 35px;
                            opacity: 0; /* Αρχικά κρυμμένο για το animation */
                            transform: translateY(40px);
                            animation: slideUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
                        }}
                        @keyframes slideUp {{
                            to {{ opacity: 1; transform: translateY(0); }}
                        }}
                        /* Το κυκλικό εικονίδιο πάνω στη γραμμή */
                        .icon {{
                            position: absolute;
                            left: -53px;
                            top: 0;
                            width: 44px;
                            height: 44px;
                            border-radius: 50%;
                            background: rgba(255, 255, 255, 0.8);
                            backdrop-filter: blur(10px);
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            font-size: 20px;
                            box-shadow: 0 6px 15px rgba(108, 90, 158, 0.12);
                            z-index: 1;
                            border: 3px solid; /* Το χρώμα μπαίνει δυναμικά από τη JS */
                            transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
                        }}
                        .item:hover .icon {{
                            transform: scale(1.2) rotate(-10deg);
                            box-shadow: 0 8px 20px rgba(108, 90, 158, 0.25);
                        }}
                        /* Η γυάλινη κάρτα (Glassmorphism) */
                        .card {{
                            background: rgba(255, 255, 255, 0.45);
                            backdrop-filter: blur(24px);
                            -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.7);
                            border-radius: 20px;
                            padding: 22px;
                            box-shadow: 0 12px 35px rgba(108, 90, 158, 0.05);
                            transition: all 0.3s ease;
                        }}
                        .item:hover .card {{
                            transform: translateY(-4px);
                            box-shadow: 0 18px 45px rgba(108, 90, 158, 0.12);
                            background: rgba(255, 255, 255, 0.6);
                        }}
                        .header {{
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 15px;
                            border-bottom: 1px solid rgba(183, 157, 242, 0.25);
                            padding-bottom: 10px;
                        }}
                        .date {{
                            font-weight: 600;
                            font-size: 14.5px;
                            color: #73658a;
                        }}
                        .metrics {{
                            display: flex;
                            gap: 10px;
                            flex-wrap: wrap;
                        }}
                        .badge {{
                            padding: 5px 12px;
                            border-radius: 20px;
                            font-size: 12.5px;
                            font-weight: 600;
                            color: #4A3D73;
                            background: rgba(255, 255, 255, 0.6);
                            border: 1px solid rgba(255, 255, 255, 0.9);
                            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
                        }}
                        .message {{
                            font-size: 15.5px;
                            line-height: 1.6;
                            color: #3b304c;
                            font-style: italic;
                            word-wrap: break-word;
                        }}
                    </style>
                </head>
                <body>
                    <div class="timeline" id="timelineContainer"></div>
                    
                    <script>
                        // Παίρνουμε τα δεδομένα από την Python
                        const data = {timeline_json};
                        const container = document.getElementById('timelineContainer');
                        
                        data.forEach((entry, index) => {{
                            // Staggered Animation: Κάθε κάρτα εμφανίζεται με διαφορά 0.15s
                            const delay = index * 0.15;
                            
                            const item = document.createElement('div');
                            item.className = 'item';
                            item.style.animationDelay = `${{delay}}s`;
                            
                            item.innerHTML = `
                                <div class="icon" style="border-color: ${{entry.color}};">
                                    ${{entry.emoji}}
                                </div>
                                <div class="card">
                                    <div class="header">
                                        <div class="date">📅 ${{entry.date}}</div>
                                        <div class="metrics">
                                            <span class="badge">🧠 Διάθεση: ${{entry.mood}}/10</span>
                                            <span class="badge">😴 Ύπνος: ${{entry.sleep}}h</span>
                                            <span class="badge">💧 Νερό: ${{entry.water}}</span>
                                        </div>
                                    </div>
                                    <div class="message">«${{entry.message}}»</div>
                                </div>
                            `;
                            container.appendChild(item);
                        }});
                    </script>
                </body>
                </html>
                """, height=750, scrolling=True)




# ============================================================
# ΣΤΑΤΙΣΤΙΚΑ TAB (Interactive Glass Charts με Chart.js)
# ============================================================

with tab_stats:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>📊 Στατιστικά Ευεξίας</h1>
          <p class='page-subtitle'>Μια ζωντανή, οπτική απεικόνιση των πρόσφατων check-ins σου.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_path = os.path.join(os.path.dirname(__file__), "..", "user_data.csv")

    if not os.path.exists(csv_path):
        st.info("Δεν βρέθηκε ακόμη αρχείο καταγραφών. Κάνε πρώτα μερικά check-ins στην καρτέλα «Chat».")
    else:
        df = pd.read_csv(csv_path)
        if df.empty:
            st.info("Το αρχείο καταγραφών είναι άδειο. Κάνε ένα πρώτο check-in στην καρτέλα «Chat».")
        else:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            current_email = st.session_state.get("user_email", "")
            if "email" in df.columns and current_email:
                df = df[df["email"].astype(str).str.lower().str.strip() == current_email.strip().lower()]

            if df.empty:
                st.info("Δεν βρέθηκαν καταγραφές για αυτόν τον λογαριασμό. Δοκίμασε ένα νέο check-in.")
            else:
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp")

                df_last = df.tail(15).copy() # Παίρνουμε τα τελευταία 15 για πιο καθαρά διαγράμματα

                for col in ["mood", "sleep", "water"]:
                    df_last[col] = pd.to_numeric(df_last[col], errors="coerce").fillna(0)

                avg_mood = df_last["mood"].mean()
                avg_sleep = df_last["sleep"].mean()
                avg_water = df_last["water"].mean()

                last_row = df_last.iloc[-1]
                last_mood = float(last_row.get("mood", 0) or 0)
                last_sleep = float(last_row.get("sleep", 0) or 0)
                last_water = float(last_row.get("water", 0) or 0)

                # --- 1. ΚΑΡΤΕΣ ΣΥΝΟΨΗΣ ΣΤΗΝ ΚΟΡΥΦΗ ---
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(
                        f"""
                        <div class="stats-card stats-card-mood" style="animation: cardFadeIn 0.4s ease-out;">
                          <div class="stats-card-label">Διάθεση</div>
                          <div class="stats-card-main">{last_mood:.1f}/10</div>
                          <div class="stats-card-sub">Μ.Ο. περιόδου: {avg_mood:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with col_b:
                    st.markdown(
                        f"""
                        <div class="stats-card stats-card-sleep" style="animation: cardFadeIn 0.6s ease-out;">
                          <div class="stats-card-label">Ύπνος</div>
                          <div class="stats-card-main">{last_sleep:.1f}h</div>
                          <div class="stats-card-sub">Μ.Ο. περιόδου: {avg_sleep:.1f}h</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with col_c:
                    st.markdown(
                        f"""
                        <div class="stats-card stats-card-water" style="animation: cardFadeIn 0.8s ease-out;">
                          <div class="stats-card-label">Νερό</div>
                          <div class="stats-card-main">{last_water:.0f} ποτ.</div>
                          <div class="stats-card-sub">Μ.Ο. περιόδου: {avg_water:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )

                # --- 2. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ΓΙΑ ΤΗ JAVASCRIPT ---
                if "timestamp" not in df_last.columns or df_last["timestamp"].isna().all():
                    st.info("Δεν υπάρχουν έγκυρες ημερομηνίες για να φτιάξω τα διαγράμματα.")
                else:
                    df_last = df_last.dropna(subset=["timestamp"])
                    
                    # Μετατροπή ημερομηνιών σε μικρή μορφή (π.χ. "15/05")
                    dates_list = df_last["timestamp"].dt.strftime("%d/%m").tolist()
                    mood_list = df_last["mood"].tolist()
                    sleep_list = df_last["sleep"].tolist()
                    water_list = df_last["water"].tolist()

                    dates_json = json.dumps(dates_list)
                    mood_json = json.dumps(mood_list)
                    sleep_json = json.dumps(sleep_list)
                    water_json = json.dumps(water_list)

                    # --- 3. JAVASCRIPT & CHART.JS ΟΠΤΙΚΟΠΟΙΗΣΗ ---
                    components.html(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                        <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600&display=swap" rel="stylesheet">
                        <style>
                            body {{
                                margin: 0; padding: 10px;
                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                background: transparent;
                            }}
                            .chart-container {{
                                background: rgba(255, 255, 255, 0.45);
                                backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                                border: 1px solid rgba(255, 255, 255, 0.7);
                                border-radius: 20px;
                                padding: 20px;
                                margin-bottom: 25px;
                                box-shadow: 0 10px 30px rgba(108, 90, 158, 0.08);
                                transition: transform 0.3s ease;
                                opacity: 0;
                                transform: translateY(20px);
                                animation: fadeInUp 0.8s ease-out forwards;
                            }}
                            .chart-container:hover {{
                                transform: translateY(-4px);
                                box-shadow: 0 15px 40px rgba(108, 90, 158, 0.15);
                                background: rgba(255, 255, 255, 0.6);
                            }}
                            .chart-title {{
                                color: #4A3D73; font-weight: 600; font-size: 16px; 
                                margin-bottom: 15px; margin-top: 0; display: flex; align-items: center; gap: 8px;
                            }}
                            /* Staggered animation delays */
                            #c1 {{ animation-delay: 0.1s; }}
                            #c2 {{ animation-delay: 0.3s; }}
                            #c3 {{ animation-delay: 0.5s; }}
                            
                            @keyframes fadeInUp {{
                                to {{ opacity: 1; transform: translateY(0); }}
                            }}
                            
                            /* Container για τα canvas ώστε να είναι responsive */
                            .canvas-wrapper {{ position: relative; height: 220px; width: 100%; }}
                        </style>
                    </head>
                    <body>

                        <div class="chart-container" id="c1">
                            <h3 class="chart-title">🧠 Διακυμάνσεις Διάθεσης</h3>
                            <div class="canvas-wrapper"><canvas id="moodChart"></canvas></div>
                        </div>

                        <div class="chart-container" id="c2">
                            <h3 class="chart-title">😴 Ποιότητα Ύπνου (Ώρες)</h3>
                            <div class="canvas-wrapper"><canvas id="sleepChart"></canvas></div>
                        </div>

                        <div class="chart-container" id="c3">
                            <h3 class="chart-title">💧 Ενυδάτωση (Ποτήρια)</h3>
                            <div class="canvas-wrapper"><canvas id="waterChart"></canvas></div>
                        </div>

                        <script>
                            // Κοινές ρυθμίσεις για γραμματοσειρές και στυλ αξόνων
                            Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
                            Chart.defaults.color = '#73658a';
                            Chart.defaults.scale.grid.color = 'rgba(183, 157, 242, 0.15)';
                            
                            const commonOptions = {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    legend: {{ display: false }},
                                    tooltip: {{
                                        backgroundColor: 'rgba(74, 61, 115, 0.9)',
                                        titleFont: {{ size: 13 }},
                                        bodyFont: {{ size: 14, weight: 'bold' }},
                                        padding: 12,
                                        cornerRadius: 8,
                                        displayColors: false
                                    }}
                                }},
                                scales: {{
                                    x: {{ grid: {{ display: false }} }},
                                    y: {{ beginAtZero: true, border: {{ display: false }} }}
                                }},
                                interaction: {{ mode: 'index', intersect: false }},
                                elements: {{
                                    line: {{ tension: 0.4 }}, // 0.4 δίνει την απαλή καμπύλη (spline)
                                    point: {{ radius: 4, hoverRadius: 7, borderWidth: 2 }}
                                }}
                            }};

                            // Δεδομένα από Python
                            const labels = {dates_json};

                            // --- 1. ΔΙΑΓΡΑΜΜΑ ΔΙΑΘΕΣΗΣ ---
                            const ctxMood = document.getElementById('moodChart').getContext('2d');
                            let gradientMood = ctxMood.createLinearGradient(0, 0, 0, 220);
                            gradientMood.addColorStop(0, 'rgba(255, 143, 163, 0.6)'); // Ροζ απαλό
                            gradientMood.addColorStop(1, 'rgba(255, 143, 163, 0.0)');
                            
                            new Chart(ctxMood, {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Διάθεση (1-10)',
                                        data: {mood_json},
                                        borderColor: '#ff758f',
                                        backgroundColor: gradientMood,
                                        pointBackgroundColor: '#ffffff',
                                        pointBorderColor: '#ff758f',
                                        fill: true,
                                    }}]
                                }},
                                options: {{ ...commonOptions, scales: {{ y: {{ max: 10, min: 0 }} }} }}
                            }});

                            // --- 2. ΔΙΑΓΡΑΜΜΑ ΥΠΝΟΥ ---
                            const ctxSleep = document.getElementById('sleepChart').getContext('2d');
                            let gradientSleep = ctxSleep.createLinearGradient(0, 0, 0, 220);
                            gradientSleep.addColorStop(0, 'rgba(183, 157, 242, 0.6)'); // Λιλά
                            gradientSleep.addColorStop(1, 'rgba(183, 157, 242, 0.0)');

                            new Chart(ctxSleep, {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Ώρες Ύπνου',
                                        data: {sleep_json},
                                        borderColor: '#9d7bea',
                                        backgroundColor: gradientSleep,
                                        pointBackgroundColor: '#ffffff',
                                        pointBorderColor: '#9d7bea',
                                        fill: true,
                                    }}]
                                }},
                                options: commonOptions
                            }});

                            // --- 3. ΔΙΑΓΡΑΜΜΑ ΝΕΡΟΥ ---
                            const ctxWater = document.getElementById('waterChart').getContext('2d');
                            let gradientWater = ctxWater.createLinearGradient(0, 0, 0, 220);
                            gradientWater.addColorStop(0, 'rgba(137, 207, 240, 0.6)'); // Γαλάζιο/Κυανό
                            gradientWater.addColorStop(1, 'rgba(137, 207, 240, 0.0)');

                            new Chart(ctxWater, {{
                                type: 'line',
                                data: {{
                                    labels: labels,
                                    datasets: [{{
                                        label: 'Ποτήρια Νερό',
                                        data: {water_json},
                                        borderColor: '#5bc0eb',
                                        backgroundColor: gradientWater,
                                        pointBackgroundColor: '#ffffff',
                                        pointBorderColor: '#5bc0eb',
                                        fill: true,
                                    }}]
                                }},
                                options: commonOptions
                            }});
                        </script>
                    </body>
                    </html>
                    """, height=900, scrolling=True)


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
    # 1. Άσκηση αναπνοής 4–2–6 (ΜΕ JAVASCRIPT & ANIMATION)
    # --------------------------------------------------------
    st.markdown("### 1. Διαδραστική Άσκηση Αναπνοής 4–2–6")
    st.write("Συγχρόνισε την αναπνοή σου με τον κύκλο. Χρησιμεύει για άμεση μείωση του στρες.")
    
    # Εδώ γράφουμε καθαρή HTML, CSS και JAVASCRIPT!
    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    display: flex; justify-content: center; align-items: center;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    height: 250px; margin: 0; background-color: transparent;
                }
                .container {
                    text-align: center;
                }
                .circle {
                    width: 120px; height: 120px;
                    border-radius: 50%;
                    background: radial-gradient(circle, #ffb6c1 0%, #ff69b4 100%);
                    box-shadow: 0 10px 30px rgba(255, 105, 180, 0.4);
                    margin: 0 auto 20px auto;
                    display: flex; justify-content: center; align-items: center;
                    color: white; font-weight: bold; font-size: 18px;
                    transition: transform 4s ease-in-out; /* Ομαλό animation */
                }
                button {
                    background-color: #6C5A9E; color: white; border: none;
                    padding: 10px 24px; border-radius: 8px; font-size: 16px;
                    cursor: pointer; transition: 0.3s;
                }
                button:hover { background-color: #4A3D73; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="circle" id="breathCircle">Έτοιμος;</div>
                <br>
                <button id="startBtn" onclick="startBreathing()">Ξεκίνα Άσκηση</button>
            </div>

            <script>
                function startBreathing() {
                    const circle = document.getElementById('breathCircle');
                    const btn = document.getElementById('startBtn');
                    btn.style.display = 'none'; // Κρύβουμε το κουμπί

                    function breatheCycle() {
                        // 1. Εισπνοή (4 δευτερόλεπτα) - Ο κύκλος μεγαλώνει
                        circle.style.transition = 'transform 4s ease-in-out';
                        circle.style.transform = 'scale(1.6)';
                        circle.innerText = 'Εισπνοή (4)';

                        setTimeout(() => {
                            // 2. Κράτημα (2 δευτερόλεπτα) - Ο κύκλος μένει μεγάλος
                            circle.innerText = 'Κράτα (2)';

                            setTimeout(() => {
                                // 3. Εκπνοή (6 δευτερόλεπτα) - Ο κύκλος μικραίνει
                                circle.style.transition = 'transform 6s ease-in-out';
                                circle.style.transform = 'scale(1)';
                                circle.innerText = 'Εκπνοή (6)';
                            }, 2000); // Περιμένει τα 2 δευτερόλεπτα του κρατήματος

                        }, 4000); // Περιμένει τα 4 δευτερόλεπτα της εισπνοής
                    }

                    // Ξεκινάμε τον πρώτο κύκλο αμέσως
                    breatheCycle();
                    // Επαναλαμβάνουμε κάθε 12 δευτερόλεπτα (4+2+6)
                    setInterval(breatheCycle, 12000);
                }
            </script>
        </body>
        </html>
        """,
        height=300,
    )
    st.markdown("---")

    # --------------------------------------------------------
    # 2. Αποφόρτιση σκέψεων (100% LLM Powered)
    # --------------------------------------------------------
    st.markdown("### 2. Μικρή άσκηση αποφόρτισης σκέψεων")
    st.write("Σημείωσε αυτό που σε βαραίνει. Το σύστημα θα το διαβάσει με ενσυναίσθηση και θα σου επιστρέψει μια σκέψη φροντίδας πριν το αφήσεις να φύγει.")

    # Χρησιμοποιούμε session_state για να θυμόμαστε αν ολοκληρώθηκε η άσκηση
    if "discharge_done" not in st.session_state:
        st.session_state.discharge_done = False
        st.session_state.discharge_msg = ""

    # Ένα placeholder container για να μπορούμε να αλλάζουμε την οθόνη δυναμικά
    discharge_placeholder = st.empty()

    if not st.session_state.discharge_done:
        with discharge_placeholder.container():
            discharge_input = st.text_area(
                "📝 «Αυτό που με βαραίνει περισσότερο είναι…»",
                height=80,
                key="discharge_input_text"
            )
            
            if st.button("Αποφόρτιση ✨", key="btn_discharge"):
                if not discharge_input.strip():
                    st.warning("Γράψε κάτι μικρό πρώτα για να μπορέσω να βοηθήσω...")
                else:
                    with st.spinner("Δίνουμε χώρο στη σκέψη σου..."):
                        # ΚΛΗΣΗ ΣΤΟ LLM
                        profile = load_profile()
                        
                        # Στέλνουμε τη σκέψη σαν μέρος του context
                        chat_tail = get_chat_tail_from_state() + [f"Θέλω να αποφορτίσω αυτή τη σκέψη που με βαραίνει: {discharge_input}"]
                        active_mode = st.session_state.get("active_mode", "NONE")

                        key = "discharge_thought"
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

                        # Αποθήκευση και αλλαγή οθόνης
                        st.session_state.discharge_msg = msg
                        st.session_state.discharge_done = True
                        st.rerun()
    else:
        # Αν η αποφόρτιση έγινε, δείχνουμε την απάντηση του LLM στο γυάλινο design
        with discharge_placeholder.container():
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.45);
                            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.6);
                            border-radius: 16px; padding: 30px; text-align: center; 
                            box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08);
                            animation: cardFadeIn 1s ease-out; margin-bottom: 15px;">
                    <div style="font-size: 45px; margin-bottom: 15px; opacity: 0.9;">🍃</div>
                    <div style="color: #3b304c; font-size: 1.1rem; line-height: 1.6; font-weight: 500;">
                        {st.session_state.discharge_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Κουμπί για επαναφορά
            col_spacer1, col_reset, col_spacer2 = st.columns([1, 1, 1])
            with col_reset:
                if st.button("🔄 Νέα αποφόρτιση", key="reset_discharge", use_container_width=True):
                    st.session_state.discharge_done = False
                    st.session_state.discharge_msg = ""
                    st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # 3. Θλίψη / μοναξιά (100% LLM Powered - Native Glassmorphism)
    # --------------------------------------------------------
    st.markdown("### 3. Άσκηση ηρεμίας για θλίψη / μοναξιά")
    st.write("Μια μικρή παύση για να υπενθυμίσεις στον εαυτό σου ότι δεν είσαι μόνος/η, προσαρμοσμένη ακριβώς σε αυτό που νιώθεις.")

    if "lonely_done" not in st.session_state:
        st.session_state.lonely_done = False
        st.session_state.lonely_msg = ""

    lonely_placeholder = st.empty()

    if not st.session_state.lonely_done:
        with lonely_placeholder.container():
            st.markdown(
                """
                <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; 
                            padding: 30px; text-align: center; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08);">
                    <div style="font-size: 60px; margin-bottom: 15px; display: inline-block; animation: softPulse 3s infinite;">🤍</div>
                    <div style="color: #3b304c; font-size: 1.1rem; margin-bottom: 20px; font-weight: 500;">
                        Όταν νιώθεις μοναξιά ή βάρος, κάνε αυτό το μικρό δώρο στον εαυτό σου.
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Βάζουμε κενά δεξιά-αριστερά για να μην απλώνει τελείως το κουμπί
            col_l1, col_lbtn, col_l2 = st.columns([1, 2, 1])
            with col_lbtn:
                if st.button("Ξεκίνα τη στιγμή φροντίδας ✨", key="btn_lonely_start", use_container_width=True):
                    with st.spinner("Αφουγκράζομαι το πώς νιώθεις..."):
                        profile = load_profile()
                        
                        # Στέλνουμε το chat context + μια κρυφή "σκηνοθετική" οδηγία στο LLM
                        chat_tail = get_chat_tail_from_state() 
                        chat_tail.append(
                            "Ο χρήστης μόλις πάτησε το κουμπί για την άσκηση ανακούφισης από μοναξιά/θλίψη. "
                            "Βάσει του ιστορικού και του προφίλ του, δώσε του 2-3 προτάσεις βαθιάς, ζεστής ενσυναίσθησης "
                            "που να τον αγγίζουν, και κλείσε με 1 θετική επιβεβαίωση (affirmation) σε εισαγωγικά που να μπορεί να πει στον εαυτό του."
                        )
                        
                        active_mode = st.session_state.get("active_mode", "NONE")
                        key = "lonely_care"
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

                        st.session_state.lonely_msg = msg
                        st.session_state.lonely_done = True
                        st.rerun()
    else:
        # Εμφάνιση της προσωποποιημένης απάντησης
        with lonely_placeholder.container():
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; 
                            padding: 30px; text-align: center; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08);
                            animation: cardFadeIn 1s ease-out;">
                    <div style="font-size: 60px; margin-bottom: 15px; display: inline-block; animation: softPulse 2s infinite;">💖</div>
                    <div style="color: #3b304c; font-size: 1.05rem; line-height: 1.6; margin-bottom: 20px;">
                        <strong>1.</strong> Βάλε το χέρι σου στο κέντρο του στήθους και νιώσε τη ζεστασιά σου.<br>
                        <strong>2.</strong> Πάρε μια αργή, βαθιά ανάσα.<br>
                        <strong>3.</strong> Διάβασε αυτά τα λόγια:
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.75); border-left: 4px solid #ff8fa3; 
                                padding: 20px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.02);">
                        <div style="color: #d81b60; font-size: 1.1rem; line-height: 1.6; font-weight: 500; font-style: italic;">
                            {st.session_state.lonely_msg}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            col_l1, col_lbtn, col_l2 = st.columns([1, 1, 1])
            with col_lbtn:
                if st.button("🔄 Επιστροφή", key="reset_lonely", use_container_width=True):
                    st.session_state.lonely_done = False
                    st.session_state.lonely_msg = ""
                    st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # 4. Νερό (ΜΕ JAVASCRIPT ANIMATION - ΓΕΜΙΣΜΑ ΠΟΤΗΡΙΟΥ)
    # --------------------------------------------------------
    st.markdown("### 4. Μικρή άσκηση φροντίδας σώματος (Ενυδάτωση)")

    col_w1, col_w2 = st.columns([1.2, 1])

    with col_w1:
        components.html(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { 
                        margin: 0; display: flex; justify-content: center; align-items: center; 
                        font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; background: transparent; height: 240px;
                    }
                    .container { text-align: center; width: 100%; }
                    .info-text { 
                        color: #4A3D73; font-weight: 600; font-size: 15px; margin-bottom: 15px;
                    }
                    /* Το ποτήρι */
                    .glass {
                        width: 70px; height: 110px;
                        border: 4px solid #acd8e5;
                        border-top: none;
                        border-radius: 0 0 15px 15px;
                        position: relative;
                        overflow: hidden;
                        margin: 0 auto;
                        cursor: pointer;
                        background: rgba(255, 255, 255, 0.7);
                        box-shadow: 0 8px 15px rgba(172, 216, 229, 0.3);
                        transition: transform 0.3s ease;
                    }
                    .glass:hover {
                        transform: translateY(-4px);
                        box-shadow: 0 12px 20px rgba(172, 216, 229, 0.5);
                    }
                    /* Το νερό με το εφέ κύματος */
                    .water {
                        position: absolute;
                        bottom: 0;
                        left: -50%;
                        width: 200%;
                        height: 15%; /* Αρχίζει σχεδόν άδειο */
                        background: #89cff0;
                        transition: height 2.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                        border-radius: 40%;
                        animation: wave-spin 3s infinite linear;
                        opacity: 0.85;
                    }
                    @keyframes wave-spin {
                        100% { transform: rotate(360deg); }
                    }
                    .hidden-msg { 
                        opacity: 0; transition: opacity 1s; color: #2e7d32; 
                        margin-top: 15px; font-weight: 600; font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="info-text">Πιες 1 ποτήρι νερό & κάνε κλικ:</div>
                    
                    <div class="glass" onclick="fillWater()" title="Κάνε κλικ για να γεμίσει!">
                       <div class="water" id="waterLvl"></div>
                    </div>
                    
                    <div id="cheers" class="hidden-msg">Εξαιρετικά! 💧 Το σώμα σου σε ευχαριστεί.</div>
                </div>
                <script>
                    function fillWater() {
                        const water = document.getElementById('waterLvl');
                        const msg = document.getElementById('cheers');
                        
                        // Ανεβάζουμε τη στάθμη του νερού πάνω από 100% για να γεμίσει το ποτήρι
                        water.style.height = '120%'; 
                        
                        // Εμφάνιση του μηνύματος αφού γεμίσει το νερό
                        setTimeout(() => {
                            msg.style.opacity = 1;
                        }, 1800);
                    }
                </script>
            </body>
            </html>
            """,
            height=250,
        )

    with col_w2:
        st.write(
            "Ένα ποτήρι νερό είναι μια πολύ μικρή, αλλά ουσιαστική πράξη φροντίδας "
            "για το νευρικό σου σύστημα.\n\n"
            "Πάρε 3 αργές αναπνοές καθώς το πίνεις."
        )
        # Κρατάμε το LLM button ανέπαφο!
        if st.button("✨ Φράση για το νερό", key="water_nudge"):
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
    # 5. 3 μικρά στηρίγματα (100% LLM Powered)
    # --------------------------------------------------------
    st.markdown("### 5. Άσκηση «3 μικρά στηρίγματα της ημέρας»")
    st.write("Σημείωσε τρία μικρά πράγματα που σε στήριξαν σήμερα (όσο μικρά κι αν φαίνονται). Το σύστημα θα τα διαβάσει και θα σου δώσει μια δική σου, μοναδική οπτική γι' αυτά.")

    if "supports_done" not in st.session_state:
        st.session_state.supports_done = False
        st.session_state.supports_msg = ""
        st.session_state.user_supports = []

    supports_placeholder = st.empty()

    if not st.session_state.supports_done:
        with supports_placeholder.container():
            # Χρησιμοποιούμε στήλες για να μπουν δίπλα-δίπλα τα πεδία
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                s1 = st.text_input("1ο στήριγμα...", key="sup1")
            with col_s2:
                s2 = st.text_input("2ο στήριγμα...", key="sup2")
            with col_s3:
                s3 = st.text_input("3ο στήριγμα...", key="sup3")
                
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("Κράτησε τα στηρίγματα ✨", use_container_width=True, key="btn_save_supports"):
                    supports = [s for s in [s1, s2, s3] if s.strip()]
                    if not supports:
                        st.warning("Γράψε τουλάχιστον ένα στήριγμα για να προχωρήσουμε 🌿")
                    else:
                        with st.spinner("Συνθέτω τα στηρίγματά σου..."):
                            profile = load_profile()
                            supports_text = ", ".join(supports)
                            
                            # Σκηνοθετική οδηγία προς το LLM!
                            chat_tail = get_chat_tail_from_state()
                            chat_tail.append(
                                f"Ο χρήστης μόλις κατέγραψε τα εξής στηρίγματα για τη μέρα του: [{supports_text}]. "
                                "Βάσει του ιστορικού και του προφίλ του, γράψε 2-3 προτάσεις που να αναγνωρίζουν "
                                "την αξία ΑΥΤΩΝ των συγκεκριμένων πραγμάτων. Δείξε του πώς αυτά τα μικρά πράγματα "
                                "δείχνουν τη δύναμή του ή τη φροντίδα προς τον εαυτό του. Μην δώσεις συμβουλές, απλώς "
                                "επικύρωσε (validate) συναισθηματικά αυτή του την προσπάθεια."
                            )
                            
                            active_mode = st.session_state.get("active_mode", "NONE")
                            key = "supports_exercise"
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

                            # Αποθήκευση στο session state
                            st.session_state.user_supports = supports
                            st.session_state.supports_msg = msg
                            st.session_state.supports_done = True
                            st.rerun()
    else:
        # Εμφάνιση των αποτελεσμάτων σε Glassmorphism UI
        with supports_placeholder.container():
            # Φτιάχνουμε δυναμικά τις κάρτες (pillars) για όσα στηρίγματα έγραψε
            import html
            pillars_html = "".join([
                f"<div style='background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); border-left: 4px solid #b79df2; padding: 12px 20px; margin: 10px auto; border-radius: 8px; font-weight: 600; color: #4A3D73; width: 80%; box-shadow: 0 2px 8px rgba(108, 90, 158, 0.15);'>{html.escape(s)}</div>" 
                for s in st.session_state.user_supports
            ])
            
            # ΕΔΩ: Η HTML χωρίς κενά στην αρχή της κάθε γραμμής
            st.markdown(
                f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.8s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.1rem; margin-top: 0;">Τα σημερινά σου θεμέλια:</h3>
{pillars_html}
<div style="background: rgba(255, 255, 255, 0.75); border-left: 4px solid #baf7c3; padding: 20px; border-radius: 12px; margin-top: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.02);">
<div style="color: #158047; font-size: 1.05rem; line-height: 1.6; font-weight: 500; font-style: italic;">
{st.session_state.supports_msg}
</div>
</div>
</div>""", 
                unsafe_allow_html=True
            )
            
            col_l1, col_lbtn, col_l2 = st.columns([1, 1, 1])
            with col_lbtn:
                if st.button("🔄 Νέα καταγραφή", key="reset_supports", use_container_width=True):
                    st.session_state.supports_done = False
                    st.session_state.supports_msg = ""
                    st.session_state.user_supports = []
                    st.rerun()
                    
    st.markdown("---")

    # --------------------------------------------------------
    # 6. Body scan (ΜΕ JAVASCRIPT ZEN ANIMATION)
    # --------------------------------------------------------
    st.markdown("### 6. Μικρή σωματική σάρωση (Guided Body Scan)")

    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    display: flex; justify-content: center; align-items: center; background: transparent;
                }
                .scan-card {
                    width: 100%; max-width: 600px; min-height: 280px;
                    background: linear-gradient(145deg, #f3f0fb 0%, #ffffff 100%);
                    border-radius: 16px; border: 1px solid #e6e0f8;
                    box-shadow: 0 10px 30px rgba(108, 90, 158, 0.08);
                    display: flex; flex-direction: column; align-items: center; justify-content: center;
                    text-align: center; padding: 20px; position: relative; overflow: hidden;
                }
                /* Η σφαίρα εστίασης */
                .orb {
                    width: 140px; height: 140px; border-radius: 50%;
                    background: radial-gradient(circle, #b19cd9 0%, #6C5A9E 100%);
                    display: flex; justify-content: center; align-items: center;
                    color: white; font-weight: 600; font-size: 15px; letter-spacing: 0.5px;
                    box-shadow: 0 0 20px rgba(108, 90, 158, 0.4);
                    opacity: 0; transform: scale(0.5); /* Αρχικά κρυμμένη */
                    transition: all 1.5s ease-in-out;
                    text-align: center; padding: 10px;
                }
                .orb.active {
                    opacity: 1; transform: scale(1);
                    animation: breathe-pulse 4s infinite alternate ease-in-out;
                }
                @keyframes breathe-pulse {
                    0% { box-shadow: 0 0 15px rgba(108, 90, 158, 0.3); transform: scale(1); }
                    100% { box-shadow: 0 0 35px rgba(108, 90, 158, 0.6); transform: scale(1.08); }
                }
                .intro-text {
                    color: #4A3D73; font-size: 16px; margin-bottom: 20px; line-height: 1.5;
                    transition: opacity 1s;
                }
                .focus-text {
                    opacity: 1; transition: opacity 1s ease-in-out;
                }
                button {
                    background: #6C5A9E; color: white; border: none; padding: 12px 28px;
                    border-radius: 25px; font-size: 15px; cursor: pointer; transition: 0.3s; font-weight: 600;
                }
                button:hover { background: #4A3D73; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(108, 90, 158, 0.2); }
            </style>
        </head>
        <body>
            <div class="scan-card">
                <div id="intro" class="intro-text">
                    <strong>1 Λεπτό Σωματικής Σάρωσης</strong><br><br>
                    Κλείσε τα μάτια, πάρε μια ανάσα και όταν είσαι έτοιμος/η πάτα το κουμπί. <br>
                    Μόνο παρατήρηση του σώματος — χωρίς προσπάθεια για διόρθωση.
                </div>
                
                <div id="orb" class="orb">
                    <span id="orbText" class="focus-text">Μέτωπο & Μάτια</span>
                </div>
                
                <br>
                <button id="startBtn" onclick="startScan()">Ξεκίνα τη σάρωση</button>
            </div>

            <script>
                function startScan() {
                    const btn = document.getElementById('startBtn');
                    const intro = document.getElementById('intro');
                    const orb = document.getElementById('orb');
                    const orbText = document.getElementById('orbText');

                    // Τα στάδια της σάρωσης
                    const stages = [
                        "Μέτωπο & Μάτια",
                        "Ώμοι & Αυχένας",
                        "Στήθος & Κοιλιά",
                        "Παλάμες & Πέλματα",
                        "Ολοκλήρωση 🌿"
                    ];
                    
                    // Κρύβουμε εισαγωγή και κουμπί
                    btn.style.display = 'none';
                    intro.style.display = 'none';
                    
                    // Εμφανίζουμε τη σφαίρα
                    orb.classList.add('active');

                    let currentStage = 0;

                    // Λειτουργία για αλλαγή κειμένου με fade out -> αλλαγή -> fade in
                    function nextStage() {
                        currentStage++;
                        if (currentStage < stages.length) {
                            orbText.style.opacity = 0; // Fade out
                            
                            setTimeout(() => {
                                orbText.innerText = stages[currentStage];
                                orbText.style.opacity = 1; // Fade in
                            }, 1000); // Περιμένουμε 1 δευτερόλεπτο να σβήσει
                        } else {
                            // Τέλος άσκησης
                            setTimeout(() => {
                                orb.classList.remove('active');
                                orb.style.opacity = 0;
                                setTimeout(() => {
                                    intro.innerHTML = "<strong>Ολοκληρώθηκε!</strong><br>Ελπίζω να νιώθεις λίγο πιο γειωμένος/η.";
                                    intro.style.display = 'block';
                                }, 1500);
                            }, 3000);
                        }
                    }

                    // Αλλάζουμε περιοχή του σώματος κάθε 8 δευτερόλεπτα (Συνολικά περίπου 40-45 sec)
                    const scanInterval = setInterval(() => {
                        if (currentStage >= stages.length - 1) {
                            clearInterval(scanInterval);
                        }
                        nextStage();
                    }, 8000); 
                }
            </script>
        </body>
        </html>
        """,
        height=320,
    )
    st.markdown("---")

    # --------------------------------------------------------
    # 7. Μικρό παιχνίδι – φροντιστική απάντηση (ΜΕ 3D FLIP CARD)
    # --------------------------------------------------------
    st.markdown("### 🧩 Μικρό παιχνίδι: «Απάντηση φροντίδας»")
    st.write("Επίλεξε μια δύσκολη σκέψη. Δες πώς μπορεί να μεταμορφωθεί όταν της δώσεις λίγο χώρο.")

    difficult_thoughts = [
        "Αποτυγχάνω σε όλα.",
        "Δεν αξίζω την αγάπη των άλλων.",
        "Οι άλλοι θα με θεωρήσουν αδύναμο/η.",
        "Δεν θα αλλάξει ποτέ τίποτα.",
    ]

    chosen_label = st.selectbox(
        "Διάλεξε μία δύσκολη σκέψη:",
        difficult_thoughts,
        key="mh_game_select",
    )

    if st.button("Μεταμόρφωσε τη σκέψη ✨", key="mh_game_button"):
        with st.spinner("Το σύστημα σκέφτεται μια πιο ζεστή οπτική..."):
            # Κλήση στο LLM (Backend)
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
            
            # Προετοιμασία κειμένων για να μπουν με ασφάλεια στην HTML
            safe_thought = html.escape(chosen_label).replace('\n', '<br>')
            safe_msg = html.escape(msg).replace('\n', '<br>')

            # Το 3D Flip Card (Frontend)
            components.html(
                f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{ 
                            margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                            display: flex; justify-content: center; background: transparent; padding-top: 15px; 
                        }}
                        .flip-card {{
                            background-color: transparent; width: 100%; max-width: 600px; 
                            height: 220px; perspective: 1000px;
                        }}
                        .flip-card-inner {{
                            position: relative; width: 100%; height: 100%; text-align: center;
                            transition: transform 1.2s cubic-bezier(0.4, 0.2, 0.2, 1);
                            transform-style: preserve-3d; cursor: pointer;
                        }}
                        /* Η κλάση που γυρνάει την κάρτα */
                        .flipped {{ transform: rotateY(180deg); }}
                        
                        .flip-card-front, .flip-card-back {{
                            position: absolute; width: 100%; height: 100%;
                            -webkit-backface-visibility: hidden; backface-visibility: hidden;
                            border-radius: 16px; display: flex; flex-direction: column;
                            justify-content: center; align-items: center; padding: 25px;
                            box-sizing: border-box; box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                        }}
                        /* Σκοτεινή πλευρά (Δύσκολη Σκέψη) */
                        .flip-card-front {{
                            background: linear-gradient(135deg, #4c4c4c 0%, #2a2a2a 100%);
                            color: #f1f1f1; border: 1px solid #555;
                        }}
                        /* Φωτεινή πλευρά (Φροντίδα - LLM) */
                        .flip-card-back {{
                            background: linear-gradient(135deg, #ffffff 0%, #f3f0fb 100%);
                            color: #4A3D73; border: 2px solid #b19cd9;
                            transform: rotateY(180deg);
                        }}
                        .hint {{ font-size: 13px; opacity: 0.6; margin-top: 15px; font-weight: 400; }}
                        .front-text {{ font-size: 19px; font-weight: 500; font-style: italic; }}
                        .back-text {{ font-size: 16px; line-height: 1.6; font-weight: 500; }}
                    </style>
                </head>
                <body>
                    <div class="flip-card" onclick="this.querySelector('.flip-card-inner').classList.toggle('flipped')">
                        <div class="flip-card-inner">
                            <div class="flip-card-front">
                                <div class="front-text">«{safe_thought}»</div>
                                <div class="hint">Περίμενε...</div>
                            </div>
                            <div class="flip-card-back">
                                <div class="back-text">{safe_msg}</div>
                                <div class="hint">✨ Κάνε κλικ για να γυρίσεις την κάρτα</div>
                            </div>
                        </div>
                    </div>
                    
                    <script>
                        // Αυτόματο γύρισμα μετά από 1.2 δευτερόλεπτα για να αποκαλυφθεί η απάντηση!
                        setTimeout(() => {{
                            const card = document.querySelector('.flip-card-inner');
                            card.classList.add('flipped');
                            document.querySelector('.flip-card-front .hint').innerText = "👆 Κάνε κλικ";
                        }}, 1200);
                    </script>
                </body>
                </html>
                """,
                height=260,
            )

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
# ΠΡΟΦΙΛ TAB (Interactive Glass Dashboard - Minimal)
# ============================================================

with tab_profile:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>🌿 Προφίλ Φροντίδας</h1>
          <p class='page-subtitle'>Μοιράσου όσα νιώθεις άνετα. Όσο πιο πολύ σε γνωρίζω, τόσο πιο σωστά θα σε υποστηρίζω.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = load_profile()

    # --- ΚΕΝΤΡΙΚΗ ΠΕΡΙΟΧΗ (Glass Wrapper) ---
    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)
    
    # Ενότητα 1: Ταυτότητα
    st.markdown("<h3 style='color: #4A3D73; font-size: 1.1rem; border-bottom: 2px solid rgba(212, 190, 250, 0.3); padding-bottom: 8px; margin-top: 0;'>👤 Ποιος/α είσαι;</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        name = st.text_input("Όνομα ή ψευδώνυμο", value=profile.get("name", ""))
    with col2:
        stored_age = profile.get("age", 0)
        try:
            stored_age = int(stored_age)
        except:
            stored_age = 0
        age = st.number_input("Ηλικία (προαιρετικό)", min_value=0, max_value=120, value=stored_age, step=1)
    with col3:
        context = st.text_input("Ρόλος/Ασχολία (π.χ. φοιτητής, υπάλληλος)", value=profile.get("context", ""))

    age_range = infer_age_range(age) if age else profile.get("age_range", "")

    # Ενότητα 2: Εσωτερικός Κόσμος
    st.markdown("<h3 style='color: #4A3D73; font-size: 1.1rem; border-bottom: 2px solid rgba(212, 190, 250, 0.3); padding-bottom: 8px; margin-top: 30px;'>🧭 Στόχοι & Δυσκολίες</h3>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        main_goals = st.text_area("Ποιοι είναι οι βασικοί σου στόχοι ευεξίας;", value=profile.get("main_goals", ""), height=120)
    with col_g2:
        main_struggles = st.text_area("Τι σε δυσκολεύει περισσότερο τον τελευταίο καιρό;", value=profile.get("main_struggles", ""), height=120)
    
    helpful_things = st.text_area("Τι σε βοηθά συνήθως να νιώσεις καλύτερα (ακόμη κι αν είναι κάτι μικρό);", value=profile.get("helpful_things", ""), height=80)

    # Ενότητα 3: Προτιμήσεις AI
    st.markdown("<h3 style='color: #4A3D73; font-size: 1.1rem; border-bottom: 2px solid rgba(212, 190, 250, 0.3); padding-bottom: 8px; margin-top: 30px;'>🤖 Προτιμήσεις Αλληλεπίδρασης</h3>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        preferred_tone = st.text_area("Πώς θες να σου μιλάω; (π.χ. πρακτικός, ζεστός, άμεσος)", value=profile.get("preferred_tone", ""), height=100)
    with col_p2:
        triggers = st.text_area("Θέματα ή λέξεις που προτιμάς να αποφεύγουμε;", value=profile.get("triggers", ""), height=100)
    
    soothing_things = st.text_area("Υπάρχουν εικόνες, σκέψεις ή συνήθειες που σε ηρεμούν;", value=profile.get("soothing_things", ""), height=80)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- ΚΟΥΜΠΙ ΑΠΟΘΗΚΕΥΣΗΣ ΚΕΝΤΡΑΡΙΣΜΕΝΟ ---
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("💾 Αποθήκευση & Ενημέρωση Προφίλ", use_container_width=True):
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
            
            # Μήνυμα επιτυχίας
            st.markdown(
                """
                <div style="background: #baf7c3; color: #158047; padding: 15px; border-radius: 12px; 
                            text-align: center; font-weight: 600; margin-top: 15px; border: 1px solid #99deb0;
                            box-shadow: 0 4px 15px rgba(21, 128, 71, 0.15); animation: cardFadeIn 0.5s;">
                    ✨ Το προφίλ σου ενημερώθηκε!
                </div>
                """, unsafe_allow_html=True
            )
            
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ΣΥΝΔΕΣΗ ΑΠΟ ΚΙΝΗΤΟ TAB (Interactive Scanner UI)
# ============================================================

with tab_mobile:
    import base64
    from io import BytesIO

    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>📱 Σύνδεση από κινητό</h1>
          <p class='page-subtitle'>
            Συνέχισε την ίδια συνεδρία στο κινητό σου, μέσα από το τοπικό σου δίκτυο.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Δημιουργία URL και QR Image
    local_url = get_local_url(port=8501)
    qr_img = generate_qr_image(local_url)
    
    # Μετατροπή της εικόνας QR σε Base64 για ενσωμάτωση στην HTML
    buffered = BytesIO()
    qr_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Το Interactive QR Scanner Component
    components.html(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0; padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .qr-container {{
                background: rgba(255, 255, 255, 0.45);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid rgba(255, 255, 255, 0.7);
                border-radius: 24px;
                padding: 40px;
                max-width: 800px;
                width: 100%;
                box-shadow: 0 15px 40px rgba(108, 90, 158, 0.08);
                display: flex;
                gap: 40px;
                align-items: center;
                animation: cardFadeIn 0.8s ease-out;
            }}
            @media (max-width: 768px) {{
                .qr-container {{ flex-direction: column; text-align: center; padding: 30px 20px; gap: 25px; }}
            }}
            
            /* Το Κουτί του QR με το Animation Σάρωσης */
            .qr-box-wrapper {{
                position: relative;
                background: #ffffff;
                padding: 15px;
                border-radius: 20px;
                box-shadow: 0 10px 25px rgba(108, 90, 158, 0.15);
                flex-shrink: 0;
                overflow: hidden;
            }}
            .qr-box-wrapper img {{
                display: block;
                width: 180px;
                height: 180px;
                border-radius: 10px;
            }}
            /* Η φωτεινή "Laser" γραμμή */
            .scanner-laser {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: rgba(183, 157, 242, 0.9);
                box-shadow: 0 0 15px rgba(183, 157, 242, 0.9), 0 0 30px rgba(183, 157, 242, 0.6);
                animation: scan 2.5s infinite alternate ease-in-out;
            }}
            @keyframes scan {{
                0% {{ top: 5%; opacity: 0; }}
                10% {{ opacity: 1; }}
                90% {{ opacity: 1; }}
                100% {{ top: 95%; opacity: 0; }}
            }}
            @keyframes cardFadeIn {{
                from {{ opacity: 0; transform: translateY(15px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            /* Περιοχή κειμένου & κουμπιών */
            .info-area {{ flex-grow: 1; }}
            
            .badge {{
                display: inline-block; padding: 6px 14px; border-radius: 20px;
                font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
                background: linear-gradient(135deg, #e4d4ff, #f2eaff); color: #4A3D73;
                margin-bottom: 15px; border: 1px solid rgba(183, 157, 242, 0.5);
                box-shadow: 0 4px 10px rgba(183, 157, 242, 0.2);
            }}
            h2 {{ margin: 0 0 10px 0; color: #3b304c; font-size: 24px; }}
            p {{ color: #73658a; font-size: 15px; line-height: 1.6; margin: 0 0 20px 0; }}
            
            /* Input Group για Αντιγραφή του URL */
            .copy-group {{
                display: flex; background: rgba(245, 247, 251, 0.6);
                border: 1px solid rgba(183, 157, 242, 0.4); border-radius: 12px;
                overflow: hidden; backdrop-filter: blur(5px); transition: all 0.3s;
            }}
            .copy-group:focus-within {{
                border-color: #b79df2; box-shadow: 0 0 0 2px rgba(183, 157, 242, 0.2);
            }}
            .copy-input {{
                flex-grow: 1; background: transparent; border: none; padding: 12px 15px;
                color: #4A3D73; font-family: monospace; font-size: 14px; outline: none;
            }}
            .copy-btn {{
                background: #e4d4ff; color: #4A3D73; border: none; padding: 0 20px;
                font-weight: 600; cursor: pointer; transition: all 0.3s;
            }}
            .copy-btn:hover {{ background: #d4befa; }}
            .copy-btn.success {{ background: #baf7c3; color: #158047; }}
        </style>
    </head>
    <body>
        <div class="qr-container">
            <div class="qr-box-wrapper">
                <!-- Εδώ μπαίνει το QR code από την Python -->
                <img src="data:image/png;base64,{img_str}" alt="QR Code">
                <!-- Η γραμμή σάρωσης -->
                <div class="scanner-laser"></div>
            </div>
            
            <div class="info-area">
                <div class="badge">🌐 Τοπικο Δικτυο</div>
                <h2>Συνδέσου από το κινητό</h2>
                <p>
                    Άνοιξε την κάμερα του κινητού σου και σκάναρε τον κωδικό. 
                    Θα μεταφερθείς απευθείας στο Project Wellness, με την προϋπόθεση 
                    ότι και οι δύο συσκευές είναι στο <b>ίδιο δίκτυο Wi-Fi</b>.
                </p>
                
                <div class="copy-group">
                    <input type="text" class="copy-input" id="urlInput" value="{local_url}" readonly>
                    <button class="copy-btn" id="copyBtn" onclick="copyUrl()">Αντιγραφή</button>
                </div>
            </div>
        </div>
        
        <script>
            // Λειτουργία JavaScript για το κουμπί Αντιγραφής
            function copyUrl() {{
                const input = document.getElementById('urlInput');
                const btn = document.getElementById('copyBtn');
                
                // Επιλογή του κειμένου & αντιγραφή στο πρόχειρο
                input.select();
                document.execCommand('copy');
                
                // Αλλαγή στυλ του κουμπιού σε "Επιτυχία"
                btn.innerHTML = 'Αντιγράφηκε! ✓';
                btn.classList.add('success');
                
                // Επαναφορά στην αρχική μορφή μετά από 2 δευτερόλεπτα
                setTimeout(() => {{
                    btn.innerHTML = 'Αντιγραφή';
                    btn.classList.remove('success');
                }}, 2000);
            }}
        </script>
    </body>
    </html>
    """, height=380)


# ============================================================
# ΣΧΕΤΙΚΑ & ΑΣΦΑΛΕΙΑ TAB (Glassmorphism Dashboard)
# ============================================================

with tab_info:
    st.markdown(
        """
        <div class="page-header">
          <h1 class='page-title'>🛡️ Σχετικά & Ασφάλεια</h1>
          <p class='page-subtitle'>Διαφάνεια, σεβασμός στα δεδομένα σου και εργαλεία υποστήριξης.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    # Χρησιμοποιούμε 2 στήλες για πιο σύγχρονο, Dashboard layout
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.4s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">🧠 Πώς Λειτουργεί</h3>
<p style="color: #73658a; line-height: 1.6; font-size: 0.95rem;">
Το Project Wellness χρησιμοποιεί προηγμένη Τεχνητή Νοημοσύνη, ρυθμισμένη να απαντά με τη <strong>Μαιευτική Μέθοδο (Socratic Questioning)</strong>. 
Δεν δίνει έτοιμες, ρομποτικές λύσεις, αλλά σε βοηθά να αναγνωρίσεις γνωστικές παγίδες και να βρεις τις δικές σου απαντήσεις μέσα από στοχευμένες ερωτήσεις.
</p>
</div>""", 
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.6s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">🚨 Μέτρα Ασφαλείας</h3>
<p style="color: #73658a; line-height: 1.6; font-size: 0.95rem;">
Το σύστημα διαθέτει μηχανισμούς ανίχνευσης λέξεων-κλειδιών. Αν αντιληφθεί ένδειξη κρίσης ή κινδύνου, 
σταματά την κανονική ροή συζήτησης και ενεργοποιεί ένα <strong>ειδικό πρωτόκολλο ασφαλείας</strong> με πηγές άμεσης βοήθειας.
</p>
</div>""", 
            unsafe_allow_html=True
        )

    with col_i2:
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.5s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">🔒 Απόρρητο Δεδομένων</h3>
<p style="color: #73658a; line-height: 1.6; font-size: 0.95rem;">
Το ιστορικό σου αποθηκεύεται <strong>τοπικά</strong> στον υπολογιστή σου (στο αρχείο <code>user_data.csv</code>) και δεν ανεβαίνει σε κάποια εξωτερική βάση δεδομένων. 
Τα δεδομένα που στέλνονται στο API του OpenAI είναι κρυπτογραφημένα και <strong>δεν</strong> χρησιμοποιούνται για την εκπαίδευση μελλοντικών μοντέλων τους.
</p>
</div>""", 
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Κάρτα Γραμμών Βοήθειας (Πιο έντονο ροζ/κόκκινο πλαίσιο για να τραβάει την προσοχή)
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid #ffcad4; border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(255, 105, 180, 0.12); border-left: 5px solid #ff758f; animation: cardFadeIn 0.7s ease-out;">
<h3 style="color: #c12530; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">☎️ Γραμμές Βοήθειας (24/7)</h3>
<ul style="color: #4A3D73; line-height: 1.8; font-size: 0.95rem; padding-left: 20px; font-weight: 500;">
    <li><strong>112</strong> – Άμεση ανάγκη (Πανευρωπαϊκός)</li>
    <li><strong>1018</strong> – Γραμμή Παρέμβασης για την Αυτοκτονία</li>
    <li><strong>10306</strong> – Γραμμή Ψυχοκοινωνικής Υποστήριξης</li>
    <li><strong>1056</strong> – Το Χαμόγελο του Παιδιού</li>
</ul>
</div>""", 
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Disclaimer Box στο κάτω μέρος
    st.markdown(
        f"""<div style="text-align: center; color: #73658a; font-size: 0.85rem; padding: 20px; background: rgba(245, 247, 251, 0.6); border-radius: 12px; border: 1px solid rgba(183, 157, 242, 0.3);">
<strong>⚠️ Σημαντική Υπενθύμιση:</strong> Το Project Wellness είναι ένα πειραματικό εργαλείο αυτοβοήθειας. Σε καμία περίπτωση δεν υποκαθιστά την κλινική διάγνωση, τον ψυχολόγο ή τον ψυχίατρο.
</div>""", 
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)