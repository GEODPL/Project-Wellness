"""
prompts.py
==========
Χτίσιμο του prompt που στέλνουμε στο LLM (context, mode blending, προφίλ,
rolling memory), καθώς και η λογική του "wrap-up" όταν κλείνει ο διάλογος
(σύνοψη + συναισθηματικός χάρτης + άσκηση) και ο δείκτης προτεινόμενης
υποστήριξης (support-need score).

ΣΗΜΕΙΩΣΗ: το build_llm_user_prompt παίρνει το `profile` ως ρητή παράμετρο
(αντί να το "δανείζεται" από το περιβάλλον γύρω του), ακριβώς επειδή τώρα
ζει σε ξεχωριστό αρχείο και δεν έχει πρόσβαση σε τοπικές μεταβλητές του
Chat tab.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from modes import MODES
from llm import llm_therapeutic_reply
from rules import exercise_suggestion
from emotional_map import render_emotional_map


def safe_trim(text: str, max_chars: int = 1400) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
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


def get_dialogue_context_text(max_pairs: int = 10) -> str:
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


def build_llm_user_prompt(
    raw_user_text: str,
    mood_value: int,
    sleep: int,
    water: int,
    active_mode: Optional[str],
    mode_weights: dict,
    profile: Dict[str, Any],
) -> str:
    ctx = get_dialogue_context_text(max_pairs=10)
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
- 4–10 προτάσεις, φυσικές.
- 0–2 ερωτήσεις μόνο αν βοηθούν.
- Όχι bullets.
- Μην εμφανίσεις reasoning μέσα στο κυρίως μήνυμα.

ΜΗΝΥΜΑ ΧΡΗΣΤΗ:
{raw_user_text}
""".strip()
    return prompt


def compute_support_need_score(
    text: str,
    mood: int,
    sleep: int,
    water: int,
    mode_weights: Optional[dict] = None,
    conversation_summary: str = "",
    open_threads: Optional[list] = None,
) -> tuple:
    t = (text or "").lower()
    reasons: List[str] = []
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

    return clamp_int(score, 0, 100), reasons


def support_need_label(score: int) -> tuple:
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
    decision_trace: List[str],
    lang_instruction: str = "",
    fallback_closing_text: str = "Σε ευχαριστώ που το μοιράστηκες. Κράτα ένα μικρό βήμα φροντίδας για σήμερα και πήγαινε απαλά.",
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
        profile=profile,
    )

    closing = llm_therapeutic_reply(
        mood=mood_value,
        sleep=sleep,
        water=water,
        user_text=wrap_prompt,
        profile=profile,
        active_mode=active_mode,
        lang_instruction=lang_instruction,
    )

    if closing:
        closing = safe_trim(closing, 1500)
        st.session_state.messages.append(("bot", closing))
        remember_bot_output(closing)
        decision_trace.append("Wrap-up: LLM συμπεράσματα/βήματα/κλείσιμο.")
    else:
        rb = fallback_closing_text
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