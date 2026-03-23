import os
import json
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_CLIENT: Optional["OpenAI"] = None


# =========================================================
# CLIENT
# =========================================================
def _build_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    if not API_KEY:
        return None
    try:
        return OpenAI(api_key=API_KEY)
    except Exception:
        return None


def _get_client() -> Optional["OpenAI"]:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_client()
    return _CLIENT


def _safe_trim(text: str, max_chars: int = 1200) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


def _format_profile_snippet(profile: Dict[str, Any]) -> str:
    if not profile:
        return ""
    keys = [
        "name",
        "context",
        "age_range",
        "main_goals",
        "main_struggles",
        "helpful_things",
        "preferred_tone",
        "triggers",
        "soothing_things",
    ]
    parts = []
    for k in keys:
        v = (profile.get(k) or "")
        v = str(v).strip()
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


# =========================================================
# MAIN THERAPEUTIC REPLY
# =========================================================
def _build_system_prompt(
    mood: int,
    sleep: int,
    water: int,
    profile: Dict[str, Any],
    active_mode: Optional[str] = None,
) -> str:
    base = (
        "Είσαι ένας ήρεμος, μη κριτικός βοηθός ευεξίας στα ελληνικά.\n"
        "Δεν κάνεις διάγνωση, δεν δίνεις ιατρικές οδηγίες.\n"
        "Δεν χρησιμοποιείς δραματικό ύφος και αποφεύγεις το «πρέπει».\n"
        f"Check-in: διάθεση {mood}/10, ύπνος {sleep}, νερό {water}.\n"
    )

    # ελαφρά mode conditioning (το πραγματικό blending το κάνεις στο user prompt)
    if active_mode == "student":
        base += "Πλαίσιο: φοιτητικό άγχος/σπουδές.\n"
    elif active_mode == "work":
        base += "Πλαίσιο: εργασιακό στρες/όρια.\n"
    elif active_mode == "sleep":
        base += "Πλαίσιο: ύπνος/αποκατάσταση.\n"
    elif active_mode == "relationships":
        base += "Πλαίσιο: σχέσεις/οικογένεια/όρια.\n"

    tone = (profile.get("preferred_tone") or "").strip()
    if tone:
        base += f"Προτιμώμενος τόνος χρήστη: {tone}\n"

    triggers = (profile.get("triggers") or "").strip()
    if triggers:
        base += "Απέφυγε θέματα/λέξεις που ο χρήστης δεν θέλει.\n"

    return base


def llm_therapeutic_reply(
    mood: int,
    sleep: int,
    water: int,
    user_text: str,
    profile: Dict[str, Any],
    active_mode: Optional[str] = None,
) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None

    system_prompt = _build_system_prompt(
        mood=mood,
        sleep=sleep,
        water=water,
        profile=profile,
        active_mode=active_mode,
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.65,
        )
        out = response.choices[0].message.content or ""
        out = out.strip()
        return out if out else None
    except Exception:
        return None


# =========================================================
# FOLLOW-UP FOR EXERCISES (word-choice etc.)
# =========================================================
def llm_exercise_followup(
    chosen_word: str,
    mood_value: int,
    sleep: int,
    water: int,
    last_text: str,
    profile: Dict[str, Any],
    active_mode: Optional[str] = None,
) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None

    prof = _format_profile_snippet(profile)

    system_prompt = (
        "Είσαι υποστηρικτικός βοηθός ευεξίας στα ελληνικά.\n"
        "Δίνεις σύντομο, τρυφερό σχόλιο 1–3 προτάσεων.\n"
        "Χωρίς διάγνωση, χωρίς ιατρικές οδηγίες, χωρίς δραματικό ύφος.\n"
    )

    user_prompt = (
        f"Άσκηση: ο χρήστης διάλεξε λέξη.\n"
        f"Λέξη: {chosen_word}\n"
        f"Check-in: διάθεση {mood_value}/10, ύπνος {sleep}, νερό {water}\n"
        f"Mode: {active_mode or 'NONE'}\n"
        f"Προφίλ: {prof or '(none)'}\n"
        f"Τελευταίο κείμενο χρήστη (πλαίσιο): {last_text}\n\n"
        "Γράψε ένα σύντομο, ζεστό σχόλιο που να δένει με το πλαίσιο."
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.55,
            max_tokens=140,
        )
        out = (response.choices[0].message.content or "").strip()
        return _safe_trim(out, 420) if out else None
    except Exception:
        return None


# =========================================================
# A) NARRATIVE CONTINUITY: UPDATE MEMORY
# =========================================================
def llm_update_memory(
    profile: Dict[str, Any],
    active_mode: str,
    mode_weights: Dict[str, float],
    prev_summary: str,
    prev_threads: List[str],
    prev_facts: List[str],
    user_text: str,
    bot_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Επιστρέφει dict:
      {"summary": str, "threads": list[str], "facts": list[str]}
    Σκοπός: mid-term memory, όχι output προς χρήστη.
    """
    client = _get_client()
    if client is None:
        return None

    prof = _format_profile_snippet(profile)

    # Βάζουμε strict JSON output για να μη σπάει.
    system_prompt = (
        "Είσαι μηχανισμός ενημέρωσης μνήμης συνομιλίας (conversation memory updater).\n"
        "Επιστρέφεις ΜΟΝΟ έγκυρο JSON, χωρίς επιπλέον κείμενο.\n"
        "Δεν κάνεις διάγνωση, δεν δίνεις συμβουλές. Απλώς συμπυκνώνεις/οργανώνεις.\n"
        "Στόχοι:\n"
        "- Rolling summary: 3–6 προτάσεις, ουδέτερο/ήπιο ύφος.\n"
        "- Open threads: 2–6 bullets ως strings (εκκρεμή θέματα/στόχοι/ερωτήματα).\n"
        "- Facts: 0–6 σταθερά facts που φαίνονται αξιόπιστα/διαχρονικά.\n"
        "Κανόνες:\n"
        "- Μην βάζεις ευαίσθητες ιατρικές/κλινικές δηλώσεις.\n"
        "- Μην αποθηκεύεις λεπτομέρειες αυτοτραυματισμού/βίας.\n"
        "- Αν κάτι είναι αβέβαιο, μην το βάζεις στα facts.\n"
    )

    weights_top = sorted(mode_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
    weights_text = ", ".join([f"{k}:{v:.2f}" for k, v in weights_top]) if weights_top else "none"

    user_prompt = (
        f"PROFILE: {prof or '(none)'}\n"
        f"ACTIVE_MODE: {active_mode or 'NONE'}\n"
        f"MODE_WEIGHTS_TOP: {weights_text}\n\n"
        f"PREV_SUMMARY:\n{(prev_summary or '').strip()}\n\n"
        f"PREV_THREADS:\n{json.dumps(prev_threads or [], ensure_ascii=False)}\n\n"
        f"PREV_FACTS:\n{json.dumps(prev_facts or [], ensure_ascii=False)}\n\n"
        f"LAST_USER:\n{user_text}\n\n"
        f"LAST_BOT:\n{bot_text}\n\n"
        "Ενημέρωσε τη μνήμη.\n"
        "Επιστροφή JSON με ακριβώς αυτά τα keys: summary, threads, facts.\n"
        "threads/facts να είναι λίστες από strings.\n"
    )

    try:
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
            max_tokens=420,
        )
        raw = (res.choices[0].message.content or "").strip()
        if not raw:
            return None

        # robust JSON parse (αν επιστρέψει code-fence, το καθαρίζουμε)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.strip("`")
            raw_clean = raw_clean.replace("json", "").strip()

        data = json.loads(raw_clean)

        summary = _safe_trim(str(data.get("summary", "")).strip(), 900)
        threads = data.get("threads", [])
        facts = data.get("facts", [])

        if not isinstance(threads, list):
            threads = []
        if not isinstance(facts, list):
            facts = []

        threads = [str(x).strip() for x in threads if str(x).strip()]
        facts = [str(x).strip() for x in facts if str(x).strip()]

        # μικρά clamps
        threads = threads[:8]
        facts = facts[:8]

        return {"summary": summary, "threads": threads, "facts": facts}
    except Exception:
        return None


# =========================================================
# MICRO PROMPTS (for Exercises buttons)
# =========================================================
_MICRO_FALLBACKS: Dict[str, List[str]] = {
    "support_phrase": [
        "Αν το πας σήμερα «με το ζόρι», είναι ήδη αρκετό. Μια μικρή ανάσα και συνεχίζουμε απαλά.",
        "Δεν χρειάζεται να είσαι τέλειος/η για να αξίζεις φροντίδα. Μόνο άνθρωπος.",
        "Μπορεί να είναι δύσκολο—και ταυτόχρονα, το ότι είσαι εδώ λέει κάτι καλό για σένα.",
        "Πάμε βήμα-βήμα. Το επόμενο μικρό βήμα είναι αρκετό.",
        "Ακόμη κι αν το μυαλό σου βιάζεται, το σώμα σου δικαιούται 30″ ηρεμίας.",
    ],
    "water_nudge": [
        "Αν μπορείς, ένα ποτήρι νερό τώρα. Είναι μικρό reset για το σώμα.",
        "Πιες λίγες γουλιές και πάρε 2 αργές ανάσες. Αυτό μετράει.",
        "Μικρή φροντίδα: νερό + χαλάρωσε τους ώμους για 3″.",
    ],
    "care_reply_game": [
        "Αυτό ακούγεται σαν σκληρή σκέψη. Μπορώ να την κρατήσω λίγο πιο απαλά, χωρίς να την πιστέψω 100%.",
        "Αν το έλεγε φίλος μου, θα του μιλούσα πιο τρυφερά. Μπορώ να το δοκιμάσω κι εγώ.",
        "Δεν χρειάζεται να λυθεί όλο τώρα. Μόνο να μην με χτυπάει από μέσα.",
    ],
}


def _format_chat_tail(chat_tail: List[str]) -> str:
    if not chat_tail:
        return "χωρίς πρόσφατο chat context"
    lines = []
    for i, msg in enumerate(chat_tail[-6:], start=1):
        lines.append(f"{i}. {_safe_trim(msg, 240)}")
    return "\n".join(lines)


def llm_micro_prompt(
    exercise_key: str,
    profile: Dict[str, Any],
    chat_tail: List[str],
    active_mode: str,
    avoid_texts: Optional[List[str]] = None,
) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None

    avoid_texts = avoid_texts or []
    profile_snippet = _format_profile_snippet(profile)

    intent_map = {
        "support_phrase": "δώσε μια μοναδική, ζεστή, σύντομη φράση στήριξης 1–2 προτάσεων",
        "water_nudge": "δώσε μια σύντομη υπενθύμιση ενυδάτωσης 1–2 προτάσεων με ήπιο τόνο",
        "care_reply_game": "δώσε μια φροντιστική απάντηση σε δύσκολη σκέψη 1–3 προτάσεων",
    }
    intent = intent_map.get(exercise_key, "δώσε μια σύντομη, εξατομικευμένη φράση 1–2 προτάσεων")

    system_prompt = (
        "Είσαι υποστηρικτικό σύστημα micro-coaching στα ελληνικά.\n"
        "Παράγεις ΜΟΝΟ 1–3 προτάσεις. Όχι bullets.\n"
        "Δεν κάνεις διάγνωση, δεν δίνεις ιατρικές οδηγίες, αποφεύγεις «πρέπει».\n"
        "Να είναι φυσικό και να πατάει διακριτικά σε 1 στοιχείο από chat ή προφίλ.\n"
    )

    user_prompt = (
        f"Στόχος: {intent}\n"
        f"Mode: {active_mode or 'NONE'}\n"
        f"Προφίλ: {profile_snippet or '(none)'}\n"
        f"Πρόσφατο chat:\n{_format_chat_tail(chat_tail)}\n\n"
        f"ΜΗΝ επαναλάβεις κάτι παρόμοιο με:\n"
        + ("\n".join([f"- {t}" for t in avoid_texts[-6:]]) if avoid_texts else "- (κανένα)\n")
    )

    try:
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.85,
            max_tokens=140,
        )
        out = (res.choices[0].message.content or "").strip()
        return _safe_trim(out, 360) if out else None
    except Exception:
        return None


def micro_prompt_with_fallback(
    exercise_key: str,
    profile: Dict[str, Any],
    chat_tail: List[str],
    active_mode: str,
    avoid_texts: Optional[List[str]] = None,
) -> str:
    avoid_texts = avoid_texts or []
    out = llm_micro_prompt(
        exercise_key=exercise_key,
        profile=profile,
        chat_tail=chat_tail,
        active_mode=active_mode,
        avoid_texts=avoid_texts,
    )
    if out:
        return out

    pool = _MICRO_FALLBACKS.get(exercise_key) or _MICRO_FALLBACKS["support_phrase"]
    idx = (len(chat_tail) + len(avoid_texts)) % len(pool)
    return pool[idx]
