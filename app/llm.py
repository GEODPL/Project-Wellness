import os
import json
import streamlit as st 
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  
 
try:
    API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
except Exception:
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
        "Είσαι ένας έμπειρος ψυχολόγος που βασίζει τις παρεμβάσεις του αυστηρά σε Τεκμηριωμένες Πρακτικές (Evidence-Based Practices). "
        "Ενσωματώνεις αρμονικά στοιχεία από: Γνωσιακή Συμπεριφορική Θεραπεία (CBT), Θεραπεία Αποδοχής και Δέσμευσης (ACT), "
        "Κινητοποιητική Συνέντευξη (Motivational Interviewing - MI) και Βραχεία Θεραπεία Εστιασμένη στη Λύση (SFBT). "
        "Ο στόχος σου είναι η γνωστική αναδόμηση, η συναισθηματική αποδοχή και η ενίσχυση του εσωτερικού κινήτρου του χρήστη.\n\n"
        
        "ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ (Αν τους παραβείς το σύστημα θα καταρρεύσει):\n"
        "1. ΜΗΔΕΝΙΚΑ ΚΛΙΣΕ & ΨΕΥΤΙΚΗ ΕΝΣΥΝΑΙΣΘΗΣΗ: Απαγορεύονται φράσεις όπως «Σε καταλαβαίνω», «Ακούγεται πολύ δύσκολο», "
        "«Είναι λογικό να νιώθεις έτσι». Η ενσυναίσθησή σου πρέπει να φαίνεται μέσα από την ενεργητική ακρόαση και τις οξυδερκείς ερωτήσεις σου, όχι από τυποποιημένες δηλώσεις.\n"
        "2. ΚΑΘΡΕΦΤΙΣΜΑ & ΕΞΑΙΡΕΣΕΙΣ (SFBT/CBT): Αντί να αναλύεις το πρόβλημα, καθρέφτισέ το και ψάξε για εξαιρέσεις. (π.χ. «Παρατηρώ ότι νιώθεις πλήρη εξάντληση. Υπήρξε κάποια στιγμή σήμερα που αυτό το συναίσθημα ήταν έστω και 1% πιο ήπιο;»).\n"
        "3. ΕΣΩΤΕΡΙΚΟ ΚΙΝΗΤΡΟ (MI): Μη δίνεις έτοιμες λύσεις. Αν ο χρήστης δείχνει αναβλητικότητα ή αντίσταση, ρώτησέ τον: «Αν αποφάσιζες να κάνεις ένα πολύ μικρό βήμα, ποιος θα ήταν ο δικός σου λόγος για να το δοκιμάσεις;».\n"
        "4. ΡΙΖΙΚΗ ΑΠΟΔΟΧΗ (ACT/DBT): Αν το συναίσθημα είναι πολύ έντονο (π.χ. πανικός, βαθιά θλίψη), μην προσπαθήσεις να το 'φτιάξεις'. Πρότεινε χώρο: «Αντί να παλέψουμε αυτή τη σκέψη τώρα, πώς θα ήταν απλώς να την παρατηρήσουμε να υπάρχει;»\n"
        "5. ΛΑΚΩΝΙΚΟΤΗΤΑ: Μίλα αυστηρά με 3 έως 4 προτάσεις το πολύ. Είσαι διάλογος, όχι άρθρο.\n"
        "6. Η ΜΑΙΕΥΤΙΚΗ ΕΡΩΤΗΣΗ: Κλείνε ΠΑΝΤΑ την απάντησή σου με ΜΙΑ (1) ανοιχτή ερώτηση που καλεί τον χρήστη σε ενδοσκόπηση.\n\n"
        
        f"Check-in: διάθεση {mood}/10, ύπνος {sleep}, νερό {water}.\n"
    )

    if active_mode == "student":
        base += "Πλαίσιο: Φοιτητικό άγχος / Burnout. Χρησιμοποίησε CBT για να αποσυνδέσεις την προσωπική αξία από την ακαδημαϊκή επίδοση. Εστίασε στον φόβο της κριτικής.\n"
    elif active_mode == "work":
        base += "Πλαίσιο: Εργασιακό στρες. Χρησιμοποίησε ACT για να βοηθήσεις στην αναγνώριση των αξιών (values) σε αντίθεση με την απλή διεκπεραίωση υποχρεώσεων, και SFBT για να θέσετε ρεαλιστικά όρια.\n"
    elif active_mode == "sleep":
        base += "Πλαίσιο: Αϋπνία / Υπερένταση. Χρησιμοποίησε αρχές CBT-I (για την αϋπνία). Αποενοχοποίησε την αγρυπνία, μειώνοντας την πίεση του «πρέπει να κοιμηθώ».\n"
    elif active_mode == "relationships":
        base += "Πλαίσιο: Διαπροσωπικές σχέσεις. Χρησιμοποίησε στοιχεία διαλεκτικής (DBT) για να επικυρώσεις (validate) το συναίσθημά του, ενώ ταυτόχρονα τον καλείς να δει την οπτική γωνία ή τα όρια της άλλης πλευράς.\n"

    tone = (profile.get("preferred_tone") or "").strip()
    if tone:
        base += f"Οδηγία ύφους: {tone}\n"

    triggers = (profile.get("triggers") or "").strip()
    if triggers:
        base += f"Απαγορευμένα θέματα (Triggers): {triggers}\n"

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
# FOLLOW-UP FOR EXERCISES 
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
# =========================================================
# ΣΥΝΔΥΑΣΜΕΝΗ ΚΛΗΣΗ: REPLY + MEMORY UPDATE ΣΕ 1 API CALL
# =========================================================
def llm_therapeutic_reply_with_memory(
    mood: int,
    sleep: int,
    water: int,
    user_text: str,
    profile: Dict[str, Any],
    active_mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Κάνει σε ΜΙΑ κλήση OpenAI ό,τι πριν γινόταν σε 2 ξεχωριστές
    (llm_therapeutic_reply + llm_update_memory) -- μισό κόστος και μισό
    latency ανά κανονικό μήνυμα του χρήστη, χωρίς αλλαγή συμπεριφοράς.

    Το `user_text` περιμένεται να είναι το ήδη-χτισμένο, εμπλουτισμένο
    prompt (από το build_llm_user_prompt), που ήδη περιέχει μέσα του την
    προηγούμενη rolling summary / open threads / facts memory -- γι' αυτό
    δεν χρειάζεται να τα περάσουμε ξανά ως ξεχωριστές παραμέτρους εδώ.

    Επιστρέφει dict {"reply", "summary", "threads", "facts"} ή None αν
    κάτι αποτύχει (τότε ο καλών κάνει fallback στο απλό, rule-based reply).
    """
    client = _get_client()
    if client is None:
        return None

    base_prompt = _build_system_prompt(
        mood=mood, sleep=sleep, water=water, profile=profile, active_mode=active_mode
    )

    memory_instructions = (
        "\n\nΕΠΙΠΛΕΟΝ ΔΟΥΛΕΙΑ -- ΕΝΗΜΕΡΩΣΗ ΜΝΗΜΗΣ ΣΥΖΗΤΗΣΗΣ:\n"
        "Μέσα στο μήνυμα του χρήστη θα βρεις την ΠΡΟΗΓΟΥΜΕΝΗ rolling summary, "
        "open threads, και facts memory (κάτω από τις αντίστοιχες επικεφαλίδες). "
        "Μαζί με την κυρίως απάντησή σου, επίστρεψε και μια ΕΝΗΜΕΡΩΜΕΝΗ εκδοχή τους:\n"
        "- summary: 3-6 προτάσεις, ουδέτερο/ήπιο ύφος (ενημέρωσε την προηγούμενη, μη γράψεις από την αρχή)\n"
        "- threads: 2-6 εκκρεμή θέματα/στόχοι/ερωτήματα, ως λίστα strings\n"
        "- facts: 0-6 σταθερά, αξιόπιστα facts, ως λίστα strings\n"
        "Μην βάζεις ευαίσθητες κλινικές δηλώσεις ή λεπτομέρειες αυτοτραυματισμού/βίας στη μνήμη.\n\n"
        "ΜΟΡΦΗ ΕΞΟΔΟΥ -- ΚΡΙΣΙΜΟ: Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ με ένα έγκυρο JSON object "
        "(χωρίς markdown code fences, χωρίς κείμενο πριν/μετά), με ΑΚΡΙΒΩΣ αυτά τα keys:\n"
        '{"reply": "<η κύρια απάντηση προς τον χρήστη, στα ελληνικά, ακολουθώντας όλους τους παραπάνω κανόνες>", '
        '"summary": "<ενημερωμένη σύνοψη>", "threads": ["..."], "facts": ["..."]}'
    )

    system_prompt = base_prompt + memory_instructions

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.65,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None

        data = json.loads(raw)

        reply = str(data.get("reply", "")).strip()
        if not reply:
            return None

        summary = _safe_trim(str(data.get("summary", "")).strip(), 900)
        threads = data.get("threads", [])
        facts = data.get("facts", [])
        if not isinstance(threads, list):
            threads = []
        if not isinstance(facts, list):
            facts = []
        threads = [str(x).strip() for x in threads if str(x).strip()][:8]
        facts = [str(x).strip() for x in facts if str(x).strip()][:8]

        return {"reply": reply, "summary": summary, "threads": threads, "facts": facts}
    except Exception:
        return None


# =========================================================
# B) THOUGHT REFRAME (RAG-based)
# =========================================================
def llm_reframe_reply(
    thought: str,
    pattern: str,
    intensity: int,
    profile: Dict[str, Any],
    techniques: List[Dict[str, Any]],
    personal_history: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Παράγει ένα εξατομικευμένο reframe, βασισμένο σε:
    - τη σκέψη + το μοτίβο που επέλεξε ο χρήστης
    - τεχνικές που ανακτήθηκαν (RAG) από την κοινή βιβλιοθήκη γνώσης
    - τυχόν παρόμοιες προηγούμενες σκέψεις/reframes του ΙΔΙΟΥ χρήστη

    Αν δεν υπάρχει διαθέσιμο LLM key, επιστρέφει None (το reframe.py κάνει
    τότε fallback στο απλό rule-based reframe, ώστε ο χρήστης να παίρνει
    πάντα κάποια απάντηση).
    """
    client = _get_client()
    if client is None:
        return None

    prof = _format_profile_snippet(profile)

    techniques_text = (
        "\n".join(f"- {t.get('content', '')}" for t in techniques)
        if techniques
        else "- (καμία ιδιαίτερα σχετική τεχνική βρέθηκε — χρησιμοποίησε τη γενική σου γνώση CBT/ACT)"
    )

    personal_text = (
        "\n".join(
            f"- Παλιότερη σκέψη: «{p.get('thought', '')}» → Reframe που δοκιμάστηκε τότε: «{p.get('reframe', '')}»"
            for p in personal_history
        )
        if personal_history
        else "- (καμία σχετική προηγούμενη καταγραφή αυτού του χρήστη)"
    )

    system_prompt = (
        "Είσαι υποστηρικτικό σύστημα Thought Reframe, βασισμένο σε αρχές CBT/ACT, στα ελληνικά.\n"
        "Δουλειά σου: να βοηθήσεις τον χρήστη να δει μια δύσκολη σκέψη από μια πιο τρυφερή, "
        "ρεαλιστική οπτική — ΧΩΡΙΣ να ακυρώσεις ή να υποτιμήσεις το συναίσθημά του.\n\n"
        "ΚΑΝΟΝΕΣ:\n"
        "- 3 έως 5 προτάσεις το πολύ.\n"
        "- Χρησιμοποίησε ΤΟΥΛΑΧΙΣΤΟΝ μία από τις 'ΣΧΕΤΙΚΕΣ ΤΕΧΝΙΚΕΣ' παρακάτω, προσαρμοσμένη "
        "στη συγκεκριμένη σκέψη του χρήστη — μην την αντιγράψεις αυτολεξεί, ενσωμάτωσέ τη φυσικά.\n"
        "- Αν υπάρχει σχετικό 'ΠΡΟΣΩΠΙΚΟ ΙΣΤΟΡΙΚΟ', μπορείς να αναφερθείς διακριτικά σε αυτό "
        "(π.χ. 'όπως και τότε που...'), αλλά ΜΟΝΟ αν πραγματικά ταιριάζει με τη νέα σκέψη — "
        "διαφορετικά αγνόησέ το.\n"
        "- Όχι διάγνωση, όχι bullets· φυσική, ζεστή ροή λόγου.\n"
        "- Μπορείς να κλείσεις με μία εναλλακτική φράση σε εισαγωγικά, που θα μπορούσε να πει "
        "ο χρήστης στον εαυτό του.\n"
    )

    user_prompt = (
        f"Δύσκολη σκέψη: {thought}\n"
        f"Μοτίβο σκέψης: {pattern}\n"
        f"Ένταση (1-10): {intensity}\n"
        f"Προφίλ χρήστη: {prof or '(none)'}\n\n"
        f"ΣΧΕΤΙΚΕΣ ΤΕΧΝΙΚΕΣ (χρησιμοποίησε τουλάχιστον 1):\n{techniques_text}\n\n"
        f"ΠΡΟΣΩΠΙΚΟ ΙΣΤΟΡΙΚΟ ΧΡΗΣΤΗ:\n{personal_text}\n"
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=280,
        )
        out = (response.choices[0].message.content or "").strip()
        return _safe_trim(out, 700) if out else None
    except Exception:
        return None


# =========================================================
# MICRO PROMPTS 
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