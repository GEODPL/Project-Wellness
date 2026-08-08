"""
screening.py
============
PHQ-9 (κατάθλιψη) και GAD-7 (άγχος): δύο σύντομα, δημόσια διαθέσιμα και
ευρέως επικυρωμένα ερωτηματολόγια αυτο-αναφοράς που χρησιμοποιούνται στην
κλινική πράξη ως εργαλεία "screening" (ένδειξη σοβαρότητας συμπτωμάτων).

ΔΕΝ αποτελούν διάγνωση — ούτε εδώ, ούτε στην πραγματική κλινική χρήση.
Είναι ένα δομημένο, επιστημονικά τεκμηριωμένο σημείο εκκίνησης, σε αντίθεση
με το απλό keyword-matching heuristic (compute_support_need_score) που
βασίζεται σε ελεύθερο κείμενο.

Πηγές (public domain, ελεύθερη χρήση για κλινικούς/εκπαιδευτικούς σκοπούς):
- PHQ-9: Kroenke K, Spitzer RL, Williams JB. "The PHQ-9: validity of a
  brief depression severity measure." J Gen Intern Med. 2001.
- GAD-7: Spitzer RL, Kroenke K, Williams JB, Löwe B. "A brief measure for
  assessing generalized anxiety disorder: the GAD-7." Arch Intern Med. 2006.

Οι ερωτήσεις παρακάτω είναι δική μας απόδοση στα ελληνικά του κλινικού
νοήματος κάθε ερώτησης (όχι αντιγραφή συγκεκριμένης δημοσιευμένης μετάφρασης).
"""

from typing import Any, Dict, List, Optional, Tuple

from auth import get_supabase_client, get_user_id


# Οι 4 τυπικές επιλογές απάντησης (0-3), όπως στο επίσημο PHQ-9/GAD-7.
ANSWER_OPTIONS: List[Tuple[int, str]] = [
    (0, "Καθόλου"),
    (1, "Αρκετές μέρες"),
    (2, "Περισσότερες από τις μισές μέρες"),
    (3, "Σχεδόν κάθε μέρα"),
]

SCREENING_INTRO = "Τις τελευταίες 2 εβδομάδες, πόσο συχνά σε έχουν ενοχλήσει τα παρακάτω;"

PHQ9_QUESTIONS: List[str] = [
    "Μειωμένο ενδιαφέρον ή απόλαυση σε δραστηριότητες που συνήθως σου άρεσαν",
    "Αίσθημα μελαγχολίας, κατάθλιψης, ή απελπισίας",
    "Δυσκολία στον ύπνο — είτε να κοιμηθείς, είτε να μείνεις κοιμισμένος/η, είτε υπερβολικός ύπνος",
    "Αίσθημα κούρασης ή έλλειψης ενέργειας",
    "Μειωμένη ή αυξημένη όρεξη",
    "Αρνητικά συναισθήματα για τον εαυτό σου — ότι απέτυχες, ή ότι απογοήτευσες τον εαυτό σου ή την οικογένειά σου",
    "Δυσκολία συγκέντρωσης, π.χ. στο διάβασμα ή στην τηλεόραση",
    "Αισθητά αργή κίνηση ή ομιλία που πρόσεξαν κι άλλοι — ή το αντίθετο, τόση ανησυχία που κινείσαι πολύ περισσότερο απ' ό,τι συνήθως",
    "Σκέψεις ότι θα ήταν καλύτερα να μην ζούσες, ή σκέψεις να βλάψεις τον εαυτό σου με κάποιον τρόπο",
]

GAD7_QUESTIONS: List[str] = [
    "Αίσθημα νευρικότητας, άγχους, ή τάσης",
    "Δυσκολία να σταματήσεις ή να ελέγξεις την ανησυχία",
    "Υπερβολική ανησυχία για διάφορα πράγματα",
    "Δυσκολία στη χαλάρωση",
    "Τόση ανησυχία που δυσκολεύεσαι να καθίσεις ήσυχος/η",
    "Εύκολη ενόχληση ή ευερεθιστότητα",
    "Αίσθημα φόβου, σαν να πρόκειται να συμβεί κάτι τρομερό",
]

# Η ερώτηση (0-indexed, δηλαδή η 9η) του PHQ-9 που αφορά σκέψεις
# αυτοτραυματισμού/θανάτου. ΠΟΤΕ δεν αντιμετωπίζεται απλά ως "πόντοι".
PHQ9_SELF_HARM_INDEX = 8


def score_phq9(answers: List[int]) -> Tuple[int, str, bool]:
    """
    Επιστρέφει (συνολικό σκορ 0-27, ετικέτα σοβαρότητας, flagged_item9).
    flagged_item9 = True αν η απάντηση στην ερώτηση αυτοτραυματισμού είναι > 0
    (δηλαδή ο χρήστης απάντησε κάτι διαφορετικό από "Καθόλου").
    """
    from i18n import t as _t

    total = sum(answers)
    flagged = len(answers) > PHQ9_SELF_HARM_INDEX and answers[PHQ9_SELF_HARM_INDEX] > 0

    if total <= 4:
        label = _t("severity_minimal")
    elif total <= 9:
        label = _t("severity_mild")
    elif total <= 14:
        label = _t("severity_moderate")
    elif total <= 19:
        label = _t("severity_mod_severe")
    else:
        label = _t("severity_severe")

    return total, label, flagged


def score_gad7(answers: List[int]) -> Tuple[int, str]:
    """Επιστρέφει (συνολικό σκορ 0-21, ετικέτα σοβαρότητας)."""
    from i18n import t as _t

    total = sum(answers)
    if total <= 4:
        label = _t("severity_minimal")
    elif total <= 9:
        label = _t("severity_mild")
    elif total <= 14:
        label = _t("severity_moderate")
    else:
        label = _t("severity_severe")
    return total, label


def save_screening_result(
    screening_type: str,
    total_score: int,
    severity_label: str,
    answers: List[int],
    flagged_item9: bool = False,
) -> None:
    """Αποθηκεύει το αποτέλεσμα στη Supabase (πίνακας screening_results)."""
    user_id = get_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    try:
        supabase.table("screening_results").insert(
            {
                "user_id": user_id,
                "screening_type": screening_type,
                "total_score": total_score,
                "severity_label": severity_label,
                "answers": answers,
                "flagged_item9": flagged_item9,
            }
        ).execute()
    except Exception as e:
        print(f"Screening save error: {e}")


def load_screening_history(screening_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Φορτώνει τις πιο πρόσφατες αξιολογήσεις του χρήστη (πιο πρόσφατη πρώτη)."""
    user_id = get_user_id()
    if not user_id:
        return []
    supabase = get_supabase_client()
    try:
        q = supabase.table("screening_results").select("*").eq("user_id", user_id)
        if screening_type:
            q = q.eq("screening_type", screening_type)
        res = q.order("created_at", desc=True).limit(limit).execute()
        return res.data or []
    except Exception:
        return []