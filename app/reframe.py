"""
reframe.py
==========
RAG (Retrieval-Augmented Generation) για το Thought Reframe Studio.

Πώς δουλεύει, βήμα-βήμα:
1. Η σκέψη του χρήστη μετατρέπεται σε "embedding" (ένα vector 384 αριθμών),
   τοπικά και δωρεάν, μέσω sentence-transformers.
2. Ψάχνουμε στη Supabase (pgvector) ποιες τεχνικές από την κοινή βιβλιοθήκη
   γνώσης (reframe_knowledge) μοιάζουν σημασιολογικά περισσότερο με τη
   σκέψη — ΚΑΙ ποιες παλιότερες σκέψεις/reframes του ΙΔΙΟΥ χρήστη μοιάζουν
   επίσης (user_reframes). Αυτό είναι το "Retrieval" (R) του RAG.
3. Δίνουμε αυτά τα ευρήματα σαν context στο LLM, το οποίο γράφει ένα
   φρέσκο, εξατομικευμένο reframe βασισμένο σε πραγματικές τεχνικές —
   αυτό είναι το "Generation" (G) του RAG.
4. Αποθηκεύουμε το νέο reframe (με το δικό του embedding) στο προσωπικό
   ιστορικό, ώστε την επόμενη φορά να μπορεί να ανακτηθεί κι αυτό.
"""

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer

from auth import get_supabase_client, get_user_id
from llm import llm_reframe_reply


@st.cache_resource
def _get_embedding_model() -> SentenceTransformer:
    """
    Φορτώνει το μοντέλο embeddings ΜΙΑ φορά και το κρατάει στη μνήμη.

    Εδώ το @st.cache_resource είναι η ΣΩΣΤΗ χρήση του (σε αντίθεση με τη
    σύνδεση Supabase στο auth.py, όπου το αποφύγαμε επίτηδες): το μοντέλο
    δεν "ξέρει" ποιος είναι συνδεδεμένος, δεν κουβαλάει προσωπικά δεδομένα
    κανενός χρήστη — είναι ένα στατικό εργαλείο μετατροπής κειμένου σε
    αριθμούς, ίδιο και ασφαλές να μοιράζεται από όλους. Το να το μοιράζονται
    όλοι γλιτώνει μνήμη και χρόνο φόρτωσης (αλλιώς θα φόρτωνε ξανά τα
    ~470MB σε κάθε rerun).
    """
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def embed_text(text: str) -> List[float]:
    """Μετατρέπει ένα κείμενο σε vector 384 αριθμών."""
    model = _get_embedding_model()
    return model.encode(text).tolist()


def retrieve_similar_techniques(embedding: List[float], match_count: int = 3) -> List[Dict[str, Any]]:
    """Οι πιο σχετικές τεχνικές από την κοινή βιβλιοθήκη γνώσης (reframe_knowledge)."""
    supabase = get_supabase_client()
    try:
        res = supabase.rpc(
            "match_reframe_knowledge",
            {"query_embedding": embedding, "match_count": match_count},
        ).execute()
        return res.data or []
    except Exception:
        return []


def retrieve_similar_personal(embedding: List[float], match_count: int = 2) -> List[Dict[str, Any]]:
    """Οι πιο σχετικές ΠΡΟΗΓΟΥΜΕΝΕΣ σκέψεις/reframes του ΙΔΙΟΥ χρήστη (user_reframes)."""
    user_id = get_user_id()
    if not user_id:
        return []
    supabase = get_supabase_client()
    try:
        res = supabase.rpc(
            "match_user_reframes",
            {"query_embedding": embedding, "match_user_id": user_id, "match_count": match_count},
        ).execute()
        return res.data or []
    except Exception:
        return []


def save_user_reframe(thought: str, pattern: str, reframe: str) -> None:
    """Αποθηκεύει το νέο reframe στο προσωπικό ιστορικό του χρήστη (με το δικό του embedding)."""
    user_id = get_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    try:
        embedding = embed_text(thought)
        supabase.table("user_reframes").insert(
            {
                "user_id": user_id,
                "thought": thought,
                "pattern": pattern,
                "reframe": reframe,
                "embedding": embedding,
            }
        ).execute()
    except Exception as e:
        print(f"Reframe save error: {e}")


def _rule_based_fallback(thought: str, pattern: str) -> str:
    """
    Το ΠΑΛΙΟ, απλό rule-based reframe -- χρησιμοποιείται ΜΟΝΟ ως δίχτυ
    ασφαλείας αν το LLM δεν απαντήσει (π.χ. πρόβλημα δικτύου/κλειδιού),
    ώστε ο χρήστης να μην μένει ποτέ χωρίς καμία απάντηση.
    """
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


def generate_reframe_rag(
    thought: str,
    pattern: str,
    intensity: int,
    profile: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Η νέα, RAG-based εκδοχή του reframe.

    Επιστρέφει (reframe_text, retrieved_techniques, retrieved_personal) --
    τα δύο τελευταία είναι χρήσιμα αν θες να δείξεις στο UI *ποιες πηγές*
    χρησιμοποιήθηκαν (explainability, στο ίδιο πνεύμα με το explainable
    layer που έχεις ήδη στο Chat tab).
    """
    t = (thought or "").strip()
    if not t:
        return "", [], []

    embedding = embed_text(t)
    techniques = retrieve_similar_techniques(embedding, match_count=3)
    personal = retrieve_similar_personal(embedding, match_count=2)

    reframe = llm_reframe_reply(
        thought=t,
        pattern=pattern,
        intensity=intensity,
        profile=profile,
        techniques=techniques,
        personal_history=personal,
    )

    used_fallback = False
    if not reframe:
        reframe = _rule_based_fallback(t, pattern)
        used_fallback = True

    # Αποθηκεύουμε στο προσωπικό ιστορικό (ακόμη κι αν χρησιμοποιήθηκε fallback,
    # έτσι ώστε την επόμενη φορά να υπάρχει έστω κάτι για προσωπική ανάκτηση).
    save_user_reframe(t, pattern, reframe)

    if used_fallback:
        techniques = []
        personal = []

    return reframe, techniques, personal