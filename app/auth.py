"""
auth.py
========
Όλα όσα έχουν να κάνουν με τη σύνδεση στη Supabase και το authentication
(εγγραφή / σύνδεση / αποσύνδεση χρήστη).

Οτιδήποτε άλλο αρχείο του project χρειάζεται να μιλήσει με τη Supabase ή
να μάθει "ποιος είναι ο συνδεδεμένος χρήστης", περνάει από εδώ.
"""

import streamlit as st
from supabase import create_client, Client


def get_supabase_client() -> Client:
    """
    Επιστρέφει μια σύνδεση Supabase.

    ΣΗΜΑΝΤΙΚΟ: ΔΕΝ χρησιμοποιούμε @st.cache_resource εδώ, γιατί αυτό θα
    μοίραζε την ΙΔΙΑ σύνδεση (άρα και το ίδιο auth session) ανάμεσα σε
    ΟΛΟΥΣ τους χρήστες της εφαρμογής ταυτόχρονα. Κάθε χρήστης (κάθε
    browser session) πρέπει να έχει τη ΔΙΚΗ ΤΟΥ σύνδεση, ώστε το login
    του να μην μπερδεύεται με του διπλανού του.
    """
    if "supabase_client" not in st.session_state:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        st.session_state.supabase_client = create_client(url, key)
    return st.session_state.supabase_client


def sign_up_user(email: str, password: str):
    """Δημιουργεί νέο λογαριασμό. Επιστρέφει (result, error_message)."""
    supabase = get_supabase_client()
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return res, None
    except Exception as e:
        return None, str(e)


def sign_in_user(email: str, password: str):
    """Συνδέει υπάρχοντα χρήστη. Επιστρέφει (result, error_message)."""
    supabase = get_supabase_client()
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return res, None
    except Exception as e:
        return None, str(e)


def sign_out_user():
    """Αποσυνδέει τον χρήστη και καθαρίζει το cached client."""
    supabase = get_supabase_client()
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    # Καθαρίζουμε το client από το session_state ώστε το επόμενο login
    # να ξεκινήσει από καθαρό session, χωρίς κατάλοιπα του προηγούμενου.
    st.session_state.pop("supabase_client", None)


def get_user_id():
    """Το ασφαλές, μοναδικό UUID που δίνει η Supabase Auth στον συνδεδεμένο χρήστη."""
    return st.session_state.get("user_id")


def get_user_email():
    """Το email του συνδεδεμένου χρήστη (fallback σε 'anon' αν κάτι λείπει)."""
    email = (st.session_state.get("user_email") or "").strip().lower()
    if not email:
        email = (st.session_state.get("user_name") or "anon").strip().lower()
    return email or "anon"