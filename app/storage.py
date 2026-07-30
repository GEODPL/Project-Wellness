"""
storage.py
==========
Αποθήκευση/ανάκτηση δεδομένων χρήστη στη Supabase (πίνακας user_profiles):
- το "προφίλ φροντίδας" (name, context, στόχοι, κ.λπ.)
- το ιστορικό συνομιλίας (chat_history)

Κάθε γραμμή του πίνακα user_profiles ανήκει σε έναν ΜΟΝΟ χρήστη (id = auth
user id) και προστατεύεται με Row Level Security, οπότε ο κώδικας εδώ δεν
χρειάζεται να ελέγχει ξανά "ποιανού είναι αυτά τα δεδομένα" — το κάνει
ήδη η βάση.
"""

from datetime import datetime
from typing import Any, Dict

import streamlit as st

from auth import get_supabase_client, get_user_id


def load_profile() -> Dict[str, Any]:
    """Φορτώνει το προφίλ του συνδεδεμένου χρήστη (από cache ή από τη Supabase)."""
    user_id = get_user_id()
    if not user_id:
        return {}

    # Ελέγχουμε πρώτα την προσωρινή μνήμη, ώστε να ξεκλειδώνει αμέσως το chat.
    if "current_profile" in st.session_state:
        return st.session_state.current_profile

    supabase = get_supabase_client()
    try:
        response = supabase.table("user_profiles").select("profile").eq("id", user_id).execute()
        if len(response.data) > 0:
            data = response.data[0].get("profile", {}) or {}
            st.session_state.current_profile = data
            return data
        return {}
    except Exception as e:
        st.error(f"⚠️ Σφάλμα φόρτωσης (Supabase): {e}")
        return {}


def save_profile(new_profile: Dict[str, Any]) -> None:
    """Αποθηκεύει το προφίλ του συνδεδεμένου χρήστη."""
    user_id = get_user_id()
    if not user_id:
        return

    st.session_state.current_profile = new_profile

    supabase = get_supabase_client()
    try:
        response = supabase.table("user_profiles").select("id").eq("id", user_id).execute()
        if len(response.data) == 0:
            supabase.table("user_profiles").insert(
                {"id": user_id, "profile": new_profile, "chat_history": {}}
            ).execute()
        else:
            supabase.table("user_profiles").update(
                {"profile": new_profile, "updated_at": datetime.now().isoformat()}
            ).eq("id", user_id).execute()
    except Exception as e:
        st.error(f"⚠️ Σφάλμα αποθήκευσης (Supabase): {e}")


def load_chat_state() -> Dict[str, Any]:
    """Φορτώνει το αποθηκευμένο ιστορικό συνομιλίας του χρήστη."""
    user_id = get_user_id()
    if not user_id:
        return {}

    supabase = get_supabase_client()
    try:
        res = supabase.table("user_profiles").select("chat_history").eq("id", user_id).execute()
        if len(res.data) > 0 and res.data[0].get("chat_history"):
            data = res.data[0]["chat_history"]
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return {"messages": data}
        return {}
    except Exception:
        return {}


def save_chat_state(state: Dict[str, Any]) -> None:
    """Αποθηκεύει το τρέχον ιστορικό συνομιλίας του χρήστη."""
    user_id = get_user_id()
    if not user_id:
        return

    supabase = get_supabase_client()
    try:
        res = supabase.table("user_profiles").select("id").eq("id", user_id).execute()
        if len(res.data) == 0:
            supabase.table("user_profiles").insert(
                {"id": user_id, "profile": {}, "chat_history": state}
            ).execute()
        else:
            supabase.table("user_profiles").update(
                {"chat_history": state, "updated_at": datetime.now().isoformat()}
            ).eq("id", user_id).execute()
    except Exception as e:
        print(f"Chat save error: {e}")