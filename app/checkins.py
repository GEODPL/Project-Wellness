"""
checkins.py
===========
Καταγραφή check-ins (mood/sleep/water/μήνυμα) στη Supabase, αντί για το
παλιό, μη προστατευμένο CSV αρχείο (user_data.csv). Κάθε γραμμή "ανήκει"
σε έναν συγκεκριμένο χρήστη και προστατεύεται με RLS, ακριβώς όπως και
τα υπόλοιπα δεδομένα της εφαρμογής (profile, chat_history, screening_results).
"""

from typing import Optional

import pandas as pd

from auth import get_supabase_client, get_user_id


def log_checkin(
    mood: Optional[int],
    sleep: Optional[int],
    water: Optional[int],
    text: str,
    is_emergency: bool = False,
) -> None:
    """Καταγράφει ένα check-in. mood/sleep/water μπορούν να είναι None
    (π.χ. όταν το check-in προέρχεται από άλλο σημείο εκτός Chat tab,
    όπως το Screening, όπου δεν υπάρχουν αυτές οι τιμές διαθέσιμες)."""
    user_id = get_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    try:
        supabase.table("checkins").insert(
            {
                "user_id": user_id,
                "mood": mood,
                "sleep": sleep,
                "water": water,
                "message": text,
                "is_emergency": is_emergency,
            }
        ).execute()
    except Exception as e:
        print(f"Checkin log error: {e}")


def load_checkins_df(limit: int = 500) -> pd.DataFrame:
    """
    Επιστρέφει DataFrame με στήλες: timestamp, mood, sleep, water, message
    -- ταξινομημένο χρονολογικά (παλιότερο -> πιο πρόσφατο). Περιέχει ΜΟΝΟ
    τα check-ins του συνδεδεμένου χρήστη (το εγγυάται το Row Level Security
    στη βάση, δεν χρειάζεται πλέον χειροκίνητο φιλτράρισμα με email).
    """
    empty = pd.DataFrame(columns=["timestamp", "mood", "sleep", "water", "message"])

    user_id = get_user_id()
    if not user_id:
        return empty

    supabase = get_supabase_client()
    try:
        res = (
            supabase.table("checkins")
            .select("mood, sleep, water, message, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = res.data or []
    except Exception:
        rows = []

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    df = df.rename(columns={"created_at": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    return df[["timestamp", "mood", "sleep", "water", "message"]]