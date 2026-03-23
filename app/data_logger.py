import os
import pandas as pd
from datetime import datetime

# Αρχείο όπου καταγράφονται τα δεδομένα
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "user_data.csv")


def log_user_data(mood, sleep, water, message, email=None):
    """
    Καταγράφει ένα νέο check-in στο user_data.csv
    Τώρα υποστηρίζει και email ώστε να ξεχωρίζουν οι χρήστες.

    Columns: timestamp, email, mood, sleep, water, message
    """
    # φτιάχνουμε το νέο row
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "email": email or "",
        "mood": mood,
        "sleep": sleep,
        "water": water,
        "message": message,
    }

    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            # Αν κάτι πάει στραβά με το υπάρχον CSV, το ξαναγράφουμε από την αρχή
            df = pd.DataFrame(columns=["timestamp", "email", "mood", "sleep", "water", "message"])
    else:
        df = pd.DataFrame(columns=["timestamp", "email", "mood", "sleep", "water", "message"])

    # αν λείπει κάποια στήλη (παλιά version), τη συμπληρώνουμε
    for col in ["timestamp", "email", "mood", "sleep", "water", "message"]:
        if col not in df.columns:
            df[col] = ""

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
