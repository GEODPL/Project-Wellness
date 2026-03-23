# app/user_profile.py
import os
import json

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "user_profile.json")

DEFAULT_PROFILE = {
    "name": "",
    "age_range": "",
    "context": "",        # π.χ. "Φοιτητής Φιλολογίας, ΕΚΠΑ"
    "main_goals": "",     # ελεύθερο κείμενο
    "main_struggles": "",
    "helpful_things": "", # τι σε βοηθά συνήθως
}

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PROFILE.copy()

def save_profile(profile: dict):
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
