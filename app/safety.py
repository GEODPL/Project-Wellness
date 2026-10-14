"""
safety.py
=========
Κοινός έλεγχος κινδύνου (emergency gate) για ΚΑΘΕ σημείο της εφαρμογής
όπου ο χρήστης γράφει ελεύθερο κείμενο που θα σταλεί στο LLM.

Πριν: ο έλεγχος is_emergency υπήρχε μόνο στο Chat tab. Το Thought Reframe
Studio και οι ασκήσεις ελεύθερου κειμένου (αποφόρτιση, 3 στηρίγματα)
έστελναν το κείμενο απευθείας στο LLM, χωρίς guardrail.

Χρήση (στο app.py):
    if guard_free_text(text, "reframe"):
        pass  # εμφανίστηκε το emergency block, ΔΕΝ γίνεται κλήση στο LLM
    else:
        ... κανονική ροή ...
"""

import streamlit as st

from checkins import log_checkin
from components import render_emergency_block
from rules import emergency_message, is_emergency


def guard_free_text(text: str, source: str) -> bool:
    """
    Επιστρέφει True αν εντοπιστεί ένδειξη κινδύνου. Σε αυτή την περίπτωση
    εμφανίζει το emergency block και καταγράφει ΜΟΝΟ την πηγή (όχι το
    κείμενο του χρήστη), όπως ήδη γίνεται για το PHQ-9 item 9.
    """
    if not is_emergency(text or ""):
        return False

    render_emergency_block(emergency_message())
    log_checkin(None, None, None, f"[{source}] emergency keyword detected", is_emergency=True)
    return True
