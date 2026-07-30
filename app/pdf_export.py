"""
pdf_export.py
=============
Δημιουργεί ένα σύντομο PDF "Αναφορά Προόδου" -- κάτι που ο χρήστης μπορεί
να κατεβάσει και να μοιραστεί με πραγματικό ψυχολόγο/ψυχίατρο αν θέλει.

ΣΗΜΑΝΤΙΚΗ ΛΕΠΤΟΜΕΡΕΙΑ: Το ReportLab δεν υποστηρίζει ελληνικούς χαρακτήρες
με τις ενσωματωμένες γραμματοσειρές του (Helvetica κ.λπ.) -- θα εμφανίζονταν
κενά/κουτάκια. Γι' αυτό "δανειζόμαστε" τη γραμματοσειρά DejaVu Sans που
κουβαλάει ήδη μέσα του το matplotlib (το οποίο έχεις ήδη ως εξάρτηση),
η οποία έχει πλήρη κάλυψη ελληνικών γλυφών.
"""

import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

_FONT_REGISTERED = False


def _register_greek_font() -> None:
    """Καταχωρεί το DejaVu Sans (από το matplotlib) ώστε το ReportLab να
    μπορεί να τυπώσει σωστά ελληνικούς χαρακτήρες. Ασφαλές να καλείται
    πολλές φορές -- η δεύτερη+ κλήση δεν κάνει τίποτα."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return

    mpl_data = matplotlib.get_data_path()
    regular_path = os.path.join(mpl_data, "fonts", "ttf", "DejaVuSans.ttf")
    bold_path = os.path.join(mpl_data, "fonts", "ttf", "DejaVuSans-Bold.ttf")

    pdfmetrics.registerFont(TTFont("Wellness", regular_path))
    pdfmetrics.registerFont(TTFont("Wellness-Bold", bold_path))
    _FONT_REGISTERED = True


def _build_styles() -> Dict[str, ParagraphStyle]:
    _register_greek_font()
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "WellnessTitle", parent=base["Title"], fontName="Wellness-Bold",
            fontSize=20, textColor=colors.HexColor("#2E2440"), spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "WellnessSubtitle", parent=base["Normal"], fontName="Wellness",
            fontSize=10.5, textColor=colors.HexColor("#6B5C82"), spaceAfter=14,
        ),
        "heading": ParagraphStyle(
            "WellnessHeading", parent=base["Heading2"], fontName="Wellness-Bold",
            fontSize=13.5, textColor=colors.HexColor("#2E2440"),
            spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "WellnessBody", parent=base["Normal"], fontName="Wellness",
            fontSize=10, textColor=colors.HexColor("#2E2440"), leading=14,
        ),
        "footer": ParagraphStyle(
            "WellnessFooter", parent=base["Normal"], fontName="Wellness",
            fontSize=8, textColor=colors.HexColor("#8B7FA0"), leading=11,
        ),
    }


def generate_wellness_pdf(
    user_name: str,
    df_user: pd.DataFrame,
    current_streak: int,
    screening_history: Optional[List[Dict[str, Any]]] = None,
    max_rows: int = 30,
) -> bytes:
    """
    Φτιάχνει το PDF και επιστρέφει τα raw bytes (έτοιμα για st.download_button).

    df_user: DataFrame με στήλες timestamp/mood/sleep/water, ήδη φιλτραρισμένο
             για τον συγκεκριμένο χρήστη (ταξινομημένο, δεν χρειάζεται να
             είναι ήδη περιορισμένο σε αριθμό γραμμών -- το κάνουμε εδώ).
    """
    styles = _build_styles()
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    story = []

    # --- Επικεφαλίδα ---
    story.append(Paragraph("Project Wellness — Αναφορά Προόδου", styles["title"]))
    today_str = datetime.now().strftime("%d/%m/%Y")
    subtitle = f"Χρήστης: {user_name or '—'}  ·  Ημερομηνία εξαγωγής: {today_str}"
    story.append(Paragraph(subtitle, styles["subtitle"]))

    # --- Σύνοψη ---
    df_valid = df_user.copy()
    for col in ["mood", "sleep", "water"]:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    avg_mood = df_valid["mood"].mean() if "mood" in df_valid.columns and not df_valid.empty else None
    avg_sleep = df_valid["sleep"].mean() if "sleep" in df_valid.columns and not df_valid.empty else None
    avg_water = df_valid["water"].mean() if "water" in df_valid.columns and not df_valid.empty else None

    story.append(Paragraph("Σύνοψη", styles["heading"]))
    summary_lines = [f"Τρέχον streak: {current_streak} συνεχόμενες μέρες με check-in."]
    if avg_mood is not None:
        summary_lines.append(f"Μέση διάθεση: {avg_mood:.1f}/10")
    if avg_sleep is not None:
        summary_lines.append(f"Μέσος ύπνος: {avg_sleep:.1f} ώρες")
    if avg_water is not None:
        summary_lines.append(f"Μέση ενυδάτωση: {avg_water:.1f} ποτήρια")
    for line in summary_lines:
        story.append(Paragraph(f"• {line}", styles["body"]))

    story.append(Spacer(1, 10))

    # --- Πίνακας πρόσφατων καταγραφών ---
    story.append(Paragraph("Πρόσφατες Καταγραφές", styles["heading"]))

    if df_valid.empty:
        story.append(Paragraph("Δεν υπάρχουν ακόμη καταγραφές.", styles["body"]))
    else:
        recent = df_valid.tail(max_rows).copy()
        if "timestamp" in recent.columns:
            recent["timestamp"] = pd.to_datetime(recent["timestamp"], errors="coerce")
            recent = recent.sort_values("timestamp", ascending=False)

        table_data = [["Ημερομηνία", "Διάθεση", "Ύπνος", "Νερό"]]
        for _, row in recent.iterrows():
            ts = row.get("timestamp")
            ts_str = ts.strftime("%d/%m/%Y") if pd.notna(ts) else "—"
            mood_v = row.get("mood")
            sleep_v = row.get("sleep")
            water_v = row.get("water")
            table_data.append([
                ts_str,
                f"{mood_v:.0f}/10" if pd.notna(mood_v) else "—",
                f"{sleep_v:.0f}h" if pd.notna(sleep_v) else "—",
                f"{water_v:.0f}" if pd.notna(water_v) else "—",
            ])

        table = Table(table_data, colWidths=[4 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Wellness-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Wellness"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEAFB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2E2440")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#2E2440")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D0EF")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F2FF")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

    # --- Screening αποτελέσματα (PHQ-9 / GAD-7), αν υπάρχουν ---
    if screening_history:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Πρόσφατες Αξιολογήσεις (PHQ-9 / GAD-7)", styles["heading"]))
        for h in screening_history[:6]:
            type_label = "PHQ-9 (κατάθλιψη)" if h.get("screening_type") == "phq9" else "GAD-7 (άγχος)"
            ts = (h.get("created_at") or "")[:10]
            line = f"• {type_label} — {h.get('total_score')} · {h.get('severity_label')} · {ts}"
            story.append(Paragraph(line, styles["body"]))

    # --- Disclaimer ---
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Αυτή η αναφορά προέρχεται από ένα εργαλείο αυτοβοήθειας και ψυχοεκπαιδευτικού "
        "χαρακτήρα (Project Wellness). Δεν αποτελεί κλινική διάγνωση ούτε αντικαθιστά "
        "αξιολόγηση από ειδικό ψυχικής υγείας. Προορίζεται ως υποστηρικτικό υλικό για "
        "συζήτηση με επαγγελματία, αν το επιθυμεί ο χρήστης.",
        styles["footer"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()