"""
icons.py
========
Custom, minimal line-art εικονίδια "καιρού διάθεσης" -- το signature
οπτικό στοιχείο της ταυτότητας "Η Ώρα του Λυκόφωτος". Αντικαθιστούν τα
γενικά emoji (☀️⛅🌧️) με συνεπές, σχεδιασμένο ύφος (ίδιο πάχος γραμμής,
στρογγυλεμένες άκρες) και χρώματα από την επίσημη παλέτα του app.

Χρωματική λογική (καμία τυχαία επιλογή):
- rain (χαμηλή διάθεση)   -> twilight-rose  (ζεστό, όχι ανησυχητικό κόκκινο)
- cloud (μέτρια διάθεση)  -> twilight-ember (ήπια ενέργεια)
- sun (υψηλή διάθεση)     -> twilight-sage  (ηρεμία/γείωση -- "να νιώθεις
  καλά" εδώ σημαίνει ισορροπία, όχι έξαψη)
"""

from typing import Tuple

TWILIGHT_ROSE = "#F2A6B8"
TWILIGHT_EMBER = "#E8916B"
TWILIGHT_SAGE = "#8FAF8C"


def _svg_rain(color: str) -> str:
    return f"""<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M7 13.5a3.8 3.8 0 0 1 .4-7.58A5.4 5.4 0 0 1 18 7.2a4.3 4.3 0 0 1-.6 8.3H7Z" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M8.5 18.5L7.3 20.7M12.5 18.5L11.3 20.7M16.5 18.5L15.3 20.7" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
</svg>"""


def _svg_partly_cloudy(color: str) -> str:
    return f"""<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="16" cy="7.5" r="3.2" stroke="{color}" stroke-width="1.8"/>
<path d="M16 2.7V1.5M20.5 4.5l.9-.9M21.7 7.5H23" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
<path d="M9.5 16.5a3.5 3.5 0 0 1 .3-6.98 5 5 0 0 1 8.1 0 4 4 0 0 1-.4 6.98H9.5Z" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def _svg_sun(color: str) -> str:
    return f"""<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="12" r="5" stroke="{color}" stroke-width="1.8"/>
<path d="M12 1.5V4M12 20v2.5M22.5 12H20M4 12H1.5M19.07 4.93L17.25 6.75M6.75 17.25L4.93 19.07M19.07 19.07L17.25 17.25M6.75 6.75L4.93 4.93" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
</svg>"""


def streak_flame_svg(color: str = TWILIGHT_EMBER, size: int = 26) -> str:
    """Μικρό line-art εικονίδιο 'φλόγας' για το streak -- ενέργεια/συνέχεια."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2.5c.4 3-2.6 4.2-2.6 7.2a2.6 2.6 0 0 0 5.2 0c0-1-.4-1.6-.8-2.2 1.8.6 3 2.6 3 4.8a4.8 4.8 0 0 1-9.6 0c0-4.4 3.6-5.6 4.8-9.8Z"
  stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def mood_weather_svg(mood_val: float) -> Tuple[str, str]:
    """
    Επιστρέφει (svg_markup, hex_color) ανάλογα με την τιμή διάθεσης (0-10).
    Ίδια κατηγοριοποίηση με πριν (≤4 / 5-7 / ≥8), απλά με σχεδιασμένα
    εικονίδια αντί για emoji.
    """
    if mood_val <= 4:
        return _svg_rain(TWILIGHT_ROSE), TWILIGHT_ROSE
    elif mood_val <= 7:
        return _svg_partly_cloudy(TWILIGHT_EMBER), TWILIGHT_EMBER
    else:
        return _svg_sun(TWILIGHT_SAGE), TWILIGHT_SAGE