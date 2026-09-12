import os
import re
import html
import socket
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
import streamlit.components.v1 as components

import streamlit as st
from auth import get_supabase_client, sign_up_user, sign_in_user, sign_out_user, get_user_id, get_user_email
from storage import load_profile, save_profile, load_chat_state, save_chat_state
from modes import MODES, detect_sleep_difficulty, detect_mode_weights, choose_active_mode_from_weights, format_mode_blend_text
from prompts import (
    safe_trim, remember_bot_output, build_llm_user_prompt,
    compute_support_need_score, support_need_label, run_wrapup_bundle,
)
from reframe import generate_reframe_rag
from screening import (
    ANSWER_OPTIONS, SCREENING_INTRO, PHQ9_QUESTIONS, GAD7_QUESTIONS,
    score_phq9, score_gad7, save_screening_result, load_screening_history,
)
from icons import mood_weather_svg, streak_flame_svg
from pdf_export import generate_wellness_pdf
import pandas as pd
import qrcode
from checkins import log_checkin, load_checkins_df
from i18n import (
    t, get_lang, set_lang, lang_directive,
    SCREENING_INTRO_EN, ANSWER_OPTIONS_EN, PHQ9_QUESTIONS_EN, GAD7_QUESTIONS_EN,
)
from PIL import Image

from llm import llm_exercise_followup, llm_therapeutic_reply_with_memory
from rules import (
    personal_reply,
    fallback_therapeutic_reply,
    exercise_suggestion,
    is_emergency,
    emergency_message,
)
from emotional_map import render_emotional_map
from components import (
    render_message,
    render_exercise_card,
    render_emergency_block,
    render_action_plan_card,
)
import json
import hashlib
from datetime import datetime

# --- ΣΥΝΔΕΣΗ ΜΕ ΤΗ ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ---
# ΣΗΜΑΝΤΙΚΟ: ΔΕΝ χρησιμοποιούμε πλέον @st.cache_resource εδώ, γιατί αυτό θα
# μοιράζει την ΙΔΙΑ σύνδεση (άρα και το ίδιο auth session) ανάμεσα σε
# ΟΛΟΥΣ τους χρήστες της εφαρμογής ταυτόχρονα. Κάθε χρήστης (κάθε browser
# session) πρέπει να έχει τη ΔΙΚΗ ΤΟΥ σύνδεση, ώστε το login του να μην
# μπερδεύεται με του διπλανού του.

supabase = get_supabase_client()


# ============================================================
# AUTH HELPERS (Supabase Auth)
# ============================================================


#============================================================
# ΠΡΟΣΩΠΙΚΟ ΠΡΟΦΙΛ ΧΡΗΣΤΗ
# ===========================================================


# ============================================================
# CSS
# ============================================================

def load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================
# MOBILE URL + QR
# ============================================================

def get_local_url(port: int = 8501) -> str:
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    return f"http://{local_ip}:{port}"


def generate_qr_image(url: str) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        box_size=4,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    if hasattr(img, "get_image"):
        img = img.get_image()
    return img


# ============================================================
# NARRATIVE CONTINUITY + MODE BLENDING + EXPLAINABLE LAYER
# ============================================================


def compute_current_streak(dates: list) -> int:
    """
    Υπολογίζει το τρέχον streak: πόσες συνεχόμενες ημερολογιακές μέρες
    (μέχρι σήμερα ή μέχρι χθες) υπάρχει τουλάχιστον 1 check-in.
    Αν η πιο πρόσφατη καταγραφή είναι πάνω από 1 μέρα πίσω, το streak
    θεωρείται "σπασμένο" (0), ακριβώς όπως στα γνωστά streak trackers.
    """
    if not dates:
        return 0
    unique_days = sorted({d.date() for d in dates}, reverse=True)
    if not unique_days:
        return 0

    today = datetime.now().date()
    if (today - unique_days[0]).days > 1:
        return 0

    streak = 1
    for i in range(1, len(unique_days)):
        if (unique_days[i - 1] - unique_days[i]).days == 1:
            streak += 1
        else:
            break
    return streak


def infer_age_range(age: Any) -> str:
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return ""
    if age_int <= 0:
        return ""
    if 18 <= age_int <= 24:
        return "18–24"
    if 25 <= age_int <= 34:
        return "25–34"
    if 35 <= age_int <= 44:
        return "35–44"
    return "45+"


def user_says_feels_ok(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        "ειμαι καλα", "είμαι καλά",
        "ειμαι οκ", "είμαι οκ",
        "ειμαι μια χαρα", "είμαι μια χαρά",
        "ολα καλα", "όλα καλά",
        "μια χαρα ειμαι", "μια χαρά είμαι",
    ]
    return any(p in t for p in patterns)


def looks_a_bit_strained(mood: int, sleep: int, water: int) -> bool:
    try:
        m = int(mood)
        s = int(sleep)
        w = int(water)
    except (TypeError, ValueError):
        return False
    return (m <= 6) or (s <= 5) or (w <= 4)


from llm import micro_prompt_with_fallback

def get_chat_tail_from_state(max_n: int = 6) -> list[str]:
    tail = []
    for sender, content in st.session_state.get("messages", []):
        if sender == "user":
            tail.append(str(content or "").strip())
    return [t for t in tail if t][-max_n:]


# ============================================================
# MODES (Contextual Coaches)
# ============================================================


# ============================================================
# MODE-SPECIFIC PLANS 
# ============================================================


# ============================================================
# THOUGHT REFRAME (Rule-based) !Still!
# ============================================================


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Project Wellness",
    page_icon="🧠",
    layout="wide",
)
load_css()


# ============================================================
# SESSION STATE INIT
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "messages" not in st.session_state:
    st.session_state.messages = []
if "exercise_followup" not in st.session_state:
    st.session_state.exercise_followup = False
if "last_decision_trace" not in st.session_state:
    st.session_state.last_decision_trace = []
if "last_checkin" not in st.session_state:
    st.session_state.last_checkin = {}

# Conversation coherence 
if "conversation_phase" not in st.session_state:
    st.session_state.conversation_phase = "dialogue"  

if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

if "checkin_ctx" not in st.session_state:
    st.session_state.checkin_ctx = None

if "pending_reflection" not in st.session_state:
    st.session_state.pending_reflection = False

if "active_mode" not in st.session_state:
    st.session_state.active_mode = None

if "study_anxiety_count" not in st.session_state:
    st.session_state.study_anxiety_count = 0
if "study_anxiety_plan_given" not in st.session_state:
    st.session_state.study_anxiety_plan_given = False

if "sleep_plan_count" not in st.session_state:
    st.session_state.sleep_plan_count = 0
if "sleep_plan_given" not in st.session_state:
    st.session_state.sleep_plan_given = False

if "reframes_history" not in st.session_state:
    st.session_state.reframes_history = []

if "active_mode" not in st.session_state:
    st.session_state.active_mode = None
if "mode_plan_given" not in st.session_state:
    st.session_state.mode_plan_given = {}

if "microcoach_history" not in st.session_state:
    st.session_state.microcoach_history = {}  

# ============================================================
# NARRATIVE CONTINUITY STATE
# ============================================================
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = "" 

if "open_threads" not in st.session_state:
    st.session_state.open_threads = []  

if "facts_memory" not in st.session_state:
    st.session_state.facts_memory = []  

if "last_bot_outputs" not in st.session_state:
    st.session_state.last_bot_outputs = []  

if "last_user_messages" not in st.session_state:
    st.session_state.last_user_messages = []  


# ============================================================
# LOGIN GATE (Supabase Auth: Σύνδεση / Εγγραφή)
# ============================================================

if not st.session_state.logged_in:
    _LOGIN_BRAND_MARK = """<svg width="64" height="64" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="36" cy="34" r="14" stroke="white" stroke-width="2"/>
<path d="M36 14V20M36 48V54M16 34H10M62 34H56M22.4 20.4L18.2 16.2M53.8 16.2L49.6 20.4" stroke="white" stroke-width="2" stroke-linecap="round"/>
<path d="M6 54H66" stroke="white" stroke-width="2.4" stroke-linecap="round" opacity="0.85"/>
</svg>"""

    st.markdown(
        """
        <style>
        .login-hero{
            height: 100%; min-height: 420px;
            border-radius: 26px;
            padding: 2.4rem 2rem;
            display:flex; flex-direction:column; justify-content:center; align-items:flex-start;
            background: linear-gradient(160deg, var(--twilight-lilac) 0%, var(--twilight-rose) 55%, var(--twilight-ember) 130%);
            box-shadow: var(--shadow-soft);
            animation: cardFadeIn var(--transition-med);
        }
        .login-hero h1{
            font-family: var(--font-display); font-size: 2.1rem; font-weight: 600;
            margin: 1rem 0 0.6rem 0; color: #ffffff;
        }
        .login-hero p{
            font-size: 0.98rem; line-height: 1.6; color: rgba(255,255,255,0.92);
            max-width: 320px; margin: 0;
        }
        .login-form-card{
            height: 100%; min-height: 420px;
            display:flex; flex-direction:column; justify-content:center;
        }
        .login-form-card h2{
            font-family: var(--font-display); margin-top: 0; margin-bottom: 0.3rem;
            color: var(--twilight-deep);
        }
        .login-form-card .login-sub{
            color: var(--text-soft); font-size: 0.92rem; margin-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Επιλογέας γλώσσας -- εδώ δεν υπάρχει ακόμα profile, οπότε κρατάμε
    # την επιλογή μόνο στο session_state (δεν χάνεται ώσπου να κάνει
    # login/signup ο χρήστης, οπότε πλέον αποθηκεύεται μόνιμα στο profile).
    _pre_lang = get_lang()
    _pre_lang_choice = st.selectbox(
        t("language_label"),
        options=["el", "en"],
        format_func=lambda x: "Ελληνικά" if x == "el" else "English",
        index=0 if _pre_lang == "el" else 1,
        key="pre_login_lang_selector",
        label_visibility="collapsed",
    )
    if _pre_lang_choice != _pre_lang:
        set_lang(_pre_lang_choice)
        st.rerun()

    col_hero, col_form = st.columns([1, 1.15], gap="medium")

    with col_hero:
        st.markdown(
            f"""
            <div class="login-hero">
              {_LOGIN_BRAND_MARK}
              <h1>{t("login_hero_title")}</h1>
              <p>{t("login_hero_desc")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_form:
        st.markdown('<div class="chat-card login-form-card">', unsafe_allow_html=True)
        st.markdown(f"<h2>{t('login_form_title')}</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='login-sub'>{t('login_form_sub')}</p>",
            unsafe_allow_html=True,
        )

        auth_tab_login, auth_tab_signup = st.tabs([t("login_tab_login"), t("login_tab_signup")])

        # --------------------------------------------------------
        # ΣΥΝΔΕΣΗ
        # --------------------------------------------------------
        with auth_tab_login:
            login_email = st.text_input(t("field_email"), key="login_email")
            login_password = st.text_input(t("field_password"), type="password", key="login_password")

            if st.button(t("btn_login"), key="btn_login", use_container_width=True):
                email = (login_email or "").strip().lower()
                password = login_password or ""

                if not email or not password:
                    st.error(t("err_fill_both"))
                else:
                    res, err = sign_in_user(email, password)
                    if err:
                        st.error(t("err_login_failed", err=err))
                    elif res and res.user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = res.user.id
                        st.session_state.user_email = res.user.email

                        # Προσπαθούμε να ανακτήσουμε το αποθηκευμένο όνομα από το προφίλ
                        fallback_name = (res.user.email or "").split("@")[0]
                        try:
                            prof_res = (
                                supabase.table("user_profiles")
                                .select("profile")
                                .eq("id", res.user.id)
                                .execute()
                            )
                            if prof_res.data and (prof_res.data[0].get("profile") or {}).get("name"):
                                st.session_state.user_name = prof_res.data[0]["profile"]["name"]
                            else:
                                st.session_state.user_name = fallback_name
                        except Exception:
                            st.session_state.user_name = fallback_name

                        st.success(t("success_login"))
                        st.rerun()

        # --------------------------------------------------------
        # ΕΓΓΡΑΦΗ
        # --------------------------------------------------------
        with auth_tab_signup:
            signup_name = st.text_input(t("field_username"), key="signup_name")
            signup_email = st.text_input(t("field_email"), key="signup_email")
            signup_password = st.text_input(
                t("field_password_min"), type="password", key="signup_password"
            )
            signup_password2 = st.text_input(
                t("field_password_confirm"), type="password", key="signup_password2"
            )

            if st.button(t("btn_signup"), key="btn_signup", use_container_width=True):
                name = (signup_name or "").strip()
                email = (signup_email or "").strip().lower()
                password = signup_password or ""
                _signup_lang = get_lang()

                if not name:
                    st.error(t("err_need_name"))
                elif not email or "@" not in email:
                    st.error(t("err_need_email"))
                elif len(password) < 6:
                    st.error(t("err_password_length"))
                elif password != signup_password2:
                    st.error(t("err_password_mismatch"))
                else:
                    res, err = sign_up_user(email, password)
                    if err:
                        st.error(t("err_signup_failed", err=err))
                    elif res and res.user:
                        # Αν το "Confirm email" είναι ενεργό στη Supabase, η εγγραφή
                        # δημιουργεί τον χρήστη αλλά ΔΕΝ δίνει ενεργό session μέχρι
                        # να πατήσει το link επιβεβαίωσης στο email του --
                        # res.session θα είναι None σε αυτή την περίπτωση.
                        if res.session is None:
                            st.info(t("signup_confirm_email", email=email))
                        else:
                            st.session_state.logged_in = True
                            st.session_state.user_id = res.user.id
                            st.session_state.user_email = res.user.email
                            st.session_state.user_name = name

                            # Αποθηκεύουμε αμέσως το όνομα (+ επιλεγμένη γλώσσα) στο
                            # προφίλ, ώστε να υπάρχουν έτοιμα την επόμενη φορά.
                            try:
                                supabase.table("user_profiles").insert(
                                    {
                                        "id": res.user.id,
                                        "profile": {"name": name, "language": _signup_lang},
                                        "chat_history": {},
                                    }
                                ).execute()
                            except Exception:
                                pass

                            st.success(t("success_signup", name=name))
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ============================================================
# ONBOARDING (εμφανίζεται ΜΙΑ φορά, μετά το πρώτο login/signup)
# ============================================================

_onboarding_profile = load_profile()

if not _onboarding_profile.get("onboarding_completed"):
    _greet_name = st.session_state.get("user_name", "")
    _ONBOARD_BRAND_MARK = """<svg width="52" height="52" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="36" cy="34" r="14" stroke="white" stroke-width="2"/>
<path d="M36 14V20M36 48V54M16 34H10M62 34H56M22.4 20.4L18.2 16.2M53.8 16.2L49.6 20.4" stroke="white" stroke-width="2" stroke-linecap="round"/>
<path d="M6 54H66" stroke="white" stroke-width="2.4" stroke-linecap="round" opacity="0.85"/>
</svg>"""

    st.markdown(
        f"""
        <div class="login-hero" style="min-height: 220px; margin-top: 0.6rem;">
          {_ONBOARD_BRAND_MARK}
          <h1 style="font-size: 1.7rem; margin-top: 0.7rem;">
            {t("onboarding_greeting", name_part=f", {_greet_name}" if _greet_name else "")}
          </h1>
          <p style="max-width: 480px;">
            {t("onboarding_sub")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-wrapper"><div class="chat-card">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <h4 style="margin-top:0;">{t("onboarding_tabs_heading")}</h4>
        <ul style="line-height:1.9; color: var(--text-main);">
          <li>{t("onboarding_tab_chat")}</li>
          <li>{t("onboarding_tab_history")}</li>
          <li>{t("onboarding_tab_stats")}</li>
          <li>{t("onboarding_tab_exercises")}</li>
          <li>{t("onboarding_tab_screening")}</li>
          <li>{t("onboarding_tab_reframe")}</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="emergency-block" style="margin-top: 0.8rem;">
          {t("onboarding_safety")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_ob1, col_ob2, col_ob3 = st.columns([1, 1.4, 1])
    with col_ob2:
        if st.button(t("onboarding_button"), key="btn_onboarding_done", use_container_width=True):
            _onboarding_profile["onboarding_completed"] = True
            save_profile(_onboarding_profile)
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# APP HEADER + TABS
# ============================================================

user_name = st.session_state.get("user_name", "")
subtitle = (
    t("app_subtitle_greeting", name=user_name)
    if user_name
    else t("app_subtitle_default")
)

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-title-left">{t("app_title")}</div>
      <div class="app-subtitle-right">{subtitle}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Λογαριασμός / Γλώσσα / Αποσύνδεση ---
with st.sidebar:
    st.markdown(f"**👤 {st.session_state.get('user_name','')}**")
    st.caption(st.session_state.get("user_email", ""))

    _current_lang = get_lang()
    _lang_choice = st.selectbox(
        t("language_label"),
        options=["el", "en"],
        format_func=lambda x: "Ελληνικά" if x == "el" else "English",
        index=0 if _current_lang == "el" else 1,
        key="lang_selector",
    )
    if _lang_choice != _current_lang:
        set_lang(_lang_choice)
        _lang_profile = load_profile()
        _lang_profile["language"] = _lang_choice
        save_profile(_lang_profile)
        st.rerun()

    if st.button(t("logout_button"), use_container_width=True):
        sign_out_user()
        for key in ["logged_in", "user_id", "user_email", "user_name", "current_profile"]:
            st.session_state.pop(key, None)
        st.rerun()

tabs = st.tabs(
    [
        t("tab_chat"),
        t("tab_history"),
        t("tab_stats"),
        t("tab_exercises"),
        t("tab_screening"),
        t("tab_reframe"),
        t("tab_profile"),
        t("tab_mobile"),
        t("tab_info"),
    ]
)

(
    tab_chat,
    tab_history,
    tab_stats,
    tab_ex,
    tab_screening,
    tab_reframe,
    tab_profile,
    tab_mobile,
    tab_info,
) = tabs

# ============================================================
# CHAT TAB 
# ============================================================
   
with tab_chat:

    profile = load_profile()

    # --- βασικό προφίλ ---
    has_basic_profile = bool(
        (profile.get("name") or "").strip() or (profile.get("context") or "").strip()
    )

    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("chat_page_title")}</h1>
          <p class='page-subtitle'>
            {t("chat_page_sub")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_basic_profile:
        st.markdown(
            f"""
            <div class="main-wrapper">
              <div class="chat-card">
                <h4>{t("chat_locked_title")}</h4>
                <p style="font-size:1.05rem; font-weight:600; color:#4A3D73;">
                  {t("chat_locked_body1")}
                </p>
                <p style="color:#6C5A9E;">
                  {t("chat_locked_body2")}
                </p>
                <ul style="color:#6C5A9E;">
                  <li>{t("chat_locked_li1")}</li>
                  <li>{t("chat_locked_li2")}</li>
                </ul>
                <p style="margin-top:10px; color:#6C5A9E;">
                  {t("chat_locked_body3")}
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:


        # ============================================================
        # Core helpers
        # ============================================================


        def _detect_dialogue_closure(text: str) -> bool:
            t = (text or "").strip().lower()
            if not t:
                return False
            closures = [
                "ευχαριστώ", "ευχαριστω", "οκ", "ok", "εντάξει", "ενταξει",
                "τέλος", "τελος", "τα λέμε", "τα λεμε", "καληνύχτα", "καληνυχτα",
                "αυτά", "αυτα", "νομίζω οκ", "νομιζω οκ", "bye", "αντίο", "αντιο"
            ]
            shortish = len(t) <= 35
            return shortish and any(c in t for c in closures)


        # ============================================================
        # Session state init + load persisted
        # ============================================================
        if "chat_loaded_for_user" not in st.session_state:
            st.session_state.chat_loaded_for_user = {}

        ukey = get_user_id()
        if not st.session_state.chat_loaded_for_user.get(ukey, False):
            persisted = load_chat_state()

            st.session_state.messages = persisted.get("messages", st.session_state.get("messages", [])) or []
            st.session_state.dialogue_active = persisted.get("dialogue_active", False)
            st.session_state.dialogue_turns = int(persisted.get("dialogue_turns", 0) or 0)
            st.session_state.wrapup_done = persisted.get("wrapup_done", False)

            st.session_state.conversation_summary = persisted.get("conversation_summary", "") or ""
            st.session_state.open_threads = persisted.get("open_threads", []) or []
            st.session_state.facts_memory = persisted.get("facts_memory", []) or []

            st.session_state.last_bot_outputs = persisted.get("last_bot_outputs", []) or []
            st.session_state.last_checkin = persisted.get("last_checkin", {}) or {}
            st.session_state.exercise_followup = persisted.get("exercise_followup", False)

            st.session_state.support_indicator_history = persisted.get("support_indicator_history", []) or []

            st.session_state.chat_loaded_for_user[ukey] = True

        # Defaults if missing
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "exercise_followup" not in st.session_state:
            st.session_state.exercise_followup = False
        if "last_decision_trace" not in st.session_state:
            st.session_state.last_decision_trace = []
        if "last_checkin" not in st.session_state:
            st.session_state.last_checkin = {}
        if "dialogue_active" not in st.session_state:
            st.session_state.dialogue_active = False
        if "dialogue_turns" not in st.session_state:
            st.session_state.dialogue_turns = 0
        if "wrapup_done" not in st.session_state:
            st.session_state.wrapup_done = False

        if "conversation_summary" not in st.session_state:
            st.session_state.conversation_summary = ""
        if "open_threads" not in st.session_state:
            st.session_state.open_threads = []
        if "facts_memory" not in st.session_state:
            st.session_state.facts_memory = []
        if "last_bot_outputs" not in st.session_state:
            st.session_state.last_bot_outputs = []

        if "support_indicator_history" not in st.session_state:
            st.session_state.support_indicator_history = []

        # ============================================================
        # UI
        # ============================================================
        st.markdown(
            """
            <style>
            .checkin-metrics-row{display:flex; gap:14px; flex-wrap:wrap; margin: 8px 0 14px 0;}
            .checkin-metric{
                width: 118px; height: 118px; border-radius: 999px;
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                border: 1px solid transparent;
                gap: 2px;
            }
            .checkin-metric.metric-mood{
                background: rgba(242, 166, 184, 0.16);
                border-color: rgba(242, 166, 184, 0.32);
                box-shadow: 0 6px 20px rgba(242, 166, 184, 0.10);
            }
            .checkin-metric.metric-sleep{
                background: rgba(183, 157, 242, 0.16);
                border-color: rgba(183, 157, 242, 0.32);
                box-shadow: 0 6px 20px rgba(183, 157, 242, 0.10);
            }
            .checkin-metric.metric-water{
                background: rgba(143, 175, 140, 0.16);
                border-color: rgba(143, 175, 140, 0.32);
                box-shadow: 0 6px 20px rgba(143, 175, 140, 0.10);
            }
            .checkin-metric .v{font-family: var(--font-mono); font-size: 24px; font-weight: 700; color: var(--twilight-deep); line-height: 1;}
            .checkin-metric .t{font-size: 12.5px; font-weight:700; color: var(--text-soft); margin-top:2px;}
            .checkin-metric .p{font-family: var(--font-mono); font-size: 11.5px; color: var(--text-soft); margin-top:1px;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="main-wrapper"><div class="chat-card">', unsafe_allow_html=True)

        col_title, col_btn = st.columns([3, 1])
        with col_title:
            st.markdown(f"#### {t('chat_how_are_you')}")
        with col_btn:
            if st.button(t("new_conversation"), use_container_width=True):
                # Καθαρισμός προσωρινής μνήμης
                st.session_state.messages = []
                st.session_state.dialogue_active = False
                st.session_state.dialogue_turns = 0
                st.session_state.wrapup_done = False
                st.session_state.conversation_summary = ""
                st.session_state.open_threads = []
                st.session_state.facts_memory = []
                st.session_state.last_bot_outputs = []
                st.session_state.last_checkin = {}
                st.session_state.exercise_followup = False
                st.session_state.last_decision_trace = []

                # Καθαρισμός αποθηκευμένης μνήμης χρήστη
                save_chat_state({
                    "messages": [],
                    "dialogue_active": False,
                    "dialogue_turns": 0,
                    "wrapup_done": False,
                    "conversation_summary": "",
                    "open_threads": [],
                    "facts_memory": [],
                    "last_bot_outputs": [],
                    "last_checkin": {},
                    "exercise_followup": False,
                    "support_indicator_history": st.session_state.get("support_indicator_history", [])
                })
                
                st.rerun()

        # ΜΕΤΑΦΟΡΑ SLIDERS ΣΕ EXPANDER
        with st.expander(t("checkin_expander"), expanded=True):
            colA, colB, colC = st.columns(3)
            with colA:
                mood_value = st.slider(t("mood_label"), 1, 10, 5, 1, key="chat_mood")
            with colB:
                sleep = st.slider(t("sleep_label"), 1, 10, 7, 1, key="chat_sleep")
            with colC:
                water = st.slider(t("water_label"), 1, 15, 6, 1, key="chat_water")

            mood_pct = int(round((mood_value / 10.0) * 100))
            sleep_pct = int(round((sleep / 10.0) * 100))
            water_pct = int(round((water / 15.0) * 100))

            mood_icon_svg, _mood_icon_color = mood_weather_svg(float(mood_value))

            st.markdown(
                f"""
                <div class="checkin-metrics-row">
                  <div class="checkin-metric metric-mood">
                    {mood_icon_svg}
                    <div class="v">{mood_value}/10</div>
                    <div class="t">{t("mood_metric")}</div>
                  </div>
                  <div class="checkin-metric metric-sleep">
                    <div class="v">{sleep}h</div>
                    <div class="t">{t("sleep_metric")}</div>
                    <div class="p">{sleep_pct}%</div>
                  </div>
                  <div class="checkin-metric metric-water">
                    <div class="v">{water}</div>
                    <div class="t">{t("water_metric")}</div>
                    <div class="p">{water_pct}%</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # RENDER CHAT HISTORY ΕΔΩ ΠΑΝΩ
        for sender, content in st.session_state.messages:
            if sender == "user":
                render_message("user", content)
            elif sender == "bot":
                render_message("bot", content)
            elif sender == "exercise":
                render_exercise_card(content)
            elif sender == "map":
                st.markdown(f"<div class='emotional-map-card'>{content}</div>", unsafe_allow_html=True)
            elif sender == "emergency":
                render_emergency_block(content)
            elif sender == "plan":
                render_action_plan_card(content)
            elif sender == "support_hint":
                render_message("bot", content)

        if st.session_state.get("last_decision_trace"):
            with st.expander(t("decision_trace_label")):
                for item in st.session_state.last_decision_trace:
                    st.markdown(f"- {item}")


        # ST.CHAT_INPUT ΜΟΝΙΜΑ ΣΤΟ ΚΑΤΩ ΜΕΡΟΣ
        user_text = st.chat_input(t("chat_input_placeholder"))

        # ============================================================
        # SEND
        # ============================================================
        if user_text:
            text = (user_text or "").strip()
            if not text:
                st.warning("Γράψε κάτι μικρό πριν πατήσεις αποστολή ")
            else:
                decision_trace: list[str] = []
                decision_trace.append(f"Check-in: mood={mood_value}/10 ({mood_pct}%), sleep={sleep} ({sleep_pct}%), water={water} ({water_pct}%).")

                # 1) Emergency gate 
                if is_emergency(text):
                    decision_trace.append("Emergency gate: ενεργοποίηση ειδικού μηνύματος.")
                    emergency_html = emergency_message()
                    st.session_state.messages.append(("emergency", emergency_html))

                    log_checkin(mood_value, sleep, water, text, is_emergency=True)

                    st.session_state.last_decision_trace = decision_trace

                    # persist
                    save_chat_state({
                        "messages": st.session_state.messages,
                        "dialogue_active": st.session_state.dialogue_active,
                        "dialogue_turns": st.session_state.dialogue_turns,
                        "wrapup_done": st.session_state.wrapup_done,
                        "conversation_summary": st.session_state.conversation_summary,
                        "open_threads": st.session_state.open_threads,
                        "facts_memory": st.session_state.facts_memory,
                        "last_bot_outputs": st.session_state.last_bot_outputs,
                        "last_checkin": st.session_state.last_checkin,
                        "exercise_followup": st.session_state.exercise_followup,
                        "support_indicator_history": st.session_state.support_indicator_history,
                    })
                    st.rerun()

                # 2) Append user message
                st.session_state.messages.append(("user", text))

                # 3) Mode blending
                mode_weights = detect_mode_weights(text, mood_value, sleep, water)
                active_mode = choose_active_mode_from_weights(mode_weights)
                decision_trace.append(f"Mode blending: {format_mode_blend_text(mode_weights)}.")
                decision_trace.append(f"Active mode: {active_mode or 'NONE'}.")

                # 4) Follow-up 
                if st.session_state.get("exercise_followup", False):
                    decision_trace.append("Exercise follow-up: λέξη.")
                    ctx = st.session_state.get("last_checkin", {})

                    followup = llm_exercise_followup(
                        chosen_word=text,
                        mood_value=ctx.get("mood", mood_value),
                        sleep=ctx.get("sleep", sleep),
                        water=ctx.get("water", water),
                        last_text=ctx.get("text", ""),
                        profile=profile,
                        active_mode=active_mode,
                        lang_instruction=lang_directive(),
                    )
                    if followup:
                        followup = safe_trim(followup, 900)
                        st.session_state.messages.append(("bot", followup))
                        remember_bot_output(followup)
                        decision_trace.append("Exercise follow-up: LLM reply.")
                    else:
                        rb = "Σε ακούω. Θες να μου πεις λίγα παραπάνω για το τι σημαίνει αυτή η λέξη για σένα;"
                        st.session_state.messages.append(("bot", rb))
                        remember_bot_output(rb)
                        decision_trace.append("Exercise follow-up: fallback (LLM None).")

                    st.session_state.exercise_followup = False
                    st.session_state.last_decision_trace = decision_trace

                    save_chat_state({
                        "messages": st.session_state.messages,
                        "dialogue_active": st.session_state.dialogue_active,
                        "dialogue_turns": st.session_state.dialogue_turns,
                        "wrapup_done": st.session_state.wrapup_done,
                        "conversation_summary": st.session_state.conversation_summary,
                        "open_threads": st.session_state.open_threads,
                        "facts_memory": st.session_state.facts_memory,
                        "last_bot_outputs": st.session_state.last_bot_outputs,
                        "last_checkin": st.session_state.last_checkin,
                        "exercise_followup": st.session_state.exercise_followup,
                        "support_indicator_history": st.session_state.support_indicator_history,
                    })
                    st.rerun()

                # 5) Rule-based opening 
                if (not st.session_state.dialogue_active) or (st.session_state.dialogue_turns == 0):
                    st.session_state.dialogue_active = True
                    st.session_state.wrapup_done = False
                    st.session_state.dialogue_turns = 0

                    if user_says_feels_ok(text) and looks_a_bit_strained(mood_value, sleep, water):
                        gentle = (
                            "Σημειώνω ότι λες πως είσαι καλά, και αυτό μετράει.\n\n"
                            f"Ταυτόχρονα (διάθεση {mood_value}/10, ύπνος {sleep}, νερό {water}) μοιάζει να υπάρχει λίγη κούραση από κάτω. "
                            "Ας το πάμε ήπια."
                        )
                        st.session_state.messages.append(("bot", gentle))
                        remember_bot_output(gentle)
                        decision_trace.append("Open: gentle mismatch note (ok + strained).")

                # 6) LLM dialogue reply 
                prompt = build_llm_user_prompt(
                    raw_user_text=text,
                    mood_value=mood_value,
                    sleep=sleep,
                    water=water,
                    active_mode=active_mode,
                    mode_weights=mode_weights,
                    profile=profile,
                )

                combined = llm_therapeutic_reply_with_memory(
                    mood=mood_value,
                    sleep=sleep,
                    water=water,
                    user_text=prompt,
                    profile=profile,
                    active_mode=active_mode,
                    lang_instruction=lang_directive(),
                )

                if combined:
                    llm_out = safe_trim(combined["reply"], 1400)
                    st.session_state.messages.append(("bot", llm_out))
                    remember_bot_output(llm_out)
                    decision_trace.append("Dialogue: LLM reply (συνδυασμένη κλήση: reply+memory σε 1 API call).")

                    if combined.get("summary"):
                        st.session_state.conversation_summary = combined["summary"]
                    if combined.get("threads"):
                        st.session_state.open_threads = combined["threads"]
                    if combined.get("facts"):
                        st.session_state.facts_memory = combined["facts"]
                    decision_trace.append("Narrative memory: ενημερώθηκαν summary/threads/facts (ίδια κλήση).")
                else:
                    rb = fallback_therapeutic_reply(mood_value, sleep, water, text)
                    st.session_state.messages.append(("bot", rb))
                    remember_bot_output(rb)
                    decision_trace.append("Dialogue: fallback rule-based (LLM None).")

                st.session_state.dialogue_turns += 1

                # 7) Discreet “support” indicator 
                score, score_reasons = compute_support_need_score(
                    text=text,
                    mood=mood_value,
                    sleep=sleep,
                    water=water,
                    mode_weights=mode_weights,
                    conversation_summary=st.session_state.get("conversation_summary", ""),
                    open_threads=st.session_state.get("open_threads", []),
                )
                label, suggestion = support_need_label(score)

                if score >= 35:
                    hint = (
                        f"**Δείκτης προτεινόμενης υποστήριξης:** {score}/100 · *{label}*\n\n"
                        f"{suggestion}"
                    )
                    st.session_state.messages.append(("support_hint", hint))
                    decision_trace.append(f"Support indicator: {score}/100 ({label}).")
                    if score_reasons:
                        decision_trace.append("Support reasons: " + ", ".join(score_reasons[:4]) + ("…" if len(score_reasons) > 4 else ""))
                else:
                    decision_trace.append(f"Support indicator: {score}/100 (no display; below threshold).")

                st.session_state.support_indicator_history.append(
                    {"score": score, "label": label, "mood": mood_value, "sleep": sleep, "water": water, "ts": datetime.now().isoformat(timespec="seconds")}
                )
                st.session_state.support_indicator_history = st.session_state.support_indicator_history[-30:]

                # 8) 
                closing_now = _detect_dialogue_closure(text)
                if closing_now and not st.session_state.wrapup_done:
                    decision_trace.append("Closure detected: run wrap-up bundle.")
                    run_wrapup_bundle(
                        mood_value=mood_value,
                        sleep=sleep,
                        water=water,
                        last_user_text=text,
                        profile=profile,
                        active_mode=active_mode,
                        mode_weights=mode_weights,
                        decision_trace=decision_trace,
                        lang_instruction=lang_directive(),
                    )
                    st.session_state.wrapup_done = True

                # 9) Logging
                log_checkin(mood_value, sleep, water, text, is_emergency=False)
                decision_trace.append("Logging: saved to Supabase (checkins).")

                # 10) last_checkin
                st.session_state.last_checkin = {
                    "mood": mood_value,
                    "sleep": sleep,
                    "water": water,
                    "text": text,
                    "active_mode": active_mode,
                    "mode_weights": mode_weights,
                }

                st.session_state.last_decision_trace = decision_trace

                # 11) Persist per-user state
                save_chat_state({
                    "messages": st.session_state.messages,
                    "dialogue_active": st.session_state.dialogue_active,
                    "dialogue_turns": st.session_state.dialogue_turns,
                    "wrapup_done": st.session_state.wrapup_done,
                    "conversation_summary": st.session_state.conversation_summary,
                    "open_threads": st.session_state.open_threads,
                    "facts_memory": st.session_state.facts_memory,
                    "last_bot_outputs": st.session_state.last_bot_outputs,
                    "last_checkin": st.session_state.last_checkin,
                    "exercise_followup": st.session_state.exercise_followup,
                    "support_indicator_history": st.session_state.support_indicator_history,
                })

                st.rerun()

        st.markdown(
            f"""
            <p class="footer-disclaimer">
              {t("chat_footer_disclaimer")}
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# ΙΣΤΟΡΙΚΟ TAB (Interactive Glass Timeline)
# ============================================================

with tab_history:
    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("history_page_title")}</h1>
          <p class='page-subtitle'>{t("history_page_sub")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Φόρτωση από τη Supabase (αντί για το παλιό, μη προστατευμένο CSV)
    df_user = load_checkins_df()

    if df_user.empty:
        st.info(t("history_empty"))
    else:
        st.markdown(f"### {t('history_heading')}")
        
        # Ταξινόμηση ώστε τα πιο πρόσφατα να είναι πάνω
        if "timestamp" in df_user.columns and not df_user["timestamp"].isna().all():
            df_user = df_user.sort_values("timestamp", ascending=False).head(60)
        else:
            df_user = df_user.iloc[::-1].head(60)

        # Ετοιμάζουμε τα δεδομένα για να τα περάσουμε στη Javascript
        timeline_data = []
        for _, row in df_user.iterrows():
            ts = row.get("timestamp", "")
            if pd.notna(ts) and str(ts).strip():
                try:
                    # Μορφοποίηση ημερομηνίας (π.χ. 15 May 2026 • 20:30)
                    ts_str = pd.to_datetime(ts).strftime("%d %b %Y • %H:%M")
                except Exception:
                    ts_str = str(ts)
            else:
                ts_str = "-"
                
            mood = row.get("mood", 5)
            sleep = row.get("sleep", 0)
            water = row.get("water", 0)
            raw_msg = str(row.get("message", "") or "")
            
            # Καθαρισμός HTML tags για ασφάλεια
            msg_no_tags = re.sub(r"<[^>]+>", "", raw_msg)
            msg_clean = html.unescape(msg_no_tags).strip()
            
            try:
                mood_val = float(mood)
            except:
                mood_val = 5.0
                
            # Καθορισμός εικονιδίου/χρώματος βάσει της διάθεσης
            mood_icon_svg, mood_color = mood_weather_svg(mood_val)

            timeline_data.append({
                "date": ts_str,
                "mood": mood,
                "sleep": sleep,
                "water": water,
                "message": msg_clean,
                "icon": mood_icon_svg,
                "color": mood_color
            })
            
        # Μετατροπή της λίστας σε JSON string για να το διαβάσει η JS
        timeline_json = json.dumps(timeline_data, ensure_ascii=False)
        
        # Το υπερ-ρεαλιστικό Timeline 
        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0; padding: 20px 10px; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: transparent;
                }}
                /* Η κεντρική γραμμή του Timeline */
                .timeline {{
                    position: relative;
                    max-width: 850px;
                    margin: 0 auto;
                    padding-left: 50px; /* Χώρος για τη γραμμή και τα εικονίδια */
                }}
                .timeline::before {{
                    content: '';
                    position: absolute;
                    left: 19px;
                    top: 10px;
                    bottom: 0;
                    width: 3px;
                    background: rgba(212, 190, 250, 0.4); /* Λιλά γραμμή */
                    border-radius: 4px;
                }}
                /* Κάθε κάρτα/καταγραφή */
                .item {{
                    position: relative;
                    margin-bottom: 35px;
                    opacity: 0; /* Αρχικά κρυμμένο για το animation */
                    transform: translateY(40px);
                    animation: slideUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
                }}
                @keyframes slideUp {{
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                /* Το κυκλικό εικονίδιο πάνω στη γραμμή */
                .icon {{
                    position: absolute;
                    left: -53px;
                    top: 0;
                    width: 44px;
                    height: 44px;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.8);
                    backdrop-filter: blur(10px);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 20px;
                    box-shadow: 0 6px 15px rgba(108, 90, 158, 0.12);
                    z-index: 1;
                    border: 3px solid; /* Το χρώμα μπαίνει δυναμικά από τη JS */
                    transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
                }}
                .item:hover .icon {{
                    transform: scale(1.2) rotate(-10deg);
                    box-shadow: 0 8px 20px rgba(108, 90, 158, 0.25);
                }}
                /* Η γυάλινη κάρτα (Glassmorphism) */
                .card {{
                    background: rgba(255, 255, 255, 0.45);
                    backdrop-filter: blur(24px);
                    -webkit-backdrop-filter: blur(24px);
                    border: 1px solid rgba(255, 255, 255, 0.7);
                    border-radius: 20px;
                    padding: 22px;
                    box-shadow: 0 12px 35px rgba(108, 90, 158, 0.05);
                    transition: all 0.3s ease;
                }}
                .item:hover .card {{
                    transform: translateY(-4px);
                    box-shadow: 0 18px 45px rgba(108, 90, 158, 0.12);
                    background: rgba(255, 255, 255, 0.6);
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    border-bottom: 1px solid rgba(183, 157, 242, 0.25);
                    padding-bottom: 10px;
                }}
                .date {{
                    font-weight: 600;
                    font-size: 14.5px;
                    color: #73658a;
                }}
                .metrics {{
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}
                .badge {{
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-size: 12.5px;
                    font-weight: 600;
                    color: #4A3D73;
                    background: rgba(255, 255, 255, 0.6);
                    border: 1px solid rgba(255, 255, 255, 0.9);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
                }}
                .message {{
                    font-size: 15.5px;
                    line-height: 1.6;
                    color: #3b304c;
                    font-style: italic;
                    word-wrap: break-word;
                }}
            </style>
        </head>
        <body>
            <div class="timeline" id="timelineContainer"></div>
            
            <script>
                // Παίρνουμε τα δεδομένα από την Python
                const data = {timeline_json};
                const container = document.getElementById('timelineContainer');
                
                data.forEach((entry, index) => {{
                    // Staggered Animation: Κάθε κάρτα εμφανίζεται με διαφορά 0.15s
                    const delay = index * 0.15;
                    
                    const item = document.createElement('div');
                    item.className = 'item';
                    item.style.animationDelay = `${{delay}}s`;
                    
                    item.innerHTML = `
                        <div class="icon" style="border-color: ${{entry.color}};">
                            ${{entry.icon}}
                        </div>
                        <div class="card">
                            <div class="header">
                                <div class="date">📅 ${{entry.date}}</div>
                                <div class="metrics">
                                    <span class="badge">{t("history_badge_mood")}: ${{entry.mood}}/10</span>
                                    <span class="badge">{t("history_badge_sleep")}: ${{entry.sleep}}h</span>
                                    <span class="badge">{t("history_badge_water")}: ${{entry.water}}</span>
                                </div>
                            </div>
                            <div class="message">«${{entry.message}}»</div>
                        </div>
                    `;
                    container.appendChild(item);
                }});
            </script>
        </body>
        </html>
        """, height=750, scrolling=True)


# ============================================================
# ΣΤΑΤΙΣΤΙΚΑ TAB (Interactive Glass Charts με Chart.js)
# ============================================================

with tab_stats:
    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("stats_page_title")}</h1>
          <p class='page-subtitle'>{t("stats_page_sub")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_checkins_df()

    if df.empty:
        st.info(t("stats_empty"))
    else:
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        # --- STREAK: υπολογίζεται πάντα στο ΠΛΗΡΕΣ ιστορικό,
        # ανεξάρτητα από το χρονικό παράθυρο που θα διαλέξει
        # ο χρήστης παρακάτω για τα γραφήματα.
        all_dates = df["timestamp"].dropna().tolist() if "timestamp" in df.columns else []
        current_streak = compute_current_streak(all_dates)
        _streak_unit = t("stats_streak_day") if current_streak == 1 else t("stats_streak_days")

        st.markdown(
            f"""
            <div class="stats-card" style="background: linear-gradient(135deg, rgba(232,145,107,0.22), rgba(232,145,107,0.08)) !important; display:flex; flex-direction:row; align-items:center; gap:12px; animation: cardFadeIn 0.3s ease-out;">
              {streak_flame_svg()}
              <div>
                <div class="stats-card-main" style="margin-bottom:0;">{current_streak} {_streak_unit}</div>
                <div class="stats-card-sub">{t("stats_streak_label")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Εξαγωγή σε PDF (για να μοιραστεί κανείς με πραγματικό ειδικό) ---
        col_pdf1, col_pdf2 = st.columns([2, 1])
        with col_pdf2:
            if st.button(t("stats_pdf_generate"), key="btn_gen_pdf", use_container_width=True):
                screening_for_pdf = load_screening_history(limit=10)
                pdf_bytes = generate_wellness_pdf(
                    user_name=st.session_state.get("user_name", ""),
                    df_user=df,
                    current_streak=current_streak,
                    screening_history=screening_for_pdf,
                )
                st.session_state["_pdf_export_bytes"] = pdf_bytes

            if st.session_state.get("_pdf_export_bytes"):
                st.download_button(
                    t("stats_pdf_download"),
                    data=st.session_state["_pdf_export_bytes"],
                    file_name=f"project_wellness_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="btn_download_pdf",
                    use_container_width=True,
                )

        # --- Επιλογή χρονικού παραθύρου για τα παρακάτω γραφήματα ---
        _window_options = [t("stats_window_15"), t("stats_window_30"), t("stats_window_90")]
        window_label = st.radio(
            "window",
            _window_options,
            horizontal=True,
            key="stats_window",
            label_visibility="collapsed",
        )
        window_n = {_window_options[0]: 15, _window_options[1]: 30, _window_options[2]: 90}[window_label]


        df_last = df.tail(window_n).copy()

        for col in ["mood", "sleep", "water"]:
            df_last[col] = pd.to_numeric(df_last[col], errors="coerce").fillna(0)

        avg_mood = df_last["mood"].mean()
        avg_sleep = df_last["sleep"].mean()
        avg_water = df_last["water"].mean()

        last_row = df_last.iloc[-1]
        last_mood = float(last_row.get("mood", 0) or 0)
        last_sleep = float(last_row.get("sleep", 0) or 0)
        last_water = float(last_row.get("water", 0) or 0)

        # --- 1. ΚΑΡΤΕΣ ΣΥΝΟΨΗΣ ΣΤΗΝ ΚΟΡΥΦΗ ---
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f"""
                <div class="stats-card stats-card-mood" style="animation: cardFadeIn 0.4s ease-out;">
                  <div class="stats-card-label">{t("mood_metric")}</div>
                  <div class="stats-card-main">{last_mood:.1f}/10</div>
                  <div class="stats-card-sub">{t("stats_avg_period")}: {avg_mood:.1f}</div>
                </div>
                """, unsafe_allow_html=True
            )
        with col_b:
            st.markdown(
                f"""
                <div class="stats-card stats-card-sleep" style="animation: cardFadeIn 0.6s ease-out;">
                  <div class="stats-card-label">{t("sleep_metric")}</div>
                  <div class="stats-card-main">{last_sleep:.1f}h</div>
                  <div class="stats-card-sub">{t("stats_avg_period")}: {avg_sleep:.1f}h</div>
                </div>
                """, unsafe_allow_html=True
            )
        with col_c:
            st.markdown(
                f"""
                <div class="stats-card stats-card-water" style="animation: cardFadeIn 0.8s ease-out;">
                  <div class="stats-card-label">{t("water_metric")}</div>
                  <div class="stats-card-main">{last_water:.0f} {t("stats_water_unit")}</div>
                  <div class="stats-card-sub">{t("stats_avg_period")}: {avg_water:.1f}</div>
                </div>
                """, unsafe_allow_html=True
            )

        # --- 2. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ΓΙΑ ΤΗ JAVASCRIPT ---
        if "timestamp" not in df_last.columns or df_last["timestamp"].isna().all():
            st.info(t("stats_no_dates"))
        else:
            df_last = df_last.dropna(subset=["timestamp"])
            
            # Μετατροπή ημερομηνιών σε μικρή μορφή (π.χ. "15/05")
            dates_list = df_last["timestamp"].dt.strftime("%d/%m").tolist()
            mood_list = df_last["mood"].tolist()
            sleep_list = df_last["sleep"].tolist()
            water_list = df_last["water"].tolist()

            dates_json = json.dumps(dates_list)
            mood_json = json.dumps(mood_list)
            sleep_json = json.dumps(sleep_list)
            water_json = json.dumps(water_list)

            # --- 3. JAVASCRIPT & CHART.JS ΟΠΤΙΚΟΠΟΙΗΣΗ ---
            components.html(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">
                <style>
                    body {{
                        margin: 0; padding: 10px;
                        font-family: 'Manrope', system-ui, sans-serif;
                        background: transparent;
                    }}
                    .chart-container {{
                        background: rgba(255, 255, 255, 0.45);
                        backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                        border: 1px solid rgba(255, 255, 255, 0.7);
                        border-radius: 20px;
                        padding: 20px;
                        margin-bottom: 25px;
                        box-shadow: 0 10px 30px rgba(108, 90, 158, 0.08);
                        transition: transform 0.3s ease;
                        opacity: 0;
                        transform: translateY(20px);
                        animation: fadeInUp 0.8s ease-out forwards;
                    }}
                    .chart-container:hover {{
                        transform: translateY(-4px);
                        box-shadow: 0 15px 40px rgba(108, 90, 158, 0.15);
                        background: rgba(255, 255, 255, 0.6);
                    }}
                    .chart-title {{
                        color: #4A3D73; font-weight: 600; font-size: 16px; 
                        margin-bottom: 15px; margin-top: 0; display: flex; align-items: center; gap: 8px;
                    }}
                    /* Staggered animation delays */
                    #c1 {{ animation-delay: 0.1s; }}
                    #c2 {{ animation-delay: 0.3s; }}
                    #c3 {{ animation-delay: 0.5s; }}
                    
                    @keyframes fadeInUp {{
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                    
                    /* Container για τα canvas ώστε να είναι responsive */
                    .canvas-wrapper {{ position: relative; height: 220px; width: 100%; }}
                </style>
            </head>
            <body>

                <div class="chart-container" id="c1">
                    <h3 class="chart-title">{t("stats_chart_mood")}</h3>
                    <div class="canvas-wrapper"><canvas id="moodChart"></canvas></div>
                </div>

                <div class="chart-container" id="c2">
                    <h3 class="chart-title">{t("stats_chart_sleep")}</h3>
                    <div class="canvas-wrapper"><canvas id="sleepChart"></canvas></div>
                </div>

                <div class="chart-container" id="c3">
                    <h3 class="chart-title">{t("stats_chart_water")}</h3>
                    <div class="canvas-wrapper"><canvas id="waterChart"></canvas></div>
                </div>

                <script>
                    // Κοινές ρυθμίσεις για γραμματοσειρές και στυλ αξόνων
                    Chart.defaults.font.family = "'Segoe UI', Tahoma, sans-serif";
                    Chart.defaults.color = '#73658a';
                    Chart.defaults.scale.grid.color = 'rgba(183, 157, 242, 0.15)';
                    
                    const commonOptions = {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                                backgroundColor: 'rgba(74, 61, 115, 0.9)',
                                titleFont: {{ size: 13 }},
                                bodyFont: {{ size: 14, weight: 'bold' }},
                                padding: 12,
                                cornerRadius: 8,
                                displayColors: false
                            }}
                        }},
                        scales: {{
                            x: {{ grid: {{ display: false }} }},
                            y: {{ beginAtZero: true, border: {{ display: false }} }}
                        }},
                        interaction: {{ mode: 'index', intersect: false }},
                        elements: {{
                            line: {{ tension: 0.4 }}, // 0.4 δίνει την απαλή καμπύλη (spline)
                            point: {{ radius: 4, hoverRadius: 7, borderWidth: 2 }}
                        }}
                    }};

                    // Δεδομένα από Python
                    const labels = {dates_json};

                    // --- 1. ΔΙΑΓΡΑΜΜΑ ΔΙΑΘΕΣΗΣ ---
                    const ctxMood = document.getElementById('moodChart').getContext('2d');
                    let gradientMood = ctxMood.createLinearGradient(0, 0, 0, 220);
                    gradientMood.addColorStop(0, 'rgba(242, 166, 184, 0.6)'); // twilight-rose
                    gradientMood.addColorStop(1, 'rgba(242, 166, 184, 0.0)');
                    
                    new Chart(ctxMood, {{
                        type: 'line',
                        data: {{
                            labels: labels,
                            datasets: [{{
                                label: '{t("stats_dataset_mood")}',
                                data: {mood_json},
                                borderColor: '#F2A6B8',
                                backgroundColor: gradientMood,
                                pointBackgroundColor: '#ffffff',
                                pointBorderColor: '#F2A6B8',
                                fill: true,
                            }}]
                        }},
                        options: {{ ...commonOptions, scales: {{ y: {{ max: 10, min: 0 }} }} }}
                    }});

                    // --- 2. ΔΙΑΓΡΑΜΜΑ ΥΠΝΟΥ ---
                    const ctxSleep = document.getElementById('sleepChart').getContext('2d');
                    let gradientSleep = ctxSleep.createLinearGradient(0, 0, 0, 220);
                    gradientSleep.addColorStop(0, 'rgba(183, 157, 242, 0.6)'); // twilight-lilac
                    gradientSleep.addColorStop(1, 'rgba(183, 157, 242, 0.0)');

                    new Chart(ctxSleep, {{
                        type: 'line',
                        data: {{
                            labels: labels,
                            datasets: [{{
                                label: '{t("stats_dataset_sleep")}',
                                data: {sleep_json},
                                borderColor: '#B79DF2',
                                backgroundColor: gradientSleep,
                                pointBackgroundColor: '#ffffff',
                                pointBorderColor: '#B79DF2',
                                fill: true,
                            }}]
                        }},
                        options: commonOptions
                    }});

                    // --- 3. ΔΙΑΓΡΑΜΜΑ ΝΕΡΟΥ ---
                    const ctxWater = document.getElementById('waterChart').getContext('2d');
                    let gradientWater = ctxWater.createLinearGradient(0, 0, 0, 220);
                    gradientWater.addColorStop(0, 'rgba(143, 175, 140, 0.6)'); // twilight-sage
                    gradientWater.addColorStop(1, 'rgba(143, 175, 140, 0.0)');

                    new Chart(ctxWater, {{
                        type: 'line',
                        data: {{
                            labels: labels,
                            datasets: [{{
                                label: '{t("stats_dataset_water")}',
                                data: {water_json},
                                borderColor: '#8FAF8C',
                                backgroundColor: gradientWater,
                                pointBackgroundColor: '#ffffff',
                                pointBorderColor: '#8FAF8C',
                                fill: true,
                            }}]
                        }},
                        options: commonOptions
                    }});
                </script>
            </body>
            </html>
            """, height=900, scrolling=True)


# ============================================================
# ΑΣΚΗΣΕΙΣ TAB
# ============================================================

with tab_ex:
    if "support1" not in st.session_state:
        st.session_state.support1 = ""
    if "support2" not in st.session_state:
        st.session_state.support2 = ""
    if "support3" not in st.session_state:
        st.session_state.support3 = ""

    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("ex_page_title")}</h1>
          <p class='page-subtitle'>
            {t("ex_page_sub")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 1. Άσκηση αναπνοής 4–2–6 (ΜΕ JAVASCRIPT & ANIMATION)
    # --------------------------------------------------------
    st.markdown(f"### {t('ex1_heading')}")
    st.write(t("ex1_caption"))
    
    # Εδώ γράφουμε καθαρή HTML, CSS και JAVASCRIPT!
    # (Παραμένει plain string -- ΟΧΙ f-string -- για να μη χρειάζεται να
    # γίνουν escape όλα τα CSS/JS { } . Το κείμενο μπαίνει μέσω .replace(),
    # που είναι 100% ασφαλές: δεν αγγίζει το CSS/JS.)
    _ex1_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    display: flex; justify-content: center; align-items: center;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    height: 250px; margin: 0; background-color: transparent;
                }
                .container {
                    text-align: center;
                }
                .circle {
                    width: 120px; height: 120px;
                    border-radius: 50%;
                    background: radial-gradient(circle, #ffb6c1 0%, #ff69b4 100%);
                    box-shadow: 0 10px 30px rgba(255, 105, 180, 0.4);
                    margin: 0 auto 20px auto;
                    display: flex; justify-content: center; align-items: center;
                    color: white; font-weight: bold; font-size: 18px;
                    transition: transform 4s ease-in-out; /* Ομαλό animation */
                }
                button {
                    background-color: #6C5A9E; color: white; border: none;
                    padding: 10px 24px; border-radius: 8px; font-size: 16px;
                    cursor: pointer; transition: 0.3s;
                }
                button:hover { background-color: #4A3D73; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="circle" id="breathCircle">__READY__</div>
                <br>
                <button id="startBtn" onclick="startBreathing()">__START__</button>
            </div>

            <script>
                function startBreathing() {
                    const circle = document.getElementById('breathCircle');
                    const btn = document.getElementById('startBtn');
                    btn.style.display = 'none'; // Κρύβουμε το κουμπί

                    function breatheCycle() {
                        // 1. Εισπνοή (4 δευτερόλεπτα) - Ο κύκλος μεγαλώνει
                        circle.style.transition = 'transform 4s ease-in-out';
                        circle.style.transform = 'scale(1.6)';
                        circle.innerText = '__INHALE__';

                        setTimeout(() => {
                            // 2. Κράτημα (2 δευτερόλεπτα) - Ο κύκλος μένει μεγάλος
                            circle.innerText = '__HOLD__';

                            setTimeout(() => {
                                // 3. Εκπνοή (6 δευτερόλεπτα) - Ο κύκλος μικραίνει
                                circle.style.transition = 'transform 6s ease-in-out';
                                circle.style.transform = 'scale(1)';
                                circle.innerText = '__EXHALE__';
                            }, 2000); // Περιμένει τα 2 δευτερόλεπτα του κρατήματος

                        }, 4000); // Περιμένει τα 4 δευτερόλεπτα της εισπνοής
                    }

                    // Ξεκινάμε τον πρώτο κύκλο αμέσως
                    breatheCycle();
                    // Επαναλαμβάνουμε κάθε 12 δευτερόλεπτα (4+2+6)
                    setInterval(breatheCycle, 12000);
                }
            </script>
        </body>
        </html>
        """
    _ex1_html = (
        _ex1_html
        .replace("__READY__", t("ex1_ready"))
        .replace("__START__", t("ex1_start"))
        .replace("__INHALE__", t("ex1_inhale"))
        .replace("__HOLD__", t("ex1_hold"))
        .replace("__EXHALE__", t("ex1_exhale"))
    )
    components.html(_ex1_html, height=300)
    st.markdown("---")

    # --------------------------------------------------------
    # 2. Αποφόρτιση σκέψεων (100% LLM Powered)
    # --------------------------------------------------------
    st.markdown(f"### {t('ex2_heading')}")
    st.write(t("ex2_caption"))

    # Χρησιμοποιούμε session_state για να θυμόμαστε αν ολοκληρώθηκε η άσκηση
    if "discharge_done" not in st.session_state:
        st.session_state.discharge_done = False
        st.session_state.discharge_msg = ""

    # Ένα placeholder container για να μπορούμε να αλλάζουμε την οθόνη δυναμικά
    discharge_placeholder = st.empty()

    if not st.session_state.discharge_done:
        with discharge_placeholder.container():
            discharge_input = st.text_area(
                t("ex2_input_label"),
                height=80,
                key="discharge_input_text"
            )
            
            if st.button(t("ex2_button"), key="btn_discharge"):
                if not discharge_input.strip():
                    st.warning(t("ex2_warning_empty"))
                else:
                    with st.spinner(t("ex2_spinner")):
                        # ΚΛΗΣΗ ΣΤΟ LLM
                        profile = load_profile()
                        
                        # Στέλνουμε τη σκέψη σαν μέρος του context
                        chat_tail = get_chat_tail_from_state() + [f"Θέλω να αποφορτίσω αυτή τη σκέψη που με βαραίνει: {discharge_input}"]
                        active_mode = st.session_state.get("active_mode", "NONE")

                        key = "discharge_thought"
                        history = st.session_state.microcoach_history.get(key, [])

                        msg = micro_prompt_with_fallback(
                            exercise_key=key,
                            profile=profile,
                            chat_tail=chat_tail,
                            active_mode=active_mode,
                            avoid_texts=history,
                        lang_instruction=lang_directive(),
                        lang=get_lang(),
                        )

                        history.append(msg)
                        st.session_state.microcoach_history[key] = history[-8:]

                        # Αποθήκευση και αλλαγή οθόνης
                        st.session_state.discharge_msg = msg
                        st.session_state.discharge_done = True
                        st.rerun()
    else:
        # Αν η αποφόρτιση έγινε, δείχνουμε την απάντηση του LLM στο γυάλινο design
        with discharge_placeholder.container():
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.45);
                            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.6);
                            border-radius: 16px; padding: 30px; text-align: center; 
                            box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08);
                            animation: cardFadeIn 1s ease-out; margin-bottom: 15px;">
                    <div style="font-size: 45px; margin-bottom: 15px; opacity: 0.9;">🍃</div>
                    <div style="color: #3b304c; font-size: 1.1rem; line-height: 1.6; font-weight: 500;">
                        {st.session_state.discharge_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Κουμπί για επαναφορά
            col_spacer1, col_reset, col_spacer2 = st.columns([1, 1, 1])
            with col_reset:
                if st.button(t("ex2_reset_button"), key="reset_discharge", use_container_width=True):
                    st.session_state.discharge_done = False
                    st.session_state.discharge_msg = ""
                    st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # 3. Θλίψη / μοναξιά (100% LLM Powered - Native Glassmorphism)
    # --------------------------------------------------------
    st.markdown(f"### {t('ex3_heading')}")
    st.write(t("ex3_caption"))

    if "lonely_done" not in st.session_state:
        st.session_state.lonely_done = False
        st.session_state.lonely_msg = ""

    lonely_placeholder = st.empty()

    if not st.session_state.lonely_done:
        with lonely_placeholder.container():
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; 
                            padding: 30px; text-align: center; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08);">
                    <div style="font-size: 60px; margin-bottom: 15px; display: inline-block; animation: softPulse 3s infinite;">🤍</div>
                    <div style="color: #3b304c; font-size: 1.1rem; margin-bottom: 20px; font-weight: 500;">
                        {t("ex3_intro")}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Βάζουμε κενά δεξιά-αριστερά για να μην απλώνει τελείως το κουμπί
            col_l1, col_lbtn, col_l2 = st.columns([1, 2, 1])
            with col_lbtn:
                if st.button(t("ex3_start_button"), key="btn_lonely_start", use_container_width=True):
                    with st.spinner(t("ex3_spinner")):
                        profile = load_profile()
                        
                        # Στέλνουμε το chat context + μια κρυφή "σκηνοθετική" οδηγία στο LLM
                        chat_tail = get_chat_tail_from_state() 
                        chat_tail.append(
                            "Ο χρήστης μόλις πάτησε το κουμπί για την άσκηση ανακούφισης από μοναξιά/θλίψη. "
                            "Βάσει του ιστορικού και του προφίλ του, δώσε του 2-3 προτάσεις βαθιάς, ζεστής ενσυναίσθησης "
                            "που να τον αγγίζουν, και κλείσε με 1 θετική επιβεβαίωση (affirmation) σε εισαγωγικά που να μπορεί να πει στον εαυτό του."
                        )
                        
                        active_mode = st.session_state.get("active_mode", "NONE")
                        key = "lonely_care"
                        history = st.session_state.microcoach_history.get(key, [])

                        msg = micro_prompt_with_fallback(
                            exercise_key=key,
                            profile=profile,
                            chat_tail=chat_tail,
                            active_mode=active_mode,
                            avoid_texts=history,
                        lang_instruction=lang_directive(),
                        lang=get_lang(),
                        )

                        history.append(msg)
                        st.session_state.microcoach_history[key] = history[-8:]

                        st.session_state.lonely_msg = msg
                        st.session_state.lonely_done = True
                        st.rerun()
    else:
        # Εμφάνιση της προσωποποιημένης απάντησης
        with lonely_placeholder.container():
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
                            border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; 
                            padding: 30px; text-align: center; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08);
                            animation: cardFadeIn 1s ease-out;">
                    <div style="font-size: 60px; margin-bottom: 15px; display: inline-block; animation: softPulse 2s infinite;">💖</div>
                    <div style="color: #3b304c; font-size: 1.05rem; line-height: 1.6; margin-bottom: 20px;">
                        {t("ex3_step1")}
                        {t("ex3_step2")}
                        {t("ex3_step3")}
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.75); border-left: 4px solid #ff8fa3; 
                                padding: 20px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.02);">
                        <div style="color: #d81b60; font-size: 1.1rem; line-height: 1.6; font-weight: 500; font-style: italic;">
                            {st.session_state.lonely_msg}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            col_l1, col_lbtn, col_l2 = st.columns([1, 1, 1])
            with col_lbtn:
                if st.button(t("ex3_return_button"), key="reset_lonely", use_container_width=True):
                    st.session_state.lonely_done = False
                    st.session_state.lonely_msg = ""
                    st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # 4. Νερό (ΜΕ JAVASCRIPT ANIMATION - ΓΕΜΙΣΜΑ ΠΟΤΗΡΙΟΥ)
    # --------------------------------------------------------
    st.markdown(f"### {t('ex4_heading')}")

    col_w1, col_w2 = st.columns([1.2, 1])

    with col_w1:
        _ex4_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { 
                        margin: 0; display: flex; justify-content: center; align-items: center; 
                        font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; background: transparent; height: 240px;
                    }
                    .container { text-align: center; width: 100%; }
                    .info-text { 
                        color: #4A3D73; font-weight: 600; font-size: 15px; margin-bottom: 15px;
                    }
                    /* Το ποτήρι */
                    .glass {
                        width: 70px; height: 110px;
                        border: 4px solid #acd8e5;
                        border-top: none;
                        border-radius: 0 0 15px 15px;
                        position: relative;
                        overflow: hidden;
                        margin: 0 auto;
                        cursor: pointer;
                        background: rgba(255, 255, 255, 0.7);
                        box-shadow: 0 8px 15px rgba(172, 216, 229, 0.3);
                        transition: transform 0.3s ease;
                    }
                    .glass:hover {
                        transform: translateY(-4px);
                        box-shadow: 0 12px 20px rgba(172, 216, 229, 0.5);
                    }
                    /* Το νερό με το εφέ κύματος */
                    .water {
                        position: absolute;
                        bottom: 0;
                        left: -50%;
                        width: 200%;
                        height: 15%; /* Αρχίζει σχεδόν άδειο */
                        background: #89cff0;
                        transition: height 2.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                        border-radius: 40%;
                        animation: wave-spin 3s infinite linear;
                        opacity: 0.85;
                    }
                    @keyframes wave-spin {
                        100% { transform: rotate(360deg); }
                    }
                    .hidden-msg { 
                        opacity: 0; transition: opacity 1s; color: #2e7d32; 
                        margin-top: 15px; font-weight: 600; font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="info-text">__HINT__</div>
                    
                    <div class="glass" onclick="fillWater()" title="__GLASS_TITLE__">
                       <div class="water" id="waterLvl"></div>
                    </div>
                    
                    <div id="cheers" class="hidden-msg">__CHEERS__</div>
                </div>
                <script>
                    function fillWater() {
                        const water = document.getElementById('waterLvl');
                        const msg = document.getElementById('cheers');
                        
                        // Ανεβάζουμε τη στάθμη του νερού πάνω από 100% για να γεμίσει το ποτήρι
                        water.style.height = '120%'; 
                        
                        // Εμφάνιση του μηνύματος αφού γεμίσει το νερό
                        setTimeout(() => {
                            msg.style.opacity = 1;
                        }, 1800);
                    }
                </script>
            </body>
            </html>
            """
        _ex4_html = (
            _ex4_html
            .replace("__HINT__", t("ex4_glass_hint"))
            .replace("__GLASS_TITLE__", t("ex4_glass_title"))
            .replace("__CHEERS__", t("ex4_cheers"))
        )
        components.html(_ex4_html, height=250)

    with col_w2:
        st.write(t("ex4_side_text"))
        # Κρατάμε το LLM button ανέπαφο!
        if st.button(t("ex4_button"), key="water_nudge"):
            profile = load_profile()
            chat_tail = get_chat_tail_from_state()
            active_mode = st.session_state.get("active_mode", "NONE")

            key = "water_nudge"
            history = st.session_state.microcoach_history.get(key, [])

            msg = micro_prompt_with_fallback(
                exercise_key=key,
                profile=profile,
                chat_tail=chat_tail,
                active_mode=active_mode,
                avoid_texts=history,
            lang_instruction=lang_directive(),
            lang=get_lang(),
            )

            history.append(msg)
            st.session_state.microcoach_history[key] = history[-8:]
            st.success(msg)

    st.markdown("---")

    # --------------------------------------------------------
    # 5. 3 μικρά στηρίγματα (100% LLM Powered)
    # --------------------------------------------------------
    st.markdown(f"### 5. {t('ex5_heading_suffix')}")
    st.write(t("ex5_caption"))

    if "supports_done" not in st.session_state:
        st.session_state.supports_done = False
        st.session_state.supports_msg = ""
        st.session_state.user_supports = []

    supports_placeholder = st.empty()

    if not st.session_state.supports_done:
        with supports_placeholder.container():
            # Χρησιμοποιούμε στήλες για να μπουν δίπλα-δίπλα τα πεδία
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                s1 = st.text_input(t("ex5_support1"), key="sup1")
            with col_s2:
                s2 = st.text_input(t("ex5_support2"), key="sup2")
            with col_s3:
                s3 = st.text_input(t("ex5_support3"), key="sup3")
                
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button(t("ex5_button"), use_container_width=True, key="btn_save_supports"):
                    supports = [s for s in [s1, s2, s3] if s.strip()]
                    if not supports:
                        st.warning(t("ex5_warning_empty"))
                    else:
                        with st.spinner(t("ex5_spinner")):
                            profile = load_profile()
                            supports_text = ", ".join(supports)
                            
                            # Σκηνοθετική οδηγία προς το LLM!
                            chat_tail = get_chat_tail_from_state()
                            chat_tail.append(
                                f"Ο χρήστης μόλις κατέγραψε τα εξής στηρίγματα για τη μέρα του: [{supports_text}]. "
                                "Βάσει του ιστορικού και του προφίλ του, γράψε 2-3 προτάσεις που να αναγνωρίζουν "
                                "την αξία ΑΥΤΩΝ των συγκεκριμένων πραγμάτων. Δείξε του πώς αυτά τα μικρά πράγματα "
                                "δείχνουν τη δύναμή του ή τη φροντίδα προς τον εαυτό του. Μην δώσεις συμβουλές, απλώς "
                                "επικύρωσε (validate) συναισθηματικά αυτή του την προσπάθεια."
                            )
                            
                            active_mode = st.session_state.get("active_mode", "NONE")
                            key = "supports_exercise"
                            history = st.session_state.microcoach_history.get(key, [])

                            msg = micro_prompt_with_fallback(
                                exercise_key=key,
                                profile=profile,
                                chat_tail=chat_tail,
                                active_mode=active_mode,
                                avoid_texts=history,
                            lang_instruction=lang_directive(),
                            lang=get_lang(),
                            )

                            history.append(msg)
                            st.session_state.microcoach_history[key] = history[-8:]

                            # Αποθήκευση στο session state
                            st.session_state.user_supports = supports
                            st.session_state.supports_msg = msg
                            st.session_state.supports_done = True
                            st.rerun()
    else:
        # Εμφάνιση των αποτελεσμάτων σε Glassmorphism UI
        with supports_placeholder.container():
            # Φτιάχνουμε δυναμικά τις κάρτες (pillars) για όσα στηρίγματα έγραψε
            import html
            pillars_html = "".join([
                f"<div style='background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); border-left: 4px solid #b79df2; padding: 12px 20px; margin: 10px auto; border-radius: 8px; font-weight: 600; color: #4A3D73; width: 80%; box-shadow: 0 2px 8px rgba(108, 90, 158, 0.15);'>{html.escape(s)}</div>" 
                for s in st.session_state.user_supports
            ])
            
            # ΕΔΩ: Η HTML χωρίς κενά στην αρχή της κάθε γραμμής
            st.markdown(
                f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.8s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.1rem; margin-top: 0;">{t("ex5_foundations_title")}</h3>
{pillars_html}
<div style="background: rgba(255, 255, 255, 0.75); border-left: 4px solid #baf7c3; padding: 20px; border-radius: 12px; margin-top: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.02);">
<div style="color: #158047; font-size: 1.05rem; line-height: 1.6; font-weight: 500; font-style: italic;">
{st.session_state.supports_msg}
</div>
</div>
</div>""", 
                unsafe_allow_html=True
            )
            
            col_l1, col_lbtn, col_l2 = st.columns([1, 1, 1])
            with col_lbtn:
                if st.button(t("ex5_reset_button"), key="reset_supports", use_container_width=True):
                    st.session_state.supports_done = False
                    st.session_state.supports_msg = ""
                    st.session_state.user_supports = []
                    st.rerun()
                    
    st.markdown("---")

    # --------------------------------------------------------
    # 6. Body scan (ΜΕ JAVASCRIPT ZEN ANIMATION)
    # --------------------------------------------------------
    st.markdown(f"### {t('ex6_heading')}")

    _ex6_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    display: flex; justify-content: center; align-items: center; background: transparent;
                }
                .scan-card {
                    width: 100%; max-width: 600px; min-height: 280px;
                    background: linear-gradient(145deg, #f3f0fb 0%, #ffffff 100%);
                    border-radius: 16px; border: 1px solid #e6e0f8;
                    box-shadow: 0 10px 30px rgba(108, 90, 158, 0.08);
                    display: flex; flex-direction: column; align-items: center; justify-content: center;
                    text-align: center; padding: 20px; position: relative; overflow: hidden;
                }
                /* Η σφαίρα εστίασης */
                .orb {
                    width: 140px; height: 140px; border-radius: 50%;
                    background: radial-gradient(circle, #b19cd9 0%, #6C5A9E 100%);
                    display: flex; justify-content: center; align-items: center;
                    color: white; font-weight: 600; font-size: 15px; letter-spacing: 0.5px;
                    box-shadow: 0 0 20px rgba(108, 90, 158, 0.4);
                    opacity: 0; transform: scale(0.5); /* Αρχικά κρυμμένη */
                    transition: all 1.5s ease-in-out;
                    text-align: center; padding: 10px;
                }
                .orb.active {
                    opacity: 1; transform: scale(1);
                    animation: breathe-pulse 4s infinite alternate ease-in-out;
                }
                @keyframes breathe-pulse {
                    0% { box-shadow: 0 0 15px rgba(108, 90, 158, 0.3); transform: scale(1); }
                    100% { box-shadow: 0 0 35px rgba(108, 90, 158, 0.6); transform: scale(1.08); }
                }
                .intro-text {
                    color: #4A3D73; font-size: 16px; margin-bottom: 20px; line-height: 1.5;
                    transition: opacity 1s;
                }
                .focus-text {
                    opacity: 1; transition: opacity 1s ease-in-out;
                }
                button {
                    background: #6C5A9E; color: white; border: none; padding: 12px 28px;
                    border-radius: 25px; font-size: 15px; cursor: pointer; transition: 0.3s; font-weight: 600;
                }
                button:hover { background: #4A3D73; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(108, 90, 158, 0.2); }
            </style>
        </head>
        <body>
            <div class="scan-card">
                <div id="intro" class="intro-text">
                    __INTRO__
                </div>
                
                <div id="orb" class="orb">
                    <span id="orbText" class="focus-text">__STAGE1__</span>
                </div>
                
                <br>
                <button id="startBtn" onclick="startScan()">__START__</button>
            </div>

            <script>
                function startScan() {
                    const btn = document.getElementById('startBtn');
                    const intro = document.getElementById('intro');
                    const orb = document.getElementById('orb');
                    const orbText = document.getElementById('orbText');

                    // Τα στάδια της σάρωσης
                    const stages = [
                        "__STAGE1__",
                        "__STAGE2__",
                        "__STAGE3__",
                        "__STAGE4__",
                        "__STAGE5__"
                    ];
                    
                    // Κρύβουμε εισαγωγή και κουμπί
                    btn.style.display = 'none';
                    intro.style.display = 'none';
                    
                    // Εμφανίζουμε τη σφαίρα
                    orb.classList.add('active');

                    let currentStage = 0;

                    // Λειτουργία για αλλαγή κειμένου με fade out -> αλλαγή -> fade in
                    function nextStage() {
                        currentStage++;
                        if (currentStage < stages.length) {
                            orbText.style.opacity = 0; // Fade out
                            
                            setTimeout(() => {
                                orbText.innerText = stages[currentStage];
                                orbText.style.opacity = 1; // Fade in
                            }, 1000); // Περιμένουμε 1 δευτερόλεπτο να σβήσει
                        } else {
                            // Τέλος άσκησης
                            setTimeout(() => {
                                orb.classList.remove('active');
                                orb.style.opacity = 0;
                                setTimeout(() => {
                                    intro.innerHTML = "__COMPLETE__";
                                    intro.style.display = 'block';
                                }, 1500);
                            }, 3000);
                        }
                    }

                    // Αλλάζουμε περιοχή του σώματος κάθε 8 δευτερόλεπτα (Συνολικά περίπου 40-45 sec)
                    const scanInterval = setInterval(() => {
                        if (currentStage >= stages.length - 1) {
                            clearInterval(scanInterval);
                        }
                        nextStage();
                    }, 8000); 
                }
            </script>
        </body>
        </html>
        """
    _ex6_html = (
        _ex6_html
        .replace("__INTRO__", t("ex6_intro"))
        .replace("__STAGE1__", t("ex6_stage1"))
        .replace("__STAGE2__", t("ex6_stage2"))
        .replace("__STAGE3__", t("ex6_stage3"))
        .replace("__STAGE4__", t("ex6_stage4"))
        .replace("__STAGE5__", t("ex6_stage5"))
        .replace("__START__", t("ex6_start_button"))
        .replace("__COMPLETE__", t("ex6_complete"))
    )
    components.html(_ex6_html, height=320)
    st.markdown("---")

    # --------------------------------------------------------
    # 7. Μικρό παιχνίδι – φροντιστική απάντηση (ΜΕ 3D FLIP CARD)
    # --------------------------------------------------------
    st.markdown(f"### {t('ex7_heading')}")
    st.write(t("ex7_caption"))

    difficult_thoughts = [
        t("ex7_thought1"),
        t("ex7_thought2"),
        t("ex7_thought3"),
        t("ex7_thought4"),
    ]

    chosen_label = st.selectbox(
        t("ex7_select_label"),
        difficult_thoughts,
        key="mh_game_select",
    )

    if st.button(t("ex7_button"), key="mh_game_button"):
        with st.spinner(t("ex7_spinner")):
            # Κλήση στο LLM (Backend)
            profile = load_profile()
            chat_tail = get_chat_tail_from_state() + [f"Δύσκολη σκέψη: {chosen_label}"]
            active_mode = st.session_state.get("active_mode", "NONE")

            key = "care_reply_game"
            history = st.session_state.microcoach_history.get(key, [])

            msg = micro_prompt_with_fallback(
                exercise_key=key,
                profile=profile,
                chat_tail=chat_tail,
                active_mode=active_mode,
                avoid_texts=history,
            lang_instruction=lang_directive(),
            lang=get_lang(),
            )

            history.append(msg)
            st.session_state.microcoach_history[key] = history[-8:]
            
            # Προετοιμασία κειμένων για να μπουν με ασφάλεια στην HTML
            safe_thought = html.escape(chosen_label).replace('\n', '<br>')
            safe_msg = html.escape(msg).replace('\n', '<br>')

            # Το 3D Flip Card (Frontend)
            components.html(
                f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{ 
                            margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                            display: flex; justify-content: center; background: transparent; padding-top: 15px; 
                        }}
                        .flip-card {{
                            background-color: transparent; width: 100%; max-width: 600px; 
                            height: 220px; perspective: 1000px;
                        }}
                        .flip-card-inner {{
                            position: relative; width: 100%; height: 100%; text-align: center;
                            transition: transform 1.2s cubic-bezier(0.4, 0.2, 0.2, 1);
                            transform-style: preserve-3d; cursor: pointer;
                        }}
                        /* Η κλάση που γυρνάει την κάρτα */
                        .flipped {{ transform: rotateY(180deg); }}
                        
                        .flip-card-front, .flip-card-back {{
                            position: absolute; width: 100%; height: 100%;
                            -webkit-backface-visibility: hidden; backface-visibility: hidden;
                            border-radius: 16px; display: flex; flex-direction: column;
                            justify-content: center; align-items: center; padding: 25px;
                            box-sizing: border-box; box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                        }}
                        /* Σκοτεινή πλευρά (Δύσκολη Σκέψη) */
                        .flip-card-front {{
                            background: linear-gradient(135deg, #4c4c4c 0%, #2a2a2a 100%);
                            color: #f1f1f1; border: 1px solid #555;
                        }}
                        /* Φωτεινή πλευρά (Φροντίδα - LLM) */
                        .flip-card-back {{
                            background: linear-gradient(135deg, #ffffff 0%, #f3f0fb 100%);
                            color: #4A3D73; border: 2px solid #b19cd9;
                            transform: rotateY(180deg);
                        }}
                        .hint {{ font-size: 13px; opacity: 0.6; margin-top: 15px; font-weight: 400; }}
                        .front-text {{ font-size: 19px; font-weight: 500; font-style: italic; }}
                        .back-text {{ font-size: 16px; line-height: 1.6; font-weight: 500; }}
                    </style>
                </head>
                <body>
                    <div class="flip-card" onclick="this.querySelector('.flip-card-inner').classList.toggle('flipped')">
                        <div class="flip-card-inner">
                            <div class="flip-card-front">
                                <div class="front-text">«{safe_thought}»</div>
                                <div class="hint">{t("ex7_hint_wait")}</div>
                            </div>
                            <div class="flip-card-back">
                                <div class="back-text">{safe_msg}</div>
                                <div class="hint">{t("ex7_flip_hint")}</div>
                            </div>
                        </div>
                    </div>
                    
                    <script>
                        // Αυτόματο γύρισμα μετά από 1.2 δευτερόλεπτα για να αποκαλυφθεί η απάντηση!
                        setTimeout(() => {{
                            const card = document.querySelector('.flip-card-inner');
                            card.classList.add('flipped');
                            document.querySelector('.flip-card-front .hint').innerText = "{t("ex7_hint_click")}";
                        }}, 1200);
                    </script>
                </body>
                </html>
                """,
                height=260,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΑΞΙΟΛΟΓΗΣΗ TAB (PHQ-9 / GAD-7)
# ============================================================

with tab_screening:
    _scr_lang = get_lang()
    _scr_intro = SCREENING_INTRO if _scr_lang == "el" else SCREENING_INTRO_EN
    _scr_answer_options = ANSWER_OPTIONS if _scr_lang == "el" else ANSWER_OPTIONS_EN
    _scr_phq9_questions = PHQ9_QUESTIONS if _scr_lang == "el" else PHQ9_QUESTIONS_EN
    _scr_gad7_questions = GAD7_QUESTIONS if _scr_lang == "el" else GAD7_QUESTIONS_EN

    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("screening_page_title")}</h1>
          <p class='page-subtitle'>
            {t("screening_page_sub")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    screening_choice = st.radio(
        t("screening_choice_label"),
        [t("screening_choice_phq9"), t("screening_choice_gad7")],
        horizontal=True,
        key="screening_choice",
    )

    st.markdown("---")
    st.markdown(f"**{_scr_intro}**")
    st.caption(t("screening_caption"))

    answer_labels = [label for _, label in _scr_answer_options]

    if screening_choice == t("screening_choice_phq9"):
        phq9_answers: list[int] = []
        for i, q in enumerate(_scr_phq9_questions):
            chosen_label = st.radio(
                f"{i + 1}. {q}",
                options=answer_labels,
                key=f"phq9_q{i}",
                horizontal=True,
            )
            score_val = next(v for v, label in _scr_answer_options if label == chosen_label)
            phq9_answers.append(score_val)

        if st.button(t("screening_submit_phq9"), key="submit_phq9"):
            total, severity_label, flagged = score_phq9(phq9_answers)

            # ΚΡΙΣΙΜΟ: η ερώτηση αυτοτραυματισμού ΠΟΤΕ δεν αντιμετωπίζεται
            # απλά ως "πόντοι" -- ενεργοποιεί άμεσα το ίδιο emergency
            # protocol που έχει ήδη η εφαρμογή στο Chat tab.
            if flagged:
                emergency_html = emergency_message()
                st.session_state.messages.append(("emergency", emergency_html))
                render_emergency_block(emergency_html)

                log_checkin(None, None, None, "PHQ-9 item 9 flagged", is_emergency=True)

            save_screening_result("phq9", total, severity_label, phq9_answers, flagged_item9=flagged)

            st.markdown(
                f"""
                <div class="reframe-card">
                  <div class="reframe-new-title">{t("screening_result_phq9")}</div>
                  <div class="reframe-new-body"><strong>{total}/27</strong> — {severity_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not flagged:
                st.info(t("screening_info"))

    else:
        gad7_answers: list[int] = []
        for i, q in enumerate(_scr_gad7_questions):
            chosen_label = st.radio(
                f"{i + 1}. {q}",
                options=answer_labels,
                key=f"gad7_q{i}",
                horizontal=True,
            )
            score_val = next(v for v, label in _scr_answer_options if label == chosen_label)
            gad7_answers.append(score_val)

        if st.button(t("screening_submit_gad7"), key="submit_gad7"):
            total, severity_label = score_gad7(gad7_answers)
            save_screening_result("gad7", total, severity_label, gad7_answers, flagged_item9=False)

            st.markdown(
                f"""
                <div class="reframe-card">
                  <div class="reframe-new-title">{t("screening_result_gad7")}</div>
                  <div class="reframe-new-body"><strong>{total}/21</strong> — {severity_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(t("screening_info"))

    st.markdown("---")
    st.markdown(f"### {t('screening_history_heading')}")
    screening_history = load_screening_history(limit=10)
    if screening_history:
        for h in screening_history:
            ts = (h.get("created_at") or "")[:16].replace("T", " ")
            type_label = t("screening_type_phq9") if h.get("screening_type") == "phq9" else t("screening_type_gad7")
            st.markdown(
                f"- **{type_label}** — {h.get('total_score')} · {h.get('severity_label')} · {ts}"
            )
    else:
        st.caption(t("screening_history_empty"))

    st.markdown(
        f"""
        <p class="footer-disclaimer">
          {t("screening_footer_disclaimer")}
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# THOUGHT REFRAME STUDIO TAB
# ============================================================

with tab_reframe:
    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("reframe_page_title")}</h1>
          <p class='page-subtitle'>
            {t("reframe_page_sub")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    st.markdown(t("reframe_intro"), unsafe_allow_html=True)

    # Γλωσσο-ανεξάρτητα "κλειδιά" μοτίβων -- το selectbox δείχνει τη
    # μεταφρασμένη ετικέτα, αλλά εσωτερικά δουλεύουμε με το key (ίδιο
    # και στα Ελληνικά και στα Αγγλικά), ώστε το rule-based fallback
    # στο reframe.py να δουλεύει σωστά ανεξαρτήτως γλώσσας.
    _pattern_keys = ["catastrophizing", "allornothing", "mindreading", "overgeneralization", "other"]
    _pattern_labels = {
        "catastrophizing": t("reframe_pattern_catastrophizing"),
        "allornothing": t("reframe_pattern_allornothing"),
        "mindreading": t("reframe_pattern_mindreading"),
        "overgeneralization": t("reframe_pattern_overgeneralization"),
        "other": t("reframe_pattern_other"),
    }
    _pattern_explanations = {
        "catastrophizing": t("reframe_explain_catastrophizing"),
        "allornothing": t("reframe_explain_allornothing"),
        "mindreading": t("reframe_explain_mindreading"),
        "overgeneralization": t("reframe_explain_overgeneralization"),
        "other": t("reframe_explain_other"),
    }

    pattern_key = st.selectbox(
        t("reframe_pattern_label"),
        _pattern_keys,
        format_func=lambda k: _pattern_labels[k],
    )
    pattern = _pattern_labels[pattern_key]

    with st.expander(t("reframe_explanation_expander")):
        st.write(_pattern_explanations.get(pattern_key, ""))

    intensity = st.slider(t("reframe_intensity_label"), min_value=1, max_value=10, value=7)

    thought_input = st.text_area(
        t("reframe_thought_label"),
        height=90,
        key="reframe_thought",
    )

    if st.button(t("reframe_button"), key="reframe_button"):
        if not (thought_input or "").strip():
            st.warning(t("reframe_warning_empty"))
        else:
            with st.spinner(t("reframe_spinner")):
                reframe_profile = load_profile()
                reframed, used_techniques, used_personal = generate_reframe_rag(
                    thought=thought_input,
                    pattern=pattern,
                    intensity=intensity,
                    profile=reframe_profile,
                    pattern_key=pattern_key,
                    lang=get_lang(),
                )

            full_reframe = (
                t("reframe_weighs_intro", intensity=intensity)
                + reframed
            )

            st.session_state.reframes_history.append(
                {
                    "pattern": pattern,
                    "thought": thought_input,
                    "reframe": full_reframe,
                    "intensity": intensity,
                }
            )

            st.markdown(
                f"""
                <div class="reframe-card">
                  <div class="reframe-original-title">{t("reframe_original_thought")}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='reframe-original-body'>{html.escape(thought_input)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='reframe-new-title'>{t('reframe_gentler_version')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='reframe-new-body'>{full_reframe}</div></div>",
                unsafe_allow_html=True,
            )

            if used_techniques or used_personal:
                with st.expander(t("reframe_sources_expander")):
                    if used_techniques:
                        st.markdown(t("reframe_sources_techniques"))
                        for tech in used_techniques:
                            sim = tech.get("similarity", 0)
                            st.markdown(f"- _{tech.get('content','')}_  ({sim:.2f})")
                    if used_personal:
                        st.markdown(t("reframe_sources_personal"))
                        for p in used_personal:
                            sim = p.get("similarity", 0)
                            st.markdown(f"- «{p.get('thought','')}» ({sim:.2f})")

    if st.session_state.reframes_history:
        st.markdown("---")
        st.markdown(f"### {t('reframe_history_heading')}")

        for item in reversed(st.session_state.reframes_history[-5:]):
            orig = html.escape(item["thought"])
            pat = item["pattern"]
            inten = item["intensity"]
            ref = item["reframe"]

            st.markdown(
                f"""
                <div class="history-card">
                  <div class="history-card-header">
                    <span class="history-date">{t("reframe_history_pattern")}: {pat}</span>
                    <span class="history-mood-pill mood-mid">{t("reframe_history_intensity")}: {inten}/10</span>
                  </div>
                  <div class="history-message"><strong>{t("reframe_history_thought")}:</strong> «{orig}»</div>
                  <div class="history-message" style="margin-top:0.3rem;"><strong>{t("reframe_history_new")}:</strong> {ref}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ΠΡΟΦΙΛ TAB (Interactive Glass Dashboard - Minimal)
# ============================================================

with tab_profile:
    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("profile_page_title")}</h1>
          <p class='page-subtitle'>{t("profile_page_sub")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = load_profile()

    # --- ΚΕΝΤΡΙΚΗ ΠΕΡΙΟΧΗ (Glass Wrapper) ---
    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)
    
    # Ενότητα 1: Ταυτότητα
    st.markdown(f"<h3 style='color: #4A3D73; font-size: 1.1rem; border-bottom: 2px solid rgba(212, 190, 250, 0.3); padding-bottom: 8px; margin-top: 0;'>{t('profile_section_identity')}</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        name = st.text_input(t("profile_name"), value=profile.get("name", ""))
    with col2:
        stored_age = profile.get("age", 0)
        try:
            stored_age = int(stored_age)
        except:
            stored_age = 0
        age = st.number_input(t("profile_age"), min_value=0, max_value=120, value=stored_age, step=1)
    with col3:
        context = st.text_input(t("profile_context"), value=profile.get("context", ""))

    age_range = infer_age_range(age) if age else profile.get("age_range", "")

    # Ενότητα 2: Εσωτερικός Κόσμος
    st.markdown(f"<h3 style='color: #4A3D73; font-size: 1.1rem; border-bottom: 2px solid rgba(212, 190, 250, 0.3); padding-bottom: 8px; margin-top: 30px;'>{t('profile_section_goals')}</h3>", unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        main_goals = st.text_area(t("profile_main_goals"), value=profile.get("main_goals", ""), height=120)
    with col_g2:
        main_struggles = st.text_area(t("profile_main_struggles"), value=profile.get("main_struggles", ""), height=120)
    
    helpful_things = st.text_area(t("profile_helpful_things"), value=profile.get("helpful_things", ""), height=80)

    # Ενότητα 3: Προτιμήσεις AI
    st.markdown(f"<h3 style='color: #4A3D73; font-size: 1.1rem; border-bottom: 2px solid rgba(212, 190, 250, 0.3); padding-bottom: 8px; margin-top: 30px;'>{t('profile_section_ai')}</h3>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        preferred_tone = st.text_area(t("profile_preferred_tone"), value=profile.get("preferred_tone", ""), height=100)
    with col_p2:
        triggers = st.text_area(t("profile_triggers"), value=profile.get("triggers", ""), height=100)
    
    soothing_things = st.text_area(t("profile_soothing_things"), value=profile.get("soothing_things", ""), height=80)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- ΚΟΥΜΠΙ ΑΠΟΘΗΚΕΥΣΗΣ ΚΕΝΤΡΑΡΙΣΜΕΝΟ ---
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button(t("profile_save_button"), use_container_width=True):
            new_profile = {
                "name": name,
                "age": int(age) if age else 0,
                "age_range": age_range,
                "context": context,
                "main_goals": main_goals,
                "main_struggles": main_struggles,
                "helpful_things": helpful_things,
                "preferred_tone": preferred_tone,
                "triggers": triggers,
                "soothing_things": soothing_things,
                # Διατηρούμε τα υπάρχοντα, μη-ορατά στη φόρμα, πεδία (γλώσσα, onboarding flag)
                "language": profile.get("language", get_lang()),
                "onboarding_completed": profile.get("onboarding_completed", False),
            }
            save_profile(new_profile)

            # Συγχρονίζουμε και το όνομα στην κορυφή/sidebar
            if name.strip():
                st.session_state.user_name = name.strip()

            st.toast(t("profile_saved_toast"), icon="✅")

            st.rerun() 
            
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ΣΥΝΔΕΣΗ ΑΠΟ ΚΙΝΗΤΟ TAB (Interactive Scanner UI)
# ============================================================

with tab_mobile:
    import base64
    from io import BytesIO

    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("mobile_page_title")}</h1>
          <p class='page-subtitle'>
            {t("mobile_page_sub")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Δημιουργία URL και QR Image
    local_url = get_local_url(port=8501)
    qr_img = generate_qr_image(local_url)
    
    # Μετατροπή της εικόνας QR σε Base64 για ενσωμάτωση στην HTML
    buffered = BytesIO()
    qr_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Το Interactive QR Scanner Component
    components.html(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0; padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .qr-container {{
                background: rgba(255, 255, 255, 0.45);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid rgba(255, 255, 255, 0.7);
                border-radius: 24px;
                padding: 40px;
                max-width: 800px;
                width: 100%;
                box-shadow: 0 15px 40px rgba(108, 90, 158, 0.08);
                display: flex;
                gap: 40px;
                align-items: center;
                animation: cardFadeIn 0.8s ease-out;
            }}
            @media (max-width: 768px) {{
                .qr-container {{ flex-direction: column; text-align: center; padding: 30px 20px; gap: 25px; }}
            }}
            
            /* Το Κουτί του QR με το Animation Σάρωσης */
            .qr-box-wrapper {{
                position: relative;
                background: #ffffff;
                padding: 15px;
                border-radius: 20px;
                box-shadow: 0 10px 25px rgba(108, 90, 158, 0.15);
                flex-shrink: 0;
                overflow: hidden;
            }}
            .qr-box-wrapper img {{
                display: block;
                width: 180px;
                height: 180px;
                border-radius: 10px;
            }}
            /* Η φωτεινή "Laser" γραμμή */
            .scanner-laser {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: rgba(183, 157, 242, 0.9);
                box-shadow: 0 0 15px rgba(183, 157, 242, 0.9), 0 0 30px rgba(183, 157, 242, 0.6);
                animation: scan 2.5s infinite alternate ease-in-out;
            }}
            @keyframes scan {{
                0% {{ top: 5%; opacity: 0; }}
                10% {{ opacity: 1; }}
                90% {{ opacity: 1; }}
                100% {{ top: 95%; opacity: 0; }}
            }}
            @keyframes cardFadeIn {{
                from {{ opacity: 0; transform: translateY(15px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            /* Περιοχή κειμένου & κουμπιών */
            .info-area {{ flex-grow: 1; }}
            
            .badge {{
                display: inline-block; padding: 6px 14px; border-radius: 20px;
                font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
                background: linear-gradient(135deg, #e4d4ff, #f2eaff); color: #4A3D73;
                margin-bottom: 15px; border: 1px solid rgba(183, 157, 242, 0.5);
                box-shadow: 0 4px 10px rgba(183, 157, 242, 0.2);
            }}
            h2 {{ margin: 0 0 10px 0; color: #3b304c; font-size: 24px; }}
            p {{ color: #73658a; font-size: 15px; line-height: 1.6; margin: 0 0 20px 0; }}
            
            /* Input Group για Αντιγραφή του URL */
            .copy-group {{
                display: flex; background: rgba(245, 247, 251, 0.6);
                border: 1px solid rgba(183, 157, 242, 0.4); border-radius: 12px;
                overflow: hidden; backdrop-filter: blur(5px); transition: all 0.3s;
            }}
            .copy-group:focus-within {{
                border-color: #b79df2; box-shadow: 0 0 0 2px rgba(183, 157, 242, 0.2);
            }}
            .copy-input {{
                flex-grow: 1; background: transparent; border: none; padding: 12px 15px;
                color: #4A3D73; font-family: monospace; font-size: 14px; outline: none;
            }}
            .copy-btn {{
                background: #e4d4ff; color: #4A3D73; border: none; padding: 0 20px;
                font-weight: 600; cursor: pointer; transition: all 0.3s;
            }}
            .copy-btn:hover {{ background: #d4befa; }}
            .copy-btn.success {{ background: #baf7c3; color: #158047; }}
        </style>
    </head>
    <body>
        <div class="qr-container">
            <div class="qr-box-wrapper">
                <img src="data:image/png;base64,{img_str}" alt="QR Code">
                <div class="scanner-laser"></div>
            </div>
            
            <div class="info-area">
                <div class="badge">{t("mobile_badge")}</div>
                <h2>{t("mobile_heading")}</h2>
                <p>
                    {t("mobile_body")}
                </p>
                
                <div class="copy-group">
                    <input type="text" class="copy-input" id="urlInput" value="{local_url}" readonly>
                    <button class="copy-btn" id="copyBtn" onclick="copyUrl()">{t("mobile_copy_button")}</button>
                </div>
            </div>
        </div>
        
        <script>
            // Λειτουργία JavaScript για το κουμπί Αντιγραφής
            function copyUrl() {{
                const input = document.getElementById('urlInput');
                const btn = document.getElementById('copyBtn');
                
                // Επιλογή του κειμένου & αντιγραφή στο πρόχειρο
                input.select();
                document.execCommand('copy');
                
                // Αλλαγή στυλ του κουμπιού σε "Επιτυχία"
                btn.innerHTML = '{t("mobile_copied_button")}';
                btn.classList.add('success');
                
                // Επαναφορά στην αρχική μορφή μετά από 2 δευτερόλεπτα
                setTimeout(() => {{
                    btn.innerHTML = '{t("mobile_copy_button")}';
                    btn.classList.remove('success');
                }}, 2000);
            }}
        </script>
    </body>
    </html>
    """, height=380)


# ============================================================
# ΣΧΕΤΙΚΑ & ΑΣΦΑΛΕΙΑ TAB (Glassmorphism Dashboard)
# ============================================================

with tab_info:
    st.markdown(
        f"""
        <div class="page-header">
          <h1 class='page-title'>{t("info_page_title")}</h1>
          <p class='page-subtitle'>{t("info_page_sub")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

    # Χρησιμοποιούμε 2 στήλες για πιο σύγχρονο, Dashboard layout
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.4s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">{t("info_how_it_works_title")}</h3>
<p style="color: #73658a; line-height: 1.6; font-size: 0.95rem;">
{t("info_how_it_works_body")}
</p>
</div>""", 
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.6s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">{t("info_safety_title")}</h3>
<p style="color: #73658a; line-height: 1.6; font-size: 0.95rem;">
{t("info_safety_body")}
</p>
</div>""", 
            unsafe_allow_html=True
        )

    with col_i2:
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(108, 90, 158, 0.08); animation: cardFadeIn 0.5s ease-out;">
<h3 style="color: #4A3D73; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">{t("info_privacy_title")}</h3>
<p style="color: #73658a; line-height: 1.6; font-size: 0.95rem;">
{t("info_privacy_body")}
</p>
</div>""", 
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Κάρτα Γραμμών Βοήθειας (Πιο έντονο ροζ/κόκκινο πλαίσιο για να τραβάει την προσοχή)
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid #ffcad4; border-radius: 16px; padding: 25px; height: 100%; box-shadow: 0 12px 40px rgba(255, 105, 180, 0.12); border-left: 5px solid #ff758f; animation: cardFadeIn 0.7s ease-out;">
<h3 style="color: #c12530; font-size: 1.2rem; margin-top: 0; display: flex; align-items: center; gap: 8px;">{t("info_helplines_title")}</h3>
<ul style="color: #4A3D73; line-height: 1.8; font-size: 0.95rem; padding-left: 20px; font-weight: 500;">
    <li>{t("info_helpline_112")}</li>
    <li>{t("info_helpline_1018")}</li>
    <li>{t("info_helpline_10306")}</li>
    <li>{t("info_helpline_1056")}</li>
</ul>
</div>""", 
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Disclaimer Box στο κάτω μέρος
    st.markdown(
        f"""<div style="text-align: center; color: #73658a; font-size: 0.85rem; padding: 20px; background: rgba(245, 247, 251, 0.6); border-radius: 12px; border: 1px solid rgba(183, 157, 242, 0.3);">
{t("info_disclaimer")}
</div>""", 
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)