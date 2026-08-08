"""
i18n.py
=======
Ελαφρύ σύστημα μετάφρασης (Ελληνικά / English). Κάθε string της
εφαρμογής έχει ένα "κλειδί" (key) και μια μετάφραση σε κάθε γλώσσα.

Η γλώσσα του χρήστη αποθηκεύεται μέσα στο ίδιο του το profile στη
Supabase (profile["language"]), οπότε τη θυμάται η εφαρμογή και σε
επόμενα logins / άλλες συσκευές -- δεν χρειάστηκε νέος πίνακας.

Χρήση:
    from i18n import t, get_lang, set_lang
    st.write(t("chat_page_title"))
"""

from typing import Any, Dict, List, Tuple

import streamlit as st

DEFAULT_LANG = "el"
SUPPORTED_LANGS = ["el", "en"]


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "el": {
        # --- Chrome / γενικά ---
        "app_title": "🧠 Project Wellness",
        "app_subtitle_greeting": "Καλωσήρθες, {name} · μικρά συναισθηματικά check-ins και μικρά αλλά σταθερά βήματα φροντίδας.",
        "app_subtitle_default": "Μικρά συναισθηματικά check-ins, μικρά βήματα φροντίδας.",
        "language_label": "Γλώσσα",
        "logout_button": "🚪 Αποσύνδεση",

        # --- Tabs ---
        "tab_chat": "Chat",
        "tab_history": "Ιστορικό",
        "tab_stats": "Στατιστικά",
        "tab_exercises": "Ασκήσεις",
        "tab_screening": "🩺 Αξιολόγηση",
        "tab_reframe": "Thought Reframe Studio",
        "tab_profile": "Προφίλ",
        "tab_mobile": "Σύνδεση από κινητό",
        "tab_info": "Σχετικά & Ασφάλεια",

        # --- Login / Signup ---
        "login_hero_title": "Project Wellness",
        "login_hero_desc": (
            "Μικρά, καθημερινά check-ins για τη διάθεση, τον ύπνο και την ενυδάτωσή σου — "
            "σαν ένα προσωπικό δελτίο καιρού για το πώς νιώθεις. Μια ήρεμη στιγμή παύσης, κάθε μέρα."
        ),
        "login_form_title": "Καλώς ήρθες",
        "login_form_sub": (
            "Συνδέσου ή δημιούργησε λογαριασμό για να συνεχίσουμε. "
            "Τα δεδομένα σου προστατεύονται με πραγματικό λογαριασμό (email + κωδικός)."
        ),
        "login_tab_login": "🔑 Σύνδεση",
        "login_tab_signup": "🆕 Εγγραφή",
        "field_email": "Email",
        "field_password": "Κωδικός",
        "field_password_min": "Κωδικός (τουλάχιστον 6 χαρακτήρες)",
        "field_password_confirm": "Επιβεβαίωση κωδικού",
        "field_username": "Όνομα χρήστη",
        "btn_login": "Σύνδεση",
        "btn_signup": "Δημιουργία λογαριασμού",
        "err_fill_both": "Συμπλήρωσε email και κωδικό.",
        "err_login_failed": "⚠️ Αποτυχία σύνδεσης: {err}",
        "err_signup_failed": "⚠️ Αποτυχία εγγραφής: {err}",
        "err_need_name": "Γράψε ένα όνομα χρήστη 🙂",
        "err_need_email": "Χρειάζομαι ένα έγκυρο email.",
        "err_password_length": "Ο κωδικός πρέπει να έχει τουλάχιστον 6 χαρακτήρες.",
        "err_password_mismatch": "Οι κωδικοί δεν ταιριάζουν.",
        "success_login": "Καλωσήρθες πίσω!",
        "success_signup": "Καλώς ήρθες, {name}! Ο λογαριασμός σου δημιουργήθηκε.",
        "signup_confirm_email": (
            "📧 Σχεδόν έτοιμοι! Σου στείλαμε ένα email επιβεβαίωσης στο **{email}**. "
            "Άνοιξέ το και πάτησε το link για να ενεργοποιηθεί ο λογαριασμός σου, και μετά "
            "επίστρεψε εδώ για να συνδεθείς."
        ),

        # --- Onboarding ---
        "onboarding_greeting": "Καλωσήρθες{name_part}!",
        "onboarding_sub": "Πριν ξεκινήσουμε, μια πολύ σύντομη ξενάγηση σε ό,τι θα βρεις εδώ μέσα.",
        "onboarding_tabs_heading": "🗺️ Οι καρτέλες της εφαρμογής",
        "onboarding_tab_chat": "💬 <strong>Chat</strong> — μίλησέ μου ελεύθερα για τη μέρα σου, κρατάμε το νήμα της συζήτησης",
        "onboarding_tab_history": "📁 <strong>Ιστορικό</strong> — μια ήρεμη ματιά στις προηγούμενες καταγραφές σου",
        "onboarding_tab_stats": "📊 <strong>Στατιστικά</strong> — τάσεις διάθεσης / ύπνου / νερού με τον καιρό, και το streak σου",
        "onboarding_tab_exercises": "🧘 <strong>Ασκήσεις</strong> — μικρές, διαδραστικές ασκήσεις φροντίδας",
        "onboarding_tab_screening": "🩺 <strong>Αξιολόγηση</strong> — δύο σύντομα, επιστημονικά εργαλεία (PHQ-9 / GAD-7)",
        "onboarding_tab_reframe": "🧩 <strong>Thought Reframe Studio</strong> — βοήθεια στο να δεις δύσκολες σκέψεις από πιο τρυφερή οπτική",
        "onboarding_safety": (
            "<strong>Μια σημαντική σημείωση:</strong> αυτή η εφαρμογή είναι εργαλείο αυτοβοήθειας, "
            "δεν αντικαθιστά ψυχολόγο ή ψυχίατρο. Αν βρεθείς σε κίνδυνο, κάλεσε το "
            "<strong>112</strong> ή τη Γραμμή Παρέμβασης <strong>1018</strong>."
        ),
        "onboarding_button": "🚀 Ξεκινάμε",

        # --- Chat tab ---
        "chat_page_title": "Wellness Edition",
        "chat_page_sub": "Μίλησέ μου για τη μέρα σου – θα κρατήσουμε συνοχή και στο τέλος θα κάνουμε ήρεμο wrap-up.",
        "chat_locked_title": "🔒 Το Chat είναι κλειδωμένο",
        "chat_locked_body1": "Για να προχωρήσεις, συμπλήρωσε πρώτα το προφίλ σου.",
        "chat_locked_body2": "Πήγαινε στην καρτέλα «Προφίλ» και συμπλήρωσε τουλάχιστον:",
        "chat_locked_li1": "ένα όνομα ή ψευδώνυμο",
        "chat_locked_li2": "ή ένα πλαίσιο ζωής (π.χ. φοιτητής, εργαζόμενος)",
        "chat_locked_body3": "Μετά το “Αποθήκευση προφίλ”, το Chat ξεκλειδώνει.",
        "checkin_expander": "📊 Σημερινό Check-in (Διάθεση, Ύπνος, Νερό)",
        "mood_label": "Διάθεση (1–10)",
        "sleep_label": "Ύπνος (ώρες, 1–10)",
        "water_label": "Νερό (ποτήρια, 1–15)",
        "mood_metric": "Διάθεση",
        "sleep_metric": "Ύπνος",
        "water_metric": "Νερό",
        "new_conversation": "🔄 Νέα Συνομιλία",
        "chat_input_placeholder": "📝 Γράψε μου ό,τι θέλεις για τη μέρα σου:",
        "chat_how_are_you": "😊 Πώς είσαι σήμερα;",
        "chat_footer_disclaimer": (
            "Το Project Wellness είναι εργαλείο αυτοβοήθειας και ψυχοεκπαιδευτικού χαρακτήρα. "
            "Δεν αντικαθιστά ψυχολόγο, ψυχίατρο ή υπηρεσίες έκτακτης ανάγκης. "
            "Αν βρίσκεσαι σε κίνδυνο, κάλεσε το 112 ή τη Γραμμή Παρέμβασης 1018."
        ),
        "decision_trace_label": "ℹ️ Explainable layer (τι έγινε σε αυτό το turn)",

        # --- Screening tab ---
        "screening_page_title": "🩺 Αξιολόγηση",
        "screening_page_sub": (
            "Δύο σύντομα, επιστημονικά επικυρωμένα ερωτηματολόγια αυτο-αναφοράς "
            "(PHQ-9 για κατάθλιψη, GAD-7 για άγχος). Δεν αποτελούν διάγνωση — "
            "είναι μια δομημένη ένδειξη που μπορεί να σε βοηθήσει να καταλάβεις "
            "καλύτερα πώς νιώθεις τις τελευταίες 2 εβδομάδες."
        ),
        "screening_choice_label": "Ποιο θέλεις να συμπληρώσεις;",
        "screening_choice_phq9": "Κατάθλιψη (PHQ-9)",
        "screening_choice_gad7": "Άγχος (GAD-7)",
        "screening_caption": "Απάντησε όσο πιο ειλικρινά μπορείς — δεν υπάρχουν σωστές/λάθος απαντήσεις.",
        "screening_submit_phq9": "Υποβολή PHQ-9 ✨",
        "screening_submit_gad7": "Υποβολή GAD-7 ✨",
        "screening_result_phq9": "Αποτέλεσμα PHQ-9",
        "screening_result_gad7": "Αποτέλεσμα GAD-7",
        "screening_info": (
            "Αυτό είναι ένα εργαλείο screening, όχι διάγνωση. Αν σε ανησυχεί το "
            "αποτέλεσμα, μια συζήτηση με ψυχολόγο ή ψυχίατρο είναι καλή ιδέα."
        ),
        "screening_history_heading": "📈 Ιστορικό αξιολογήσεων",
        "screening_history_empty": "Δεν υπάρχουν ακόμη καταγραφές αξιολόγησης.",
        "screening_type_phq9": "PHQ-9 (κατάθλιψη)",
        "screening_type_gad7": "GAD-7 (άγχος)",
        "screening_footer_disclaimer": (
            "Τα PHQ-9 και GAD-7 είναι δημόσια διαθέσιμα εργαλεία screening, ευρέως "
            "χρησιμοποιούμενα στην κλινική πράξη. Δεν αντικαθιστούν κλινική αξιολόγηση "
            "από ειδικό. Αν βρίσκεσαι σε κίνδυνο, κάλεσε το 112 ή τη Γραμμή Παρέμβασης 1018."
        ),
        "severity_minimal": "Ελάχιστη ένδειξη",
        "severity_mild": "Ήπια ένδειξη",
        "severity_moderate": "Μέτρια ένδειξη",
        "severity_mod_severe": "Μέτρια προς σοβαρή ένδειξη",
        "severity_severe": "Σοβαρή ένδειξη",

        # --- Thought Reframe Studio ---
        "reframe_page_title": "🧩 Thought Reframe Studio",
        "reframe_page_sub": "Ένας μικρός χώρος για να μετασχηματίζεις δύσκολες σκέψεις σε πιο τρυφερές, ρεαλιστικές.",
        "reframe_intro": (
            "Σημείωσε μια σκέψη που σε ζορίζει και επίλεξε τον τρόπο σκέψης που της ταιριάζει περισσότερο. "
            "Ο στόχος δεν είναι να ακυρώσουμε το συναίσθημα, αλλά να βρούμε μια φράση πιο επιεική με εσένα."
        ),
        "reframe_pattern_label": "Τι μοιάζει περισσότερο με αυτή τη σκέψη;",
        "reframe_pattern_catastrophizing": "Καταστροφολογία (τα βλέπω όλα χάλια)",
        "reframe_pattern_allornothing": "Όλα ή τίποτα (είτε τέλειο είτε αποτυχία)",
        "reframe_pattern_mindreading": "Διάβασμα σκέψης (υποθέτω τι σκέφτονται οι άλλοι)",
        "reframe_pattern_overgeneralization": "Υπεργενίκευση (μια εμπειρία = πάντα έτσι)",
        "reframe_pattern_other": "Άλλο / δεν είμαι σίγουρος",
        "reframe_intensity_label": "Πόσο σε βαραίνει αυτή η σκέψη αυτή τη στιγμή;",
        "reframe_thought_label": "✏️ Δύσκολη σκέψη (όπως ακριβώς σου έρχεται στο μυαλό):",
        "reframe_button": "Μετασχηματισμός σκέψης ✨",
        "reframe_warning_empty": "Γράψε πρώτα τη σκέψη που σε δυσκολεύει 📝",
        "reframe_spinner": "Ψάχνω σχετικές τεχνικές και συνθέτω μια απάντηση...",
        "reframe_sources_expander": "🔎 Πηγές που χρησιμοποιήθηκαν (RAG)",
        "reframe_sources_techniques": "**Σχετικές τεχνικές από τη βιβλιοθήκη γνώσης:**",
        "reframe_sources_personal": "**Σχετικές προηγούμενες σκέψεις σου:**",
        "reframe_explanation_expander": "🔍 Μικρή εξήγηση για αυτό το μοτίβο",
        "reframe_explain_catastrophizing": "Ο νους πηγαίνει στο χειρότερο σενάριο, σαν να είναι βέβαιο ότι όλα θα πάνε στραβά.",
        "reframe_explain_allornothing": "Άσπρο–μαύρο: ή τέλειο ή αποτυχία, χωρίς τις ενδιάμεσες αποχρώσεις.",
        "reframe_explain_mindreading": "Υποθέτεις αρνητική κρίση των άλλων χωρίς πραγματικές αποδείξεις.",
        "reframe_explain_overgeneralization": "Μια εμπειρία γίνεται «απόδειξη» ότι πάντα έτσι θα είναι.",
        "reframe_explain_other": "Δεν ταιριάζει καθαρά σε ένα μοτίβο. Είναι ΟΚ.",
        "reframe_original_thought": "Αρχική σκέψη",
        "reframe_gentler_version": "Πιο φροντιστική εκδοχή",
        "reframe_weighs_intro": "Βλέπω ότι αυτή η σκέψη σε βαραίνει περίπου **{intensity}/10** αυτή τη στιγμή.\n\n",
        "reframe_history_heading": "📚 Μικρό ιστορικό μετασχηματισμών",
        "reframe_history_pattern": "Μοτίβο",
        "reframe_history_intensity": "Ένταση",
        "reframe_history_thought": "Σκέψη",
        "reframe_history_new": "Νέα εκδοχή",

        # --- LLM language directive ---
        "llm_lang_directive": "Απάντησε αυστηρά στα Ελληνικά.",
    },

    "en": {
        # --- Chrome / general ---
        "app_title": "🧠 Project Wellness",
        "app_subtitle_greeting": "Welcome, {name} · small emotional check-ins and small, steady steps of self-care.",
        "app_subtitle_default": "Small emotional check-ins, small steps of self-care.",
        "language_label": "Language",
        "logout_button": "🚪 Log out",

        # --- Tabs ---
        "tab_chat": "Chat",
        "tab_history": "History",
        "tab_stats": "Statistics",
        "tab_exercises": "Exercises",
        "tab_screening": "🩺 Screening",
        "tab_reframe": "Thought Reframe Studio",
        "tab_profile": "Profile",
        "tab_mobile": "Mobile Connection",
        "tab_info": "About & Safety",

        # --- Login / Signup ---
        "login_hero_title": "Project Wellness",
        "login_hero_desc": (
            "Small, daily check-ins for your mood, sleep and hydration — like a personal "
            "weather report for how you feel. A calm pause, every day."
        ),
        "login_form_title": "Welcome",
        "login_form_sub": (
            "Log in or create an account to continue. Your data is protected by a real "
            "account (email + password)."
        ),
        "login_tab_login": "🔑 Log in",
        "login_tab_signup": "🆕 Sign up",
        "field_email": "Email",
        "field_password": "Password",
        "field_password_min": "Password (at least 6 characters)",
        "field_password_confirm": "Confirm password",
        "field_username": "Username",
        "btn_login": "Log in",
        "btn_signup": "Create account",
        "err_fill_both": "Please fill in both email and password.",
        "err_login_failed": "⚠️ Login failed: {err}",
        "err_signup_failed": "⚠️ Sign up failed: {err}",
        "err_need_name": "Please enter a username 🙂",
        "err_need_email": "A valid email is required.",
        "err_password_length": "Password must be at least 6 characters.",
        "err_password_mismatch": "Passwords don't match.",
        "success_login": "Welcome back!",
        "success_signup": "Welcome, {name}! Your account has been created.",
        "signup_confirm_email": (
            "📧 Almost there! We sent a confirmation email to **{email}**. "
            "Open it and click the link to activate your account, then come back "
            "here to log in."
        ),

        # --- Onboarding ---
        "onboarding_greeting": "Welcome{name_part}!",
        "onboarding_sub": "Before we start, a very quick tour of what you'll find here.",
        "onboarding_tabs_heading": "🗺️ App tabs",
        "onboarding_tab_chat": "💬 <strong>Chat</strong> — talk to me freely about your day, we keep the thread of the conversation",
        "onboarding_tab_history": "📁 <strong>History</strong> — a calm look at your past entries",
        "onboarding_tab_stats": "📊 <strong>Statistics</strong> — trends in mood / sleep / water over time, and your streak",
        "onboarding_tab_exercises": "🧘 <strong>Exercises</strong> — small, interactive self-care exercises",
        "onboarding_tab_screening": "🩺 <strong>Screening</strong> — two short, scientific tools (PHQ-9 / GAD-7)",
        "onboarding_tab_reframe": "🧩 <strong>Thought Reframe Studio</strong> — help seeing difficult thoughts from a gentler angle",
        "onboarding_safety": (
            "<strong>An important note:</strong> this app is a self-help tool, it does not "
            "replace a psychologist or psychiatrist. If you're ever at risk, call your local "
            "emergency number, or in Greece <strong>112</strong> or the Suicide Helpline "
            "<strong>1018</strong>."
        ),
        "onboarding_button": "🚀 Let's start",

        # --- Chat tab ---
        "chat_page_title": "Wellness Edition",
        "chat_page_sub": "Tell me about your day – we'll keep continuity and do a calm wrap-up at the end.",
        "chat_locked_title": "🔒 Chat is locked",
        "chat_locked_body1": "To continue, please fill in your profile first.",
        "chat_locked_body2": "Go to the «Profile» tab and fill in at least:",
        "chat_locked_li1": "a name or nickname",
        "chat_locked_li2": "or a life context (e.g. student, employee)",
        "chat_locked_body3": "After “Save profile”, Chat unlocks.",
        "checkin_expander": "📊 Today's Check-in (Mood, Sleep, Water)",
        "mood_label": "Mood (1–10)",
        "sleep_label": "Sleep (hours, 1–10)",
        "water_label": "Water (glasses, 1–15)",
        "mood_metric": "Mood",
        "sleep_metric": "Sleep",
        "water_metric": "Water",
        "new_conversation": "🔄 New Conversation",
        "chat_input_placeholder": "📝 Write anything about your day:",
        "chat_how_are_you": "😊 How are you today?",
        "chat_footer_disclaimer": (
            "Project Wellness is a self-help and psychoeducational tool. It does not replace "
            "a psychologist, psychiatrist, or emergency services. If you are at risk, call your "
            "local emergency number, or in Greece 112 or the Suicide Helpline 1018."
        ),
        "decision_trace_label": "ℹ️ Explainable layer (what happened this turn)",

        # --- Screening tab ---
        "screening_page_title": "🩺 Screening",
        "screening_page_sub": (
            "Two short, scientifically validated self-report questionnaires "
            "(PHQ-9 for depression, GAD-7 for anxiety). They are not a diagnosis — "
            "they are a structured indicator that can help you understand better how "
            "you've been feeling over the past 2 weeks."
        ),
        "screening_choice_label": "Which one would you like to complete?",
        "screening_choice_phq9": "Depression (PHQ-9)",
        "screening_choice_gad7": "Anxiety (GAD-7)",
        "screening_caption": "Answer as honestly as you can — there are no right or wrong answers.",
        "screening_submit_phq9": "Submit PHQ-9 ✨",
        "screening_submit_gad7": "Submit GAD-7 ✨",
        "screening_result_phq9": "PHQ-9 Result",
        "screening_result_gad7": "GAD-7 Result",
        "screening_info": (
            "This is a screening tool, not a diagnosis. If the result concerns you, "
            "talking to a psychologist or psychiatrist is a good idea."
        ),
        "screening_history_heading": "📈 Screening history",
        "screening_history_empty": "No screening entries yet.",
        "screening_type_phq9": "PHQ-9 (depression)",
        "screening_type_gad7": "GAD-7 (anxiety)",
        "screening_footer_disclaimer": (
            "PHQ-9 and GAD-7 are publicly available screening tools, widely used in clinical "
            "practice. They do not replace clinical assessment by a specialist. If you are at "
            "risk, call your local emergency number, or in Greece 112 or the Suicide Helpline 1018."
        ),
        "severity_minimal": "Minimal",
        "severity_mild": "Mild",
        "severity_moderate": "Moderate",
        "severity_mod_severe": "Moderately severe",
        "severity_severe": "Severe",

        # --- Thought Reframe Studio ---
        "reframe_page_title": "🧩 Thought Reframe Studio",
        "reframe_page_sub": "A small space to transform difficult thoughts into gentler, more realistic ones.",
        "reframe_intro": (
            "Write down a thought that's troubling you and choose the thinking pattern that "
            "fits it best. The goal isn't to invalidate the feeling, but to find a phrase "
            "that's kinder to you."
        ),
        "reframe_pattern_label": "What does this thought resemble the most?",
        "reframe_pattern_catastrophizing": "Catastrophizing (everything looks terrible)",
        "reframe_pattern_allornothing": "All-or-nothing (either perfect or a failure)",
        "reframe_pattern_mindreading": "Mind reading (assuming what others think)",
        "reframe_pattern_overgeneralization": "Overgeneralization (one experience = always like this)",
        "reframe_pattern_other": "Other / not sure",
        "reframe_intensity_label": "How much is this thought weighing on you right now?",
        "reframe_thought_label": "✏️ Difficult thought (exactly as it comes to mind):",
        "reframe_button": "Reframe this thought ✨",
        "reframe_warning_empty": "Please write the thought that's troubling you first 📝",
        "reframe_spinner": "Looking for relevant techniques and composing a response...",
        "reframe_sources_expander": "🔎 Sources used (RAG)",
        "reframe_sources_techniques": "**Relevant techniques from the knowledge base:**",
        "reframe_sources_personal": "**Your relevant past thoughts:**",
        "reframe_explanation_expander": "🔍 A short explanation of this pattern",
        "reframe_explain_catastrophizing": "The mind jumps to the worst-case scenario, as if it were certain everything will go wrong.",
        "reframe_explain_allornothing": "Black-and-white: either perfect or a failure, with no shades in between.",
        "reframe_explain_mindreading": "You assume a negative judgment from others without real evidence.",
        "reframe_explain_overgeneralization": "One experience becomes \"proof\" that it will always be this way.",
        "reframe_explain_other": "Doesn't clearly fit one pattern. That's OK.",
        "reframe_original_thought": "Original thought",
        "reframe_gentler_version": "Gentler version",
        "reframe_weighs_intro": "I see this thought is weighing on you about **{intensity}/10** right now.\n\n",
        "reframe_history_heading": "📚 Reframe history",
        "reframe_history_pattern": "Pattern",
        "reframe_history_intensity": "Intensity",
        "reframe_history_thought": "Thought",
        "reframe_history_new": "New version",

        # --- LLM language directive ---
        "llm_lang_directive": "IMPORTANT: Write your entire reply in English, regardless of the language used elsewhere in these instructions.",
    },
}


# ============================================================
# PHQ-9 / GAD-7 -- επίσημα Αγγλικά κείμενα (public domain)
# ============================================================
# Οι Ελληνικές εκδοχές παραμένουν εκεί που ήταν, στο screening.py
# (δική μας απόδοση, όχι επίσημη μετάφραση). Εδώ βάζουμε τις
# επίσημες, δημόσια διαθέσιμες Αγγλικές διατυπώσεις.

SCREENING_INTRO_EN = "Over the last 2 weeks, how often have you been bothered by any of the following problems?"

ANSWER_OPTIONS_EN: List[Tuple[int, str]] = [
    (0, "Not at all"),
    (1, "Several days"),
    (2, "More than half the days"),
    (3, "Nearly every day"),
]

PHQ9_QUESTIONS_EN: List[str] = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed — or the opposite, "
    "being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]

GAD7_QUESTIONS_EN: List[str] = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]


def get_lang() -> str:
    """
    Επιστρέφει την τρέχουσα γλώσσα ("el" ή "en"). Προτεραιότητα:
    1) το cached profile στο session_state (current_profile["language"])
    2) ό,τι έχει επιλεγεί μέσα στο ίδιο session (st.session_state["lang"])
    3) το προεπιλεγμένο (Ελληνικά)
    """
    profile = st.session_state.get("current_profile") or {}
    lang = profile.get("language")
    if lang in SUPPORTED_LANGS:
        return lang
    return st.session_state.get("lang", DEFAULT_LANG)


def set_lang(lang: str) -> None:
    if lang in SUPPORTED_LANGS:
        st.session_state["lang"] = lang


def t(key: str, **kwargs: Any) -> str:
    """Μετάφραση με βάση το key, στην τρέχουσα γλώσσα (fallback σε Ελληνικά)."""
    lang = get_lang()
    template = TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS[DEFAULT_LANG].get(key) or key
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


def lang_directive() -> str:
    """Η γραμμή-οδηγία γλώσσας που προστίθεται σε κάθε system prompt του LLM."""
    return t("llm_lang_directive")