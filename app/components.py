import streamlit as st
import html 

# --------------------------------------------------------
# Βοηθητικό: ένα wrapper για όλο το thread
# (το καλεί έμμεσα το app με διαδοχικά render_* calls)
# --------------------------------------------------------

def render_message(sender, content):
    safe_content = html.escape(content).replace("\n", "<br>")

    if sender == "user":
        bubble = f"""
        <div class="chat-row chat-row-user">
            <div class="chat-bubble chat-user">
                {safe_content}
            </div>
        </div>
        """
    else:
        bubble = f"""
        <div class="chat-row chat-row-bot">
            <div class="chat-bubble chat-bot">
                {safe_content}
            </div>
        </div>
        """

    st.markdown(bubble, unsafe_allow_html=True)


def render_exercise_card(content: str):
    """
    Απαλό λιλά «συννεφάκι» για τη Μικρή Άσκηση.
    """
    html = f"""
    <div class="chat-row chat-row-bot">
        <div class="exercise-bubble">
            <div class="exercise-title">Μικρή άσκηση για εσένα</div>
            <div class="exercise-body">
                {content}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_emergency_block(content: str):
    """
    Έντονο block για emergency μήνυμα.
    """
    html = f"""
    <div class="chat-row chat-row-bot">
        <div class="emergency-block">
            {content}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_action_plan_card(content: str):
    """
    Κάρτα για action plans (άγχος σπουδών, ύπνος κλπ).
    """
    html = f"""
    <div class="chat-row chat-row-bot">
        <div class="plan-card">
            <div class="plan-title">Μικρό πλάνο δράσης</div>
            <div class="plan-body">
                {content}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
