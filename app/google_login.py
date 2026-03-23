import os
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["openid", "email"]


def run_google_login():
    credentials_path = os.path.join(os.path.dirname(__file__), "google_oauth_credentials.json")

    if st.button("🔐 Σύνδεση με Google", use_container_width=True):
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path,
                scopes=SCOPES
            )

            # 🔥 Console-based OAuth (χωρίς localhost browsers)
            auth_url, _ = flow.authorization_url(prompt="consent")

            st.markdown(
                f"""
                ### 🔐 Βήμα 1  
                Πάτησε εδώ για να συνδεθείς στο Google:  
                👉 <a href="{auth_url}" target="_blank">**Σύνδεση Google**</a>

                ### 🔐 Βήμα 2  
                Πάρε τον κωδικό επιβεβαίωσης και βάλ' τον εδώ:
                """,
                unsafe_allow_html=True
            )

            code = st.text_input("Κωδικός Google")

            if st.button("Επιβεβαίωση"):
                flow.fetch_token(code=code)
                creds = flow.credentials

                # Παίρνουμε email από το id_token
                from google.oauth2 import id_token
                from google.auth.transport import requests

                idinfo = id_token.verify_oauth2_token(
                    creds.id_token, requests.Request(), flow.client_config["client_id"]
                )

                user_email = idinfo["email"]
                st.session_state["user_email"] = user_email

                st.success(f"Επιτυχής σύνδεση ως {user_email} ✨")
                st.rerun()

        except Exception as e:
            st.error(f"Σφάλμα: {e}")


def logout():
    if "user_email" in st.session_state:
        del st.session_state["user_email"]
        st.rerun()
