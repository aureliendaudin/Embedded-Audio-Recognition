import time
import numpy as np
import streamlit as st

from raspberry_pi_inference_no_button import SpeechCommandRecognizer  # importe ta classe

# ---------- init modèle (en cache pour éviter re-load à chaque run) ----------
@st.cache_resource
def load_recognizer():
    return SpeechCommandRecognizer()

recognizer = load_recognizer()

st.set_page_config(page_title="Voice Command Demo", page_icon="🎙️", layout="centered")

st.title("Commande vocale – Démo Raspberry Pi")
st.write("Clique sur **Enregistrer** puis dis `on`, `off`, `left`, `right`, `up`, `down` vers le micro de la Pi.")

# état Streamlit pour stocker le dernier label
if "last_label" not in st.session_state:
    st.session_state.last_label = None
if "last_score" not in st.session_state:
    st.session_state.last_score = None

# ---------- bouton d'enregistrement (micro sur la Pi) ----------
if st.button("🎙️ Enregistrer (Pi)"):
    with st.spinner("Enregistrement en cours..."):
        audio = recognizer.record_once()          # utilise ta méthode existante
        label, score = recognizer.classify(audio)

    st.session_state.last_label = label
    st.session_state.last_score = score
    st.success(f"Commande détectée : **{label}** (score {score:.2f})")


label = st.session_state.last_label
score = st.session_state.last_score

# ---------- affichage lampe + bascule ----------

col_lamp, col_toggle = st.columns(2)

# Lampe on/off (simple carré coloré)
with col_lamp:
    if label == "on":
        lamp_color = "yellow"
        lamp_text = "Allumée"
    elif label == "off":
        lamp_color = "gray"
        lamp_text = "Éteinte"
    else:
        lamp_color = "lightgray"
        lamp_text = "Inconnue"

    st.markdown(
        f"""
        <div style="width:80px;height:80px;border-radius:8px;
                    background-color:{lamp_color};
                    border:2px solid #333;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:bold;">
            {lamp_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------- Ligne 2 : 2 boutons côte à côte -------------------
col_left_right, col_up_down = st.columns(2)

# ========= BOUTON LEFT / RIGHT =========
with col_left_right:
    if label == "left":
        pos = "left"
    elif label == "right":
        pos = "right"
    else:
        pos = "center"

    if pos == "left":
        left_bg, right_bg = "#4CAF50", "#000000"
    elif pos == "right":
        left_bg, right_bg = "#000000", "#4CAF50"
    else:
        left_bg, right_bg = "#000000", "#000000"

    st.markdown(
        f"""
        <div style="width:160px;height:50px;border-radius:25px;
                    border:2px solid #333;display:flex;
                    overflow:hidden;">
            <div style="flex:1;background-color:{left_bg};
                        display:flex;align-items:center;justify-content:center;">
                Left
            </div>
            <div style="flex:1;background-color:{right_bg};
                        display:flex;align-items:center;justify-content:center;">
                Right
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========= BOUTON UP / DOWN =========
with col_up_down:
    if label == "up":
        pos = "up"
    elif label == "down":
        pos = "down"
    else:
        pos = "center"

    if pos == "up":
        up_bg, down_bg = "#4CAF50", "#000000"
    elif pos == "down":
        up_bg, down_bg = "#000000", "#4CAF50"
    else:
        up_bg, down_bg = "#000000", "#000000"

    st.markdown(
        f"""
        <div style="width:50px;height:160px;border-radius:25px;
                    border:2px solid #333;display:flex;flex-direction:column;overflow:hidden;">
            <div style="flex:1;background-color:{up_bg};color:white;
                        display:flex;align-items:center;justify-content:center;">
                Up
            </div>
            <div style="flex:1;background-color:{down_bg};color:white;
                        display:flex;align-items:center;justify-content:center;">
                Down
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if label is not None:
    st.caption(f"Dernière commande : `{label}` – score {score:.2f}")
else:
    st.caption("Aucune commande encore enregistrée.")
