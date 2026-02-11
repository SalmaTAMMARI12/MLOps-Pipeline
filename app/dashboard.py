import os
import streamlit as st
import requests

# ✅ Docker inter-container: utiliser "api" (nom du service compose)
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

# Session HTTP (plus propre)
SESSION = requests.Session()

st.set_page_config(page_title="Diagnostic Étudiant", layout="wide")

st.markdown("""
    <style>
    .result-card { padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px; }
    .risk { background-color: #ff4b4b; }
    .safe { background-color: #28a745; }
    </style>
""", unsafe_allow_html=True)

st.title("Student Success Diagnostic")

# ✅ Petit indicateur santé API
with st.sidebar:
    st.subheader("Connexion API")
    try:
        r = SESSION.get(HEALTH_URL, timeout=3)
        if r.status_code == 200:
            st.success("API OK")
        else:
            st.warning(f"API répond mais status={r.status_code}")
    except Exception:
        st.error("API inaccessible (vérifie docker compose)")

with st.form("student_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Âge", 15, 50, 20)
        education = st.selectbox("Niveau d'éducation", ["High School", "Undergraduate", "Postgraduate"])
        study_h = st.slider("Heures d'étude / jour", 0.0, 15.0, 5.0)

    with c2:
        uses_ai = st.selectbox("Utilise l'IA ?", ["Yes", "No"])
        ai_tools = st.selectbox("Outil IA principal", ["ChatGPT", "Gemini", "Copilot", "None"])
        ai_purpose = st.selectbox("Objectif IA", ["Studying", "Research", "Coding", "None"])

    with c3:
        g_before = st.number_input("Note avant IA", 0, 100, 70)
        g_after = st.number_input("Note après IA", 0, 100, 75)
        screen_h = st.slider("Temps d'écran / jour", 0.0, 15.0, 4.0)

    submitted = st.form_submit_button("LANCER LE DIAGNOSTIC", use_container_width=True)

if submitted:
    payload = {
        "age": float(age),
        "education_level": education,
        "study_hours_per_day": float(study_h),
        "uses_ai": uses_ai,
        "ai_tools_used": ai_tools,
        "purpose_of_ai": ai_purpose,
        "grades_before_ai": float(g_before),
        "grades_after_ai": float(g_after),
        "daily_screen_time_hours": float(screen_h)
    }

    try:
        with st.spinner("Analyse..."):
            resp = SESSION.post(PREDICT_URL, json=payload, timeout=20)

        # ✅ si l'API renvoie erreur
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            # ✅ parse JSON
            res = resp.json()

            if "risk" not in res:
                st.error(f"Réponse inattendue API: {res}")
            else:
                # 1) Image SHAP
                if res.get("force_plot"):
                    st.write("### 🔍 Analyse de Décision (SHAP Force Plot)")
                    st.image(f"data:image/png;base64,{res['force_plot']}", use_container_width=True)

                # 2) Résultat
                prob = float(res["probability"])
                if int(res["risk"]) == 1:
                    st.markdown(
                        f'<div class="result-card risk"><h1>🚩 RISQUE DÉTECTÉ ({prob:.1%})</h1></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-card safe"><h1>✅ PROFIL STABLE ({1 - prob:.1%})</h1></div>',
                        unsafe_allow_html=True
                    )

                # 3) Recommandations
                st.write("### 💡 Recommandations Personnalisées")
                for r in res.get("recommendations", []):
                    st.info(r)

    except requests.exceptions.ConnectTimeout:
        st.error("Timeout connexion API (trop lent).")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Erreur de connexion à l'API: {e}")
    except ValueError:
        st.error(f"L'API n'a pas renvoyé du JSON valide: {resp.text}")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
