import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CardioSense | Heart Risk Check",
    page_icon="Heart",
    layout="wide",
    initial_sidebar_state="collapsed",
)

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink:#17242b; --muted:#64747b; --teal:#087f8c; --teal-dark:#075e69; --line:#dce8e6; }
    .stApp { background:linear-gradient(135deg,#f8fbfa 0%,#eef8f5 52%,#fffaf6 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
    .stApp::before,.stApp::after { content:''; position:fixed; width:260px; height:260px; border-radius:50%; filter:blur(2px); opacity:.45; pointer-events:none; z-index:0; animation:drift 9s ease-in-out infinite alternate; }
    .stApp::before { background:#c9eee4; top:-100px; right:-70px; }
    .stApp::after { background:#fde3d9; bottom:-120px; left:-80px; animation-delay:-3s; }
    .block-container { max-width:1180px; padding:3.5rem 2rem 4rem; position:relative; z-index:1; }
    .hero { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; margin-bottom:2.2rem; animation:rise .7s ease both; }
    .eyebrow { color:var(--teal); font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.55rem; }
    .hero h1 { color:var(--ink); font-family:'Manrope',sans-serif; font-size:clamp(2.3rem,4vw,4rem); letter-spacing:-.045em; line-height:1.03; margin:0; }
    .hero h1 span { color:var(--teal); }
    .hero p { color:var(--muted); font-size:1.05rem; line-height:1.6; margin:.9rem 0 0; max-width:560px; }
    .badge { background:rgba(255,255,255,.7); border:1px solid var(--line); border-radius:999px; color:var(--teal-dark); font-size:.83rem; font-weight:700; padding:.7rem 1rem; white-space:nowrap; box-shadow:0 8px 25px rgba(25,83,78,.06); }
    .form-shell { background:rgba(255,255,255,.86); border:1px solid rgba(220,232,230,.9); border-radius:24px; box-shadow:0 22px 60px rgba(34,75,70,.1); padding:1.7rem 1.8rem 1.4rem; animation:rise .8s .1s ease both; backdrop-filter:blur(12px); }
    .section-title { color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; margin:.3rem 0 1.1rem; }
    .section-title span { color:var(--teal); margin-right:.45rem; }
    [data-testid="stNumberInput"],[data-testid="stSelectbox"],[data-testid="stSlider"] { margin-bottom:.55rem; }
    label { color:#53666d !important; font-size:.82rem !important; font-weight:600 !important; }
    div[data-baseweb="select"] > div,input { background:#fbfdfc !important; border-color:var(--line) !important; border-radius:11px !important; color:var(--ink) !important; }
    div[data-baseweb="select"] [data-baseweb="select"] > div,
    div[data-baseweb="select"] [role="option"],
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    [data-testid="stNumberInput"] input { color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
    [data-baseweb="popover"] [role="option"] { background:#ffffff !important; color:var(--ink) !important; }
    [data-baseweb="popover"] [role="option"]:hover { background:#e5f6f1 !important; color:var(--teal-dark) !important; }
    [data-testid="stSlider"] [data-testid="stMarkdownContainer"], [data-testid="stSlider"] output { color:var(--ink) !important; }
    div[data-baseweb="select"] > div:hover,input:focus { border-color:#70bcb5 !important; box-shadow:0 0 0 3px rgba(8,127,140,.1) !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--teal); }
    .stButton > button, [data-testid="stFormSubmitButton"] button { background:linear-gradient(100deg,var(--teal-dark),var(--teal)) !important; border:0 !important; border-radius:12px !important; box-shadow:0 10px 22px rgba(8,127,140,.22) !important; color:#ffffff !important; font-size:1rem !important; font-weight:700 !important; height:3.1rem; margin-top:.8rem; transition:transform .2s ease,box-shadow .2s ease; }
    .stButton > button *, [data-testid="stFormSubmitButton"] button * { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background:linear-gradient(100deg,#064f59,var(--teal-dark)) !important; box-shadow:0 14px 28px rgba(8,127,140,.3) !important; transform:translateY(-2px); }
    .stButton > button:active, [data-testid="stFormSubmitButton"] button:active { transform:translateY(0); }
    .note { color:var(--muted); font-size:.78rem; line-height:1.5; margin:.85rem 0 .1rem; text-align:center; }
    .result { border-radius:18px; margin-top:1.25rem; padding:1.25rem 1.4rem; animation:pop .5s ease both; }
    .result.high { background:#fff0ed; border:1px solid #f5c4bb; color:#9a3f38; }
    .result.low { background:#e9f8f2; border:1px solid #b6e4d7; color:#146b5e; }
    .result strong { display:block; font-family:'Manrope',sans-serif; font-size:1.15rem; margin-bottom:.25rem; }
    .result small { color:inherit; opacity:.82; }
    .footer { color:#829196; font-size:.75rem; margin-top:1.6rem; text-align:center; }
    @keyframes rise { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
    @keyframes pop { from { opacity:0; transform:scale(.97); } to { opacity:1; transform:scale(1); } }
    @keyframes drift { from { transform:translate(0,0) rotate(0deg); } to { transform:translate(15px,18px) rotate(8deg); } }
    @media (max-width:700px) { .block-container { padding:2rem 1rem 3rem; } .hero { align-items:flex-start; flex-direction:column; gap:1rem; } .hero h1 { font-size:2.55rem; } .form-shell { border-radius:18px; padding:1.2rem 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div>
            <div class="eyebrow">CardioSense · Private screening</div>
            <h1>Understand your<br><span>heart health.</span></h1>
            <p>Enter a few clinical details for a quick, model-based heart disease risk check.</p>
        </div>
        <div class="badge">Ready for assessment</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("heart_assessment"):
    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>01</span> Personal profile</div>', unsafe_allow_html=True)
    profile_left, profile_right = st.columns(2)
    with profile_left:
        age = st.slider("Age", 18, 100, 40)
    with profile_right:
        sex = st.selectbox("Sex", ["M", "F"])

    st.markdown('<div class="section-title"><span>02</span> Clinical indicators</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    with right:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slop = st.selectbox("ST Slope", ["UP", "Flat", "Down"])
        st.markdown('<div class="note">Your inputs stay in this session and are used only for this prediction.</div>', unsafe_allow_html=True)

    predict = st.form_submit_button("Run heart risk check  ->", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if predict:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholestrol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slop: 1,
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.markdown('<div class="result high"><strong>High risk signal detected</strong><small>This model suggests discussing these results with a qualified healthcare professional.</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result low"><strong>Lower risk signal detected</strong><small>This result is reassuring, but it does not replace professional medical advice.</small></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">CardioSense · A screening aid, not a diagnosis</div>', unsafe_allow_html=True)
