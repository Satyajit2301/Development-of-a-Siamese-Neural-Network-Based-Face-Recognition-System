"""
app.py
------
Streamlit interface for the Siamese-network face recognition project.
Beautiful dark-themed UI with animated elements and modern design.

Pages: Home, Register New User, Start Face Recognition, View Registered Users,
       Delete Registered User, About & Architecture.

Run:  streamlit run app.py
Optional password for public deployments: set APP_PASSWORD in the Streamlit secrets.
"""
import os
import sys
import json
import hmac

import cv2
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import face_detection
import face_recognition_system as frs
import onboarding
from database import UserDatabase, is_valid_user_id

# ─────────────────── page config & theme ───────────────────
st.set_page_config(
    page_title="Siamese Face Recognition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── custom CSS (dark theme) ───────────────────
DARK_CSS = """
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Root variables ---- */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #16161f;
    --bg-card-hover: #1c1c28;
    --border-color: #2a2a3a;
    --border-glow: #6c63ff;
    --accent-primary: #6c63ff;
    --accent-secondary: #00d4aa;
    --accent-warning: #ff6b6b;
    --accent-info: #4ecdc4;
    --text-primary: #e8e8f0;
    --text-secondary: #9898b0;
    --text-muted: #6a6a80;
    --gradient-1: linear-gradient(135deg, #6c63ff 0%, #00d4aa 100%);
    --gradient-2: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
    --gradient-3: linear-gradient(135deg, #4ecdc4 0%, #6c63ff 100%);
    --shadow-card: 0 4px 20px rgba(0,0,0,0.3);
    --shadow-glow: 0 0 20px rgba(108,99,255,0.15);
}

/* ---- Global ---- */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.7rem !important;
}

section[data-testid="stSidebar"] .stRadio > div > label {
    color: var(--text-primary) !important;
    padding: 0.65rem 1rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    margin-bottom: 2px !important;
}

section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--bg-card) !important;
}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-checked="true"] {
    background: var(--bg-card) !important;
    border-left: 3px solid var(--accent-primary) !important;
}

/* ---- Cards ---- */
.dark-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    box-shadow: var(--shadow-card);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.dark-card:hover {
    border-color: var(--accent-primary);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}

/* ---- Hero section ---- */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-bottom: 2rem;
    line-height: 1.6;
}

/* ---- Metric Cards ---- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: var(--accent-primary);
    box-shadow: var(--shadow-glow);
}
.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.25rem 0;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ---- Status badges ---- */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}
.badge-success {
    background: rgba(0,212,170,0.15);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.3);
}
.badge-danger {
    background: rgba(255,107,107,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(255,107,107,0.3);
}
.badge-warning {
    background: rgba(254,202,87,0.15);
    color: #feca57;
    border: 1px solid rgba(254,202,87,0.3);
}
.badge-info {
    background: rgba(108,99,255,0.15);
    color: #6c63ff;
    border: 1px solid rgba(108,99,255,0.3);
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* ---- Pipeline step ---- */
.pipeline-step {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
}
.pipeline-step .step-num {
    background: var(--gradient-1);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 0.6rem;
}
.pipeline-step .step-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.pipeline-step .step-desc {
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.4;
}

/* ---- Distance Table ---- */
.distance-row {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
}
.distance-row.match {
    border-color: rgba(0,212,170,0.4);
    background: rgba(0,212,170,0.05);
}
.distance-row.no-match {
    border-color: rgba(255,107,107,0.2);
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border: 1px solid var(--border-color) !important;
}

button[kind="primary"] {
    background: var(--gradient-1) !important;
    border: none !important;
    color: white !important;
}

/* ---- Inputs ---- */
.stTextInput > div > div,
.stSelectbox > div > div {
    border-radius: 10px !important;
    border-color: var(--border-color) !important;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    border-radius: 12px !important;
    background: var(--bg-card) !important;
}

/* ---- Divider ---- */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    margin: 2rem 0;
}

/* ---- Tech tag ---- */
.tech-tag {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
    background: var(--bg-card);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
    font-family: 'JetBrains Mono', monospace;
}

/* ---- User card ---- */
.user-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}
.user-card:hover {
    border-color: var(--accent-primary);
    box-shadow: var(--shadow-glow);
}

/* ---- Glow dot animation ---- */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(0,212,170,0.5); }
    50% { box-shadow: 0 0 15px rgba(0,212,170,0.8); }
}
.glow-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-secondary);
    display: inline-block;
    animation: pulse-glow 2s infinite;
    margin-right: 0.5rem;
}

/* ---- Footer ---- */
.footer-text {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border-color);
    margin-top: 3rem;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}

/* ---- Image borders ---- */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

/* ---- Alert overrides ---- */
.stAlert {
    border-radius: 12px !important;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

PAGES = [
    "🏠  Home",
    "📝  Register New User",
    "🔍  Face Recognition",
    "👥  Registered Users",
    "🗑️  Delete User",
    "ℹ️  About & Architecture",
]


# ─────────────────── config & shared resources ───────────────────
def load_config():
    try:
        with open(os.path.join(BASE_DIR, "models", "threshold.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"threshold": 0.63}


CFG = load_config()
DEFAULT_T = float(CFG.get("operating_threshold", CFG.get("threshold", 0.63)))


@st.cache_resource(show_spinner=False)
def get_models():
    with st.spinner("⏳ Loading neural network models..."):
        embedder = onboarding.load_embedding_model()
        try:
            detector = face_detection.load_face_detector()
        except Exception:
            detector = None
    return embedder, detector


def get_db():
    return UserDatabase()


def check_password():
    """Ask for a password only if APP_PASSWORD is set in the Streamlit secrets."""
    try:
        secret = st.secrets["APP_PASSWORD"]
    except Exception:
        return True
    if st.session_state.get("authed"):
        return True
    st.markdown('<div class="hero-title" style="font-size:2rem;text-align:center;">🔐 Authentication Required</div>',
                unsafe_allow_html=True)
    entered = st.text_input("Password", type="password", placeholder="Enter your password...")
    if entered:
        if hmac.compare_digest(entered, str(secret)):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("❌ Wrong password")
    return False


def decode_upload(uploaded, max_width=960):
    """Streamlit upload / camera object -> BGR image (large images are scaled down)."""
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    return img


def show_bgr(img, caption=None):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


def flash():
    msg = st.session_state.pop("flash", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def load_training_history():
    """Load training history JSON for charts."""
    try:
        with open(os.path.join(BASE_DIR, "models", "training_history.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ─────────────────── pages ───────────────────

def page_home():
    db = get_db()

    # Hero
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div class="hero-title">🧬 Siamese Neural Network<br>Face Recognition System</div>
        <div class="hero-subtitle">
            Real-time face verification powered by a Siamese CNN with 128-d embeddings,
            contrastive loss training, and Viola-Jones face detection.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    n_users = db.count_users()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👤</div>
            <div class="metric-value">{n_users}</div>
            <div class="metric-label">Registered Users</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{DEFAULT_T:.2f}</div>
            <div class="metric-label">Default Threshold</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📐</div>
            <div class="metric-value">128-d</div>
            <div class="metric-label">Embedding Size</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">96.0%</div>
            <div class="metric-label">Balanced Accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # How it works — Pipeline
    st.markdown('<div class="section-header">⚡ How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Three-phase pipeline from training to real-time recognition.</div>',
                unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-num">1</div>
            <div class="step-title">📸 Detect</div>
            <div class="step-desc">Viola-Jones (Haar cascade) locates faces in the frame</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-num">2</div>
            <div class="step-title">✂️ Crop & Preprocess</div>
            <div class="step-desc">Calibrated crop, resize to 112×92, normalize to [0,1]</div>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-num">3</div>
            <div class="step-title">🧠 Embed</div>
            <div class="step-desc">Shared CNN maps the face to a 128-dimensional vector</div>
        </div>""", unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-num">4</div>
            <div class="step-title">📏 Compare</div>
            <div class="step-desc">Euclidean distance decides same or different person</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Quick start
    st.markdown('<div class="section-header">🚀 Quick Start Guide</div>', unsafe_allow_html=True)
    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown("""
        <div class="dark-card">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">📝</div>
            <div style="font-weight:600;color:#e8e8f0;margin-bottom:0.4rem;">Step 1: Register</div>
            <div style="color:#9898b0;font-size:0.85rem;line-height:1.5;">
                Enter a User ID and name, then provide about 10 photos. Turn your head slightly between captures
                for better coverage.
            </div>
        </div>""", unsafe_allow_html=True)
    with qs2:
        st.markdown("""
        <div class="dark-card">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">🔍</div>
            <div style="font-weight:600;color:#e8e8f0;margin-bottom:0.4rem;">Step 2: Recognize</div>
            <div style="color:#9898b0;font-size:0.85rem;line-height:1.5;">
                Take or upload a photo. The system detects your face, generates an embedding,
                and matches it against registered users.
            </div>
        </div>""", unsafe_allow_html=True)
    with qs3:
        st.markdown("""
        <div class="dark-card">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚙️</div>
            <div style="font-weight:600;color:#e8e8f0;margin-bottom:0.4rem;">Step 3: Manage</div>
            <div style="color:#9898b0;font-size:0.85rem;line-height:1.5;">
                View all registered users and their face embeddings, or remove users from the database
                when no longer needed.
            </div>
        </div>""", unsafe_allow_html=True)

    # Training history charts
    history = load_training_history()
    if history:
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Training Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">10-epoch training on AT&T face dataset (35 identities, contrastive loss, Adam optimizer).</div>',
                    unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        epochs = list(range(1, len(history["loss"]) + 1))
        with ch1:
            import pandas as pd
            loss_df = pd.DataFrame({
                "Epoch": epochs + epochs,
                "Loss": history["loss"] + history["val_loss"],
                "Type": ["Training"] * len(epochs) + ["Validation"] * len(epochs)
            })
            st.markdown('<div class="dark-card" style="padding:1rem;">', unsafe_allow_html=True)
            st.caption("📉 Loss over Epochs")
            st.line_chart(loss_df, x="Epoch", y="Loss", color="Type", height=280)
            st.markdown('</div>', unsafe_allow_html=True)
        with ch2:
            acc_df = pd.DataFrame({
                "Epoch": epochs + epochs,
                "Accuracy": history["accuracy"] + history["val_accuracy"],
                "Type": ["Training"] * len(epochs) + ["Validation"] * len(epochs)
            })
            st.markdown('<div class="dark-card" style="padding:1rem;">', unsafe_allow_html=True)
            st.caption("📈 Accuracy over Epochs")
            st.line_chart(acc_df, x="Epoch", y="Accuracy", color="Type", height=280)
            st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.info("⚠️ **Disclaimer:** This is a classroom demo (CSE 4192), not a security system. Accuracy on webcam images "
            "is lower than on the AT&T dataset. Face data is biometric data — register only people who have consented.")


def page_register():
    st.markdown('<div class="section-header">📝 Register New User</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Enroll a new user by providing their details and multiple face photos for the embedding database.</div>',
                unsafe_allow_html=True)

    db = get_db()
    embedder, detector = get_models()

    # User details
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;">1. User Credentials</div>', unsafe_allow_html=True)
    col_id, col_name = st.columns(2)
    with col_id:
        user_id = st.text_input("🆔 User ID", key="reg_input_user_id", max_chars=32, placeholder="e.g., john_doe",
                                help="Letters, digits, _ or - (1-32 chars)").strip()
    with col_name:
        name = st.text_input("👤 Full Name", key="reg_input_name", placeholder="e.g., John Doe").strip()

    # Inline validation feedback
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if not user_id:
            st.markdown('<span class="badge badge-warning">⚠️ User ID is required</span>', unsafe_allow_html=True)
        elif not is_valid_user_id(user_id):
            st.markdown('<span class="badge badge-danger">❌ Invalid: letters, numbers, _ and - only</span>', unsafe_allow_html=True)
        elif db.user_exists(user_id):
            st.markdown(f'<span class="badge badge-danger">❌ ID \'{user_id}\' is already taken</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-success">✓ User ID valid & available</span>', unsafe_allow_html=True)

    with v_col2:
        if not name:
            st.markdown('<span class="badge badge-warning">⚠️ Full Name is required</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-success">✓ Full Name provided</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Photo capture
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;">2. Capture or Upload Face Photos</div>', unsafe_allow_html=True)

    st.session_state.setdefault("reg_photos", [])
    st.session_state.setdefault("reg_cam_key", 0)
    photos = st.session_state["reg_photos"]

    tab_cam, tab_up = st.tabs(["📷 Camera Capture", "📤 Upload Photos"])
    with tab_cam:
        st.markdown("""<div style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.75rem;">
        Take a photo and click "Add this photo". Turn your head slightly each time for better coverage.</div>""",
                    unsafe_allow_html=True)
        shot = st.camera_input("Take a photo",
                               key=f"reg_cam_{st.session_state['reg_cam_key']}")
        if shot is not None and st.button("➕ Add this photo", type="primary"):
            img = decode_upload(shot)
            if img is not None:
                photos.append(img)
            st.session_state["reg_cam_key"] += 1
            st.rerun()

    with tab_up:
        st.markdown("""<div style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.75rem;">
        Upload several photos of the same person (different angles recommended).</div>""",
                    unsafe_allow_html=True)
        files = st.file_uploader("Upload photos", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="reg_upload", label_visibility="collapsed")

    frames = list(photos) + [decode_upload(f) for f in (files or [])]
    frames = [f for f in frames if f is not None]

    # Photo count display
    min_needed = onboarding.MIN_FACES
    has_enough_photos = len(frames) >= min_needed
    count_color = "#00d4aa" if has_enough_photos else "#ff6b6b"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:0.75rem 0 0.25rem;">
        <div>
            <span style="color:var(--text-secondary);font-size:0.85rem;">Photos ready:</span>
            <span style="color:{count_color};font-weight:700;font-size:1.3rem;margin-left:0.5rem;">{len(frames)}</span>
            <span style="color:var(--text-muted);font-size:0.8rem;margin-left:0.3rem;">/ {min_needed} minimum (10 recommended)</span>
        </div>
        <div>
            <span class="badge {'badge-success' if has_enough_photos else 'badge-warning'}">
                {'✓ Ready' if has_enough_photos else f'Need {min_needed - len(frames)} more'}
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Photo thumbnails
    if photos:
        st.image([cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in photos], width=80)
        if st.button("🗑️ Clear camera photos"):
            st.session_state["reg_photos"] = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Validation readiness check
    is_id_ok = bool(user_id) and is_valid_user_id(user_id) and not db.user_exists(user_id)
    is_name_ok = bool(name)
    can_register = is_id_ok and is_name_ok and has_enough_photos

    # Clear instructions if not ready
    if not can_register:
        missing_items = []
        if not user_id:
            missing_items.append("Enter a User ID")
        elif not is_valid_user_id(user_id):
            missing_items.append("Provide a valid User ID (letters, numbers, _, -)")
        elif db.user_exists(user_id):
            missing_items.append("Choose an unregistered User ID")
        if not is_name_ok:
            missing_items.append("Enter Full Name")
        if not has_enough_photos:
            missing_items.append(f"Add at least {min_needed} photos (current: {len(frames)})")

        st.markdown(f"""
        <div class="dark-card" style="border-color:rgba(254,202,87,0.3);background:rgba(254,202,87,0.05);">
            <div style="color:#feca57;font-size:0.85rem;font-weight:600;">⚠️ Registration Checklist:</div>
            <div style="color:var(--text-secondary);font-size:0.8rem;margin-top:0.25rem;">
                {' &nbsp;•&nbsp; '.join(missing_items)}
            </div>
        </div>""", unsafe_allow_html=True)

    # Register button
    if st.button("🚀 Register User", type="primary", disabled=not can_register, use_container_width=True):
        if not is_id_ok:
            st.warning("⚠️ Please provide a valid, unique User ID.")
            return
        if not is_name_ok:
            st.warning("⚠️ Please provide a Full Name.")
            return
        if not has_enough_photos:
            st.warning(f"⚠️ Please provide at least {min_needed} photos.")
            return

        with st.spinner("Processing faces and generating embeddings..."):
            res = onboarding.enrol_from_frames(user_id, name, frames, embedder, detector, db)
        if res["success"]:
            st.balloons()
            st.success(f"✅ {res['message']}")
            st.session_state["reg_photos"] = []
            st.session_state["reg_cam_key"] += 1
        else:
            st.error(f"❌ {res['message']}")

        # Stats
        st.markdown(f"""
        <div class="dark-card" style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div><span class="metric-label">Received</span><br>
                 <span style="color:var(--text-primary);font-weight:600;">{res['frames_received']}</span></div>
            <div><span class="metric-label">Used</span><br>
                 <span style="color:#00d4aa;font-weight:600;">{res['faces_used']}</span></div>
            <div><span class="metric-label">No Face</span><br>
                 <span style="color:#ff6b6b;font-weight:600;">{res['skipped_no_face']}</span></div>
            <div><span class="metric-label">Multi Face</span><br>
                 <span style="color:#feca57;font-weight:600;">{res['skipped_multiple_faces']}</span></div>
        </div>""", unsafe_allow_html=True)


def page_recognize():
    st.markdown('<div class="section-header">🔍 Face Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Identify a person by comparing their face against all registered embeddings.</div>',
                unsafe_allow_html=True)

    db = get_db()
    embedder, detector = get_models()

    if db.count_users() == 0:
        st.markdown("""
        <div class="dark-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🚫</div>
            <div style="font-size:1.2rem;font-weight:600;color:var(--text-primary);margin-bottom:0.5rem;">No Registered Users</div>
            <div style="color:var(--text-secondary);">Please register at least one user before using face recognition.</div>
        </div>""", unsafe_allow_html=True)
        return

    # Threshold slider
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    t = st.slider("🎚️ Decision Threshold (lower = stricter)",
                  0.30, 1.00, DEFAULT_T, 0.01,
                  help="A face is accepted only if its distance to the closest registered user is below this value.")
    strict = "Very Strict" if t < 0.45 else ("Strict" if t < 0.55 else ("Balanced" if t < 0.65 else ("Relaxed" if t < 0.80 else "Very Relaxed")))
    badge_class = "badge-success" if t < 0.55 else ("badge-info" if t < 0.65 else ("badge-warning" if t < 0.80 else "badge-danger"))
    st.markdown(f'<span class="badge {badge_class}">{strict} — T={t:.2f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input
    tab_cam, tab_up = st.tabs(["📷 Camera", "📤 Upload"])
    with tab_cam:
        shot = st.camera_input("Look at the camera", key="rec_cam")
    with tab_up:
        up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="rec_up")
    source = shot if shot is not None else up
    if source is None:
        st.markdown("""
        <div class="dark-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📸</div>
            <div style="color:var(--text-secondary);">Take or upload a photo to start recognition.</div>
        </div>""", unsafe_allow_html=True)
        return

    frame = decode_upload(source)
    if frame is None:
        st.error("❌ The image could not be read.")
        return

    with st.spinner("🧠 Analyzing face..."):
        rec = frs.FaceRecognizer(embedder, detector, db, threshold=t)
        result = rec.recognize_frame(frame)

    # Result display
    annotated = frs.annotate_frame(frame, result)
    show_bgr(annotated)

    status = result["status"]
    if status == "recognized":
        p = result["primary"]
        st.markdown(f"""
        <div class="dark-card" style="border-color: rgba(0,212,170,0.5);">
            <div style="display:flex;align-items:center;gap:0.75rem;">
                <div class="glow-dot"></div>
                <div>
                    <div style="font-weight:700;font-size:1.2rem;color:#00d4aa;">✅ Recognized: {p['name']}</div>
                    <div style="color:var(--text-secondary);font-size:0.85rem;">
                        User ID: <code>{p['user_id']}</code> &nbsp;|&nbsp;
                        Distance: <strong>{p['distance']:.3f}</strong> &nbsp;|&nbsp;
                        Threshold: {t:.2f}
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    elif status == "unknown":
        st.markdown(f"""
        <div class="dark-card" style="border-color: rgba(255,107,107,0.5);">
            <div style="display:flex;align-items:center;gap:0.75rem;">
                <div style="font-size:1.5rem;">🚷</div>
                <div>
                    <div style="font-weight:700;font-size:1.2rem;color:#ff6b6b;">Unknown Person</div>
                    <div style="color:var(--text-secondary);font-size:0.85rem;">{result['message']}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ {result['message']}")

    # Distance scores table
    p = result["primary"]
    if p and p["scores"]:
        st.markdown('<div class="section-header" style="font-size:1.1rem;margin-top:1.5rem;">📊 Distance Scores</div>',
                    unsafe_allow_html=True)
        for u, s in sorted(p["scores"].items(), key=lambda kv: kv[1]):
            is_match = s < t
            row_class = "match" if is_match else "no-match"
            badge = '<span class="badge badge-success">Match</span>' if is_match else '<span class="badge badge-danger">Rejected</span>'
            user_name = rec.registry[u]["name"] if u in rec.registry else u
            bar_width = max(5, min(95, int((1 - s / 1.5) * 100)))
            bar_color = "#00d4aa" if is_match else "#ff6b6b"
            st.markdown(f"""
            <div class="distance-row {row_class}">
                <div style="flex:1;">
                    <div style="font-weight:600;color:var(--text-primary);">{user_name}</div>
                    <div style="color:var(--text-muted);font-size:0.75rem;font-family:'JetBrains Mono',monospace;">{u}</div>
                </div>
                <div style="flex:2;margin:0 1rem;">
                    <div style="background:var(--bg-primary);border-radius:4px;height:6px;overflow:hidden;">
                        <div style="width:{bar_width}%;height:100%;background:{bar_color};border-radius:4px;"></div>
                    </div>
                </div>
                <div style="font-weight:700;color:var(--text-primary);font-family:'JetBrains Mono',monospace;min-width:60px;text-align:right;">{s:.3f}</div>
                <div style="min-width:80px;text-align:right;">{badge}</div>
            </div>""", unsafe_allow_html=True)


def page_view():
    st.markdown('<div class="section-header">👥 Registered Users</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Browse all users enrolled in the face recognition database.</div>',
                unsafe_allow_html=True)

    db = get_db()
    users = db.list_users()
    if not users:
        st.markdown("""
        <div class="dark-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
            <div style="font-size:1.2rem;font-weight:600;color:var(--text-primary);margin-bottom:0.5rem;">No Users Registered</div>
            <div style="color:var(--text-secondary);">Go to the Register page to add your first user.</div>
        </div>""", unsafe_allow_html=True)
        return

    # Stats bar
    total = len(users)
    total_emb = sum(u["n_embeddings"] for u in users)
    st.markdown(f"""
    <div class="dark-card" style="display:flex;gap:3rem;flex-wrap:wrap;">
        <div><span class="metric-label">Total Users</span><br>
             <span style="color:var(--accent-primary);font-weight:700;font-size:1.5rem;">{total}</span></div>
        <div><span class="metric-label">Total Embeddings</span><br>
             <span style="color:var(--accent-secondary);font-weight:700;font-size:1.5rem;">{total_emb}</span></div>
        <div><span class="metric-label">Avg per User</span><br>
             <span style="color:var(--accent-info);font-weight:700;font-size:1.5rem;">{total_emb/total:.1f}</span></div>
    </div>""", unsafe_allow_html=True)

    # User cards
    for u in users:
        with st.expander(f"👤 {u['name']}  —  `{u['user_id']}`  ·  {u['n_embeddings']} embeddings"):
            st.markdown(f"""
            <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:0.75rem;">
                <div><span class="metric-label">User ID</span><br>
                    <span style="color:var(--text-primary);font-family:'JetBrains Mono',monospace;">{u['user_id']}</span></div>
                <div><span class="metric-label">Full Name</span><br>
                    <span style="color:var(--text-primary);">{u['name']}</span></div>
                <div><span class="metric-label">Registered</span><br>
                    <span style="color:var(--text-secondary);">{u['created_at']}</span></div>
                <div><span class="metric-label">Embeddings</span><br>
                    <span class="badge badge-info">{u['n_embeddings']}</span></div>
            </div>""", unsafe_allow_html=True)

            folder = os.path.join(db.images_dir, u["user_id"])
            files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))[:8] if os.path.isdir(folder) else []
            if files:
                st.image([os.path.join(folder, f) for f in files], width=85)
            else:
                st.caption("No face images stored on disk.")


def page_delete():
    st.markdown('<div class="section-header">🗑️ Delete Registered User</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Permanently remove a user and all their face data from the database.</div>',
                unsafe_allow_html=True)

    flash()
    db = get_db()
    users = db.list_users()
    if not users:
        st.markdown("""
        <div class="dark-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">✨</div>
            <div style="font-size:1.2rem;font-weight:600;color:var(--text-primary);margin-bottom:0.5rem;">No Users to Delete</div>
            <div style="color:var(--text-secondary);">The database is already empty.</div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    labels = {f"👤 {u['name']}  ({u['user_id']})  —  {u['n_embeddings']} embeddings": u["user_id"] for u in users}
    choice = st.selectbox("Select user to delete", list(labels), label_visibility="visible")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    confirm = st.checkbox("⚠️ I understand this permanently deletes the user's embeddings and face images")

    if st.button("🗑️ Delete User", type="primary", disabled=not confirm, use_container_width=True):
        uid = labels[choice]
        if db.delete_user(uid):
            st.session_state["flash"] = ("success", f"✅ User '{uid}' has been deleted.")
        else:
            st.session_state["flash"] = ("error", f"❌ User '{uid}' could not be deleted.")
        st.rerun()


def page_about():
    st.markdown('<div class="section-header">ℹ️ About & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Technical details of the Siamese Neural Network Face Recognition System.</div>',
                unsafe_allow_html=True)

    # Architecture overview
    st.markdown("""
    <div class="dark-card">
        <div style="font-weight:700;color:var(--text-primary);font-size:1.1rem;margin-bottom:1rem;">🏗️ System Architecture</div>
        <div style="color:var(--text-secondary);line-height:1.8;font-size:0.9rem;">
            The system uses a <strong style="color:#6c63ff;">Siamese Neural Network</strong> with shared weights.
            Two identical CNNs process face images in parallel, producing 128-dimensional embedding vectors.
            The <strong style="color:#00d4aa;">Euclidean distance</strong> between embeddings determines whether
            two faces belong to the same person. The network is trained with
            <strong style="color:#feca57;">contrastive loss</strong> (margin = 1.0) to pull same-person embeddings
            together and push different-person embeddings apart.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Shared CNN architecture
    st.markdown("""
    <div class="dark-card">
        <div style="font-weight:700;color:var(--text-primary);font-size:1.1rem;margin-bottom:1rem;">🧠 Shared CNN Architecture</div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem;">
            <div class="tech-tag">Input 112×92×1</div>
            <span style="color:var(--text-muted);">→</span>
            <div class="tech-tag">Conv2D(64, 3×3, ReLU)</div>
            <span style="color:var(--text-muted);">→</span>
            <div class="tech-tag">MaxPool2D</div>
            <span style="color:var(--text-muted);">→</span>
            <div class="tech-tag">Conv2D(64, 3×3, ReLU)</div>
            <span style="color:var(--text-muted);">→</span>
            <div class="tech-tag">Flatten</div>
            <span style="color:var(--text-muted);">→</span>
            <div class="tech-tag" style="border-color:#6c63ff;color:#6c63ff;">Dense(128, Sigmoid)</div>
        </div>
        <div style="color:var(--text-muted);font-size:0.8rem;">
            Output: 128-d feature vector (face embedding)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training details
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:var(--text-primary);font-size:1.1rem;margin-bottom:1rem;">📋 Training Configuration</div>',
                unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="dark-card">
            <div style="font-weight:600;color:var(--accent-primary);margin-bottom:0.8rem;">Dataset</div>
            <table style="width:100%;color:var(--text-secondary);font-size:0.85rem;">
                <tr><td style="padding:0.3rem 0;">Dataset</td><td style="text-align:right;color:var(--text-primary);">AT&T (ORL) Faces</td></tr>
                <tr><td style="padding:0.3rem 0;">Individuals</td><td style="text-align:right;color:var(--text-primary);">40 (35 train / 5 test)</td></tr>
                <tr><td style="padding:0.3rem 0;">Images / person</td><td style="text-align:right;color:var(--text-primary);">10</td></tr>
                <tr><td style="padding:0.3rem 0;">Image size</td><td style="text-align:right;color:var(--text-primary);">112 × 92, grayscale</td></tr>
                <tr><td style="padding:0.3rem 0;">Total images</td><td style="text-align:right;color:var(--text-primary);">400</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="dark-card">
            <div style="font-weight:600;color:var(--accent-secondary);margin-bottom:0.8rem;">Hyperparameters</div>
            <table style="width:100%;color:var(--text-secondary);font-size:0.85rem;">
                <tr><td style="padding:0.3rem 0;">Loss function</td><td style="text-align:right;color:var(--text-primary);">Contrastive (margin 1.0)</td></tr>
                <tr><td style="padding:0.3rem 0;">Optimizer</td><td style="text-align:right;color:var(--text-primary);">Adam (lr = 0.001)</td></tr>
                <tr><td style="padding:0.3rem 0;">Batch size</td><td style="text-align:right;color:var(--text-primary);">128</td></tr>
                <tr><td style="padding:0.3rem 0;">Epochs</td><td style="text-align:right;color:var(--text-primary);">10</td></tr>
                <tr><td style="padding:0.3rem 0;">Scoring method</td><td style="text-align:right;color:var(--text-primary);">kNN-3 (mean of 3 closest)</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # Results
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:var(--text-primary);font-size:1.1rem;margin-bottom:1rem;">🏆 Results</div>',
                unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value" style="color:#00d4aa;">96.0%</div>
            <div class="metric-label">Balanced Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value" style="color:#6c63ff;">0.984</div>
            <div class="metric-label">ROC AUC</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value" style="color:#feca57;">0.63</div>
            <div class="metric-label">Optimal Threshold</div>
        </div>""", unsafe_allow_html=True)

    # Technology stack
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:var(--text-primary);font-size:1.1rem;margin-bottom:1rem;">🛠️ Technology Stack</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-card" style="display:flex;gap:0.4rem;flex-wrap:wrap;">
        <span class="tech-tag" style="border-color:#6c63ff;color:#6c63ff;">TensorFlow 2.21</span>
        <span class="tech-tag" style="border-color:#ff6b6b;color:#ff6b6b;">Keras 3.13</span>
        <span class="tech-tag" style="border-color:#00d4aa;color:#00d4aa;">OpenCV 4.14</span>
        <span class="tech-tag" style="border-color:#feca57;color:#feca57;">NumPy</span>
        <span class="tech-tag" style="border-color:#4ecdc4;color:#4ecdc4;">Streamlit</span>
        <span class="tech-tag" style="border-color:#ff9ff3;color:#ff9ff3;">SQLite</span>
        <span class="tech-tag" style="border-color:#48dbfb;color:#48dbfb;">Scikit-learn</span>
        <span class="tech-tag" style="border-color:#f368e0;color:#f368e0;">Python 3.13</span>
    </div>""", unsafe_allow_html=True)

    # Module map
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:var(--text-primary);font-size:1.1rem;margin-bottom:1rem;">📁 Project Modules</div>',
                unsafe_allow_html=True)

    modules = [
        ("app.py", "Streamlit web interface", "🖥️"),
        ("utils.py", "Data loading, pairs, shared CNN, distance, loss", "🔧"),
        ("siamese_nn.py", "Siamese model build, train, save/load", "🧬"),
        ("face_detection.py", "Viola-Jones detection & calibrated cropping", "📷"),
        ("onboarding.py", "User enrolment pipeline", "📝"),
        ("face_recognition_system.py", "Real-time recognition pipeline", "🔍"),
        ("database.py", "SQLite user & embedding storage", "🗄️"),
        ("crop_calibration.json", "Haar-to-AT&T crop geometry mapping", "📐"),
    ]
    for mod_name, mod_desc, mod_icon in modules:
        st.markdown(f"""
        <div class="dark-card" style="padding:0.8rem 1.2rem;display:flex;align-items:center;gap:0.75rem;margin:0.4rem 0;">
            <span style="font-size:1.2rem;">{mod_icon}</span>
            <div>
                <span style="font-weight:600;color:var(--text-primary);font-family:'JetBrains Mono',monospace;font-size:0.85rem;">{mod_name}</span>
                <span style="color:var(--text-muted);font-size:0.8rem;margin-left:0.75rem;">{mod_desc}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────── sidebar & main ───────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2.5rem;">🧬</div>
            <div style="font-weight:700;font-size:1.1rem;color:var(--text-primary);margin-top:0.3rem;">Face Recognition</div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem;">Siamese Neural Network</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="styled-divider" style="margin:0.5rem 0 1rem;"></div>', unsafe_allow_html=True)

        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        st.markdown('<div class="styled-divider" style="margin:1rem 0;"></div>', unsafe_allow_html=True)

        # System status
        model_exists = os.path.exists(os.path.join(BASE_DIR, "models", "shared_cnn.h5"))
        db = get_db()
        st.markdown(f"""
        <div style="padding:0 0.5rem;">
            <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;font-weight:600;">System Status</div>
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                <div class="glow-dot" style="{'animation:pulse-glow 2s infinite;background:#00d4aa;' if model_exists else 'background:#ff6b6b;animation:none;'}"></div>
                <span style="color:var(--text-secondary);font-size:0.8rem;">Model {'Loaded' if model_exists else 'Missing'}</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                <div class="glow-dot"></div>
                <span style="color:var(--text-secondary);font-size:0.8rem;">Database Online</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;">
                <div class="glow-dot"></div>
                <span style="color:var(--text-secondary);font-size:0.8rem;">{db.count_users()} Users Registered</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer-text" style="margin-top:2rem;border:none;padding-top:1rem;">
            CSE 4192 · Lab Assignment 06<br>
            Machine Learning Projects with Python
        </div>
        """, unsafe_allow_html=True)

    # Auth check
    if not check_password():
        return

    # Model check
    if not os.path.exists(os.path.join(BASE_DIR, "models", "shared_cnn.h5")):
        st.error("⚠️ `models/shared_cnn.h5` is missing. Train the model with the notebook first.")
        return

    # Page routing
    page_map = {
        "🏠  Home": page_home,
        "📝  Register New User": page_register,
        "🔍  Face Recognition": page_recognize,
        "👥  Registered Users": page_view,
        "🗑️  Delete User": page_delete,
        "ℹ️  About & Architecture": page_about,
    }
    page_map[page]()


main()
