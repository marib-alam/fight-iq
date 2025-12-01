# --- 1. Page Config ---
st.set_page_config(page_title="Fight IQ: Auto-Detection", page_icon="🥊", layout="wide")

st.title("🥊 Fight IQ: Session Analysis")
st.markdown("_Upload a full session. Rounds are detected intelligently using Density Clustering (DBSCAN)._")

# --- 2. Sidebar Controls ---
st.sidebar.header("⚙️ Signal Tuning")

# UPDATED: Frequency Filtering Controls
st.sidebar.subheader("Sonic Fingerprint")
st.sidebar.info("Your peak impact is **247 Hz**. Keep the filter centered around this.")
freq_low = st.sidebar.slider("Low Cut (Hz)", 50, 300, 150, help="Remove low rumble (below your 247Hz peak)")
freq_high = st.sidebar.slider("High Cut (Hz)", 300, 1000, 400, help="Remove high slap/hiss (above your 247Hz peak)")

st.sidebar.subheader("Detection Thresholds")
# Default sensitivity set to 0.25
sensitivity = st.sidebar.slider("Impact Sensitivity", 0.05, 0.80, 0.25, 0.01)
min_gap = st.sidebar.slider("Debounce (Min Gap)", 0.05, 0.50, 0.15, 0.01)

use_hpss = st.sidebar.checkbox(
    "Deep Noise Cleaning (Slow)", 
    value=False, 
    help="Enable this ONLY if you have constant background drone. The default 'Thud Filter' is usually better."
)

st.sidebar.divider()
st.sidebar.header("⏱️ Smart Round Logic")
max_pause = st.sidebar.slider("Max Pause Allowed (sec)", 10, 60, 30, 5)

st.sidebar.divider()
st.sidebar.header("🧠 Combo Logic")
# Default combo gap set to 0.8
combo_max_gap = st.sidebar.slider("Max Gap for Combo", 0.3, 1.5, 0.8, 0.1)
min_punches_for_combo = st.sidebar.slider("Min Punches in Combo", 2, 6, 3, 1)

# --- 3. Optimized Loader (THE THUD FILTER) ---
def bandpass_filter(data, lowcut, highcut, fs, order=5):
