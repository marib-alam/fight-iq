import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.signal
from sklearn.cluster import DBSCAN
from datetime import datetime
import io

# --- 1. Page Config ---
st.set_page_config(page_title="Fight IQ: Auto-Detection", page_icon="🥊", layout="wide")

st.title("🥊 Fight IQ: Session Analysis")
st.markdown("_Upload a full session. Rounds are detected intelligently using Density Clustering (DBSCAN)._")

# --- 2. Sidebar Controls ---
st.sidebar.header("⚙️ Signal Tuning")
sensitivity = st.sidebar.slider("Impact Sensitivity", 0.05, 0.80, 0.25, 0.01)
min_gap = st.sidebar.slider("Debounce (Min Gap)", 0.05, 0.50, 0.15, 0.01)

use_hpss = st.sidebar.checkbox(
    "Deep Noise Cleaning (Slow)", 
    value=False, 
    help="Enable this ONLY if you have loud background music/traffic."
)

st.sidebar.divider()
st.sidebar.header("⏱️ Smart Round Logic")
max_pause = st.sidebar.slider("Max Pause Allowed (sec)", 10, 60, 30, 5)

st.sidebar.divider()
st.sidebar.header("🧠 Combo Logic")
combo_max_gap = st.sidebar.slider("Max Gap for Combo", 0.3, 1.5, 0.5, 0.1)
min_punches_for_combo = st.sidebar.slider("Min Punches in Combo", 2, 6, 3, 1)

# --- 3. Optimized Loader (Memory Safe) ---
def highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y

@st.cache_data
def load_audio(uploaded_file, use_hpss):
    # 1. Load Audio (Downsampled)
    # Streamlit Cloud has ~1GB RAM limit. We must be careful.
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    if not use_hpss:
        # Fast Mode: Simple Filter (Low Memory)
        y_clean = highpass_filter(y, 200, sr)
        return y, sr, y_clean
    else:
        # Heavy Mode: HPSS (Harmonic-Percussive Source Separation)
        # We MUST process this in chunks to avoid crashing the server.
        
        chunk_duration = 60 # Process 60 seconds at a time
        chunk_samples = chunk_duration * sr
        total_samples = len(y)
        
        y_percussive_full = []
        
        # Create a progress bar because this will be slow
        progress_bar = st.progress(0, text="Deep Cleaning Audio Chunks...")
        
        for i in range(0, total_samples, chunk_samples):
            # Slice the chunk
            chunk = y[i : i + chunk_samples]
            
            # Pad the last chunk if it's too short (to avoid edge errors)
            if len(chunk) < 2048: 
                break 
                
            # Run HPSS on just this small slice
            # This keeps RAM usage tiny (~50MB instead of 2GB)
            _, y_p_chunk = librosa.effects.hpss(chunk, margin=3.0)
            y_percussive_full.append(y_p_chunk)
            
            # Update Progress
            percent = min(1.0, (i + chunk_samples) / total_samples)
            progress_bar.progress(percent)
            
        progress_bar.empty() # Clear bar when done
        
        # Stitch chunks back together
        y_percussive = np.concatenate(y_percussive_full)
        
        # Ensure length matches original (in case of rounding errors)
        # We pad with zeros or trim to match exact length
        if len(y_percussive) < len(y):
            y_percussive = np.pad(y_percussive, (0, len(y) - len(y_percussive)))
        elif len(y_percussive) > len(y):
            y_percussive = y_percussive[:len(y)]
            
        return y, sr, y_percussive

# --- 4. Helper Function: Metrics ---
def get_metrics(df_chunk, duration_seconds=None):
    if df_chunk.empty:
        return None
    
    total_punches = len(df_chunk)
    intervals = df_chunk['Time'].diff().fillna(100)
    
    is_chain = intervals < combo_max_gap
    combo_id = (is_chain != is_chain.shift()).cumsum()
    combo_groups = df_chunk[is_chain].groupby(combo_id)
    total_combos = sum((len(group) + 1) >= min_punches_for_combo for _, group in combo_groups)

    active_intervals = intervals[intervals < 2.0]
    if not active_intervals.empty:
        avg_tempo = active_intervals.mean()
        peak_rate = 1.0 / active_intervals.min() if active_intervals.min() > 0 else 0
    else:
        avg_tempo = 0
        peak_rate = 0

    if duration_seconds is None:
        duration_seconds = df_chunk['Time'].max() - df_chunk['Time'].min()
        if duration_seconds < 1: duration_seconds = 1
        
    ppm = total_punches / (duration_seconds / 60)

    return {
        "punches": total_punches,
        "combos": total_combos,
        "avg_power": df_chunk['Power'].mean(),
        "avg_tempo": avg_tempo,
        "peak_rate": peak_rate,
        "ppm": ppm,
        "duration_sec": duration_seconds
    }

# --- 5. Main Logic ---
uploaded_file = st.file_uploader("Upload Session Audio (WAV/MP3)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    
    # We moved the Spinner inside to handle the progress bar better
    y, sr, y_percussive = load_audio(uploaded_file, use_hpss)

    with st.spinner('Detecting Punches...'):
        wait_frames = int(min_gap * sr / 512) 
        onset_frames = librosa.onset.onset_detect(
            y=y_percussive, sr=sr, wait=wait_frames, delta=sensitivity
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        punch_data = []
        for i, frame in enumerate(onset_frames):
            window = int(0.03 * sr)
            start = max(0, frame - window)
            end = min(len(y), frame + window)
            power = np.max(np.abs(y[start:end]))
            punch_data.append({"Time": onset_times[i], "Power": power})
        
        df = pd.DataFrame(punch_data)

    if not df.empty and len(df) > 5:
        # --- DBSCAN LOGIC ---
        X = df['Time'].values.reshape(-1, 1)
        db = DBSCAN(eps=max_pause, min_samples=5).fit(X)
        df['Round_ID'] = db.labels_
        df = df[df['Round_ID'] != -1].copy()
        
        unique_labels = sorted(df['Round_ID'].unique())
        label_map = {label: i+1 for i, label in enumerate(unique_labels)}
        df['Round'] = df['Round_ID'].map(label_map)
        
        detected_rounds = len(unique_labels)
        
        # --- CALCULATE METRICS FOR DOWNLOAD ---
        metrics = get_metrics(df, duration_seconds=len(y)/sr)
        
        # Prepare Export Data (Single Row Summary)
        export_data = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Filename": uploaded_file.name,
            "Total Punches": metrics['punches'],
            "Total Rounds": detected_rounds,
            "Combos": metrics['combos'],
            "Avg Pace (PPM)": round(metrics['ppm']),
            "Avg Power": round(metrics['avg_power'], 4),
            "Peak Rate (Hit/s)": round(metrics['peak_rate'], 2)
        }
        df_export = pd.DataFrame([export_data])

        # --- DISPLAY ---
        st.toast(f"🤖 AI Detected {detected_rounds} Rounds")
        
        st.subheader("Session Timeline")
        fig_timeline = px.scatter(
            df, x="Time", y="Power", color="Round",
            size="Power", color_continuous_scale="turbo",
            title=f"Detected {detected_rounds} Rounds (Max Pause: {max_pause}s)"
        )
        fig_timeline.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # --- SAVE/DOWNLOAD SECTION ---
        st.divider()
        c_dl1, c_dl2 = st.columns([3, 1])
        with c_dl1:
            st.markdown("#### 💾 Save Your Progress")
            st.caption("Download this session's stats to build your own history tracker.")
        with c_dl2:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Session Report (CSV)",
                data=csv,
                file_name=f"fight_iq_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

        # Tabs
        rounds = sorted(df['Round'].unique())
        tab_titles = ["📝 Full Summary"] + [f"Round {r}" for r in rounds]
        tabs = st.tabs(tab_titles)

        # TAB 0: SUMMARY
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Punches", metrics['punches'])
            c2.metric("Total Combos", metrics['combos'])
            c3.metric("Avg Pace", f"{metrics['ppm']:.0f} PPM")
            c4.metric("Avg Power", f"{metrics['avg_power']:.3f}")
            
            with st.expander("Preview Export Data"):
                st.dataframe(df_export)

        # TAB X: ROUNDS
        for i, r in enumerate(rounds):
            with tabs[i+1]:
                df_round = df[df['Round'] == r].copy()
                r_metrics = get_metrics(df_round)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"R{r} Punches", r_metrics['punches'])
                m2.metric(f"R{r} Combos", r_metrics['combos'])
                m3.metric("Round Pace", f"{r_metrics['ppm']:.0f} PPM")
                m4.metric("Est. Duration", f"{r_metrics['duration_sec']:.0f} sec")
                
                st.divider()
                
                df_round['Interval'] = df_round['Time'].diff().fillna(100)
                df_round['Type'] = np.where(df_round['Interval'] < combo_max_gap, 'Combo Hit', 'Single Shot')
                
                fig_round = px.scatter(
                    df_round, x="Time", y="Power", color="Type",
                    size="Power", 
                    color_discrete_map={'Combo Hit': '#ff4b4b', 'Single Shot': '#4b88ff'},
                    title=f"Round {r} Analysis"
                )
                st.plotly_chart(fig_round, use_container_width=True)
                
    else:
        st.warning("No punches found or session too short for clustering.")
