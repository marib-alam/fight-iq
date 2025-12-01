import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    """
    The 'Sonic Fingerprint' Filter.
    It rejects everything except the specific frequency range of the user's punch.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Create a Bandpass filter
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.filtfilt(b, a, data)
    return y

@st.cache_data
def load_audio(uploaded_file, use_hpss, low_cut, high_cut):
    # Load with fixed SR for consistency
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    if not use_hpss:
        # --- THE FIX ---
        # Apply the Dynamic Bandpass Filter based on sidebar inputs
        # This focuses detection specifically on the user's 247Hz 'Thud'
        y_clean = bandpass_filter(y, low_cut, high_cut, sr)
        return y, sr, y_clean
    else:
        # Legacy Deep Clean (HPSS)
        chunk_duration = 60 
        chunk_samples = chunk_duration * sr
        total_samples = len(y)
        y_percussive_full = []
        progress_bar = st.progress(0, text="Deep Cleaning Audio Chunks...")
        for i in range(0, total_samples, chunk_samples):
            chunk = y[i : i + chunk_samples]
            if len(chunk) < 2048: break 
            _, y_p_chunk = librosa.effects.hpss(chunk, margin=3.0)
            y_percussive_full.append(y_p_chunk)
            percent = min(1.0, (i + chunk_samples) / total_samples)
            progress_bar.progress(percent)
        progress_bar.empty()
        y_percussive = np.concatenate(y_percussive_full)
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

    # Convert everything to standard Python floats to avoid JSON errors
    return {
        "punches": int(total_punches),
        "combos": int(total_combos),
        "avg_power": float(df_chunk['Power'].mean()),
        "avg_tempo": float(avg_tempo),
        "peak_rate": float(peak_rate),
        "ppm": float(ppm),
        "duration_sec": float(duration_seconds)
    }

# --- 5. Main Logic ---
uploaded_file = st.file_uploader("Upload Session Audio (WAV/MP3)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    
    # Pass the frequency controls to the loader
    y, sr, y_percussive = load_audio(uploaded_file, use_hpss, freq_low, freq_high)

    # --- WAVEFORM VISUALIZER WITH DOTS ---
    st.subheader("Audio Waveform & Detection")
    
    # 0. Audio Player
    uploaded_file.seek(0) 
    st.audio(uploaded_file)

    # 1. Downsample the wave for plotting (max 10k points for speed)
    step = max(1, int(len(y_percussive) / 10000))
    y_view = y_percussive[::step]
    x_view = np.arange(len(y_view)) * (step / sr)
    
    fig_wave = go.Figure()
    
    # Add the Green Wave
    fig_wave.add_trace(go.Scatter(
        x=x_view, y=y_view,
        mode='lines',
        name='Signal',
        line=dict(color='#00FF00', width=1),
        fill='tozeroy'
    ))

    fig_wave.update_layout(
        height=250, 
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Seconds'),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", y=1.1)
    )

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
            # Ensure stored power is a standard float
            punch_data.append({"Time": float(onset_times[i]), "Power": float(power)})
        
        df = pd.DataFrame(punch_data)

    # Add the Orange Dots to the Waveform if punches exist
    if not df.empty:
        fig_wave.add_trace(go.Scatter(
            x=df['Time'], 
            y=df['Power'], # Place dot at the height of the punch power
            mode='markers',
            name='Detected Punch',
            marker=dict(color='#FFA500', size=8, symbol='circle', line=dict(color='white', width=1)) 
        ))
    
    # Now render the waveform chart
    st.plotly_chart(fig_wave, use_container_width=True)

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
        
        # --- CALCULATE METRICS ---
        metrics = get_metrics(df, duration_seconds=len(y)/sr)
        
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

        st.toast(f"🤖 AI Detected {detected_rounds} Rounds")
        
        st.subheader("Session Timeline")
        fig_timeline = px.scatter(
            df, x="Time", y="Power", color="Round",
            size="Power", color_continuous_scale="turbo",
            title=f"Detected {detected_rounds} Rounds (Max Pause: {max_pause}s)"
        )
        fig_timeline.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # --- ROUND SUMMARY TABLE ---
        st.subheader("📊 Round-by-Round Breakdown")
        
        rounds_summary = []
        for r in unique_labels:
            round_num = label_map[r]
            df_round = df[df['Round'] == round_num].copy()
            r_metrics = get_metrics(df_round)
            
            rounds_summary.append({
                "Round": int(round_num),
                "Punches": int(r_metrics['punches']),
                "Combos": int(r_metrics['combos']),
                "Pace (PPM)": int(round(r_metrics['ppm'])),
                "Duration (s)": int(round(r_metrics['duration_sec'])),
                "Avg Power": float(round(r_metrics['avg_power'], 4)),
                "Peak Speed": float(round(r_metrics['peak_rate'], 1))
            })
            
        df_rounds_summary = pd.DataFrame(rounds_summary)
        
        st.dataframe(
            df_rounds_summary, 
            hide_index=True,
            column_config={
                "Round": st.column_config.NumberColumn(format="R %d"),
                "Avg Power": st.column_config.ProgressColumn(
                    format="%.4f", 
                    min_value=0.0, 
                    max_value=float(df['Power'].max())
                ),
            },
            use_container_width=True
        )

        # --- SAVE/DOWNLOAD ---
        st.divider()
        c_dl1, c_dl2 = st.columns([3, 1])
        with c_dl1:
            st.markdown("#### 💾 Save Your Progress")
            st.caption("Download this session's stats to build your own history tracker.")
        with c_dl2:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name=f"fight_iq_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

        # Tabs
        rounds = sorted(df['Round'].unique())
        tab_titles = ["📝 Full Summary"] + [f"Round {r}" for r in rounds]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Punches", metrics['punches'])
            c2.metric("Total Combos", metrics['combos'])
            c3.metric("Avg Pace", f"{metrics['ppm']:.0f} PPM")
            c4.metric("Avg Power", f"{metrics['avg_power']:.3f}")
            
            with st.expander("Preview Export Data"):
                st.dataframe(df_export)

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
