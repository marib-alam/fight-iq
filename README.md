🥊 Fight IQ: Audio Analytics for Boxing

Fight IQ is a boxing analytics dashboard that turns audio recordings into professional training data.

Instead of using expensive hardware sensors (accelerometers/gyroscopes), Fight IQ uses Digital Signal Processing (DSP) and Unsupervised Machine Learning (DBSCAN) to analyze the sound of your gloves hitting the heavy bag.

Simply record your session on your phone or laptop while listening to music in your headphones ("Silent Disco" setup), upload the audio, and get instant feedback on your pace, power, and rhythm.

📸 Demo

Automated round detection and punch clustering visualization.

🚀 Key Features

🎙️ Audio-Based Counting: Uses librosa and scipy to detect sharp transients (punches) while filtering out background noise.

🤖 Auto-Round Detection: Uses DBSCAN Clustering to automatically find rest periods and segment your workout into rounds without manual input.

⚡ Combo Analysis: Tracks your punch density to identify "Combos" vs. "Pot Shots" based on time intervals.

📊 Interactive Dashboard: Built with Streamlit and Plotly for zoomable timelines, power heatmaps, and pace tracking.

💾 Session History: Export your workout stats to CSV to track stamina trends over time.

🛠️ Tech Stack

Frontend: Streamlit

Audio Processing: Librosa, Scipy (High-Pass Filters, Onset Strength)

Machine Learning: Scikit-learn (DBSCAN for round segmentation)

Visualization: Plotly Express

Data Handling: Pandas, Numpy

📦 Installation

Prerequisites

Python 3.8+

FFmpeg (for MP3 processing)

Local Setup

Clone the repository

git clone [https://github.com/yourusername/fight-iq.git](https://github.com/marib-alam/fight-iq.git)
cd fight-iq


Create a Virtual Environment

python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate


Install Dependencies

pip install -r requirements.txt


Run the App

streamlit run app.py


🎯 How to Use

The Setup: Wear headphones (music) and set your phone/laptop to record audio near the heavy bag or double-end bag.

The Workout: Box normally. Ensure distinct impacts.

The Analysis:

Upload the audio file (WAV/MP3).

Adjust the Sensitivity slider until the "Total Punches" matches your perception.

Use Deep Noise Cleaning if you had background noise (traffic/fans).

Download your Session Report CSV.

🧠 The Logic

Punch Detection: We use a high-pass filter (>200Hz) to remove bag rumble, then calculate the RMS amplitude of transients.

Round Separation: Instead of hard-coded timers, we calculate the time gaps between every punch and use Density-Based Spatial Clustering (DBSCAN) to identify "Rest Clusters" vs "Active Clusters."

📄 License

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.
