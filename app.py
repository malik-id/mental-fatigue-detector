import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from math import pi

# Load model
model = joblib.load("model.pkl")

# Fungsi ekstraksi fitur (harus sama dengan training)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    # 13 MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    # Pitch
    pitch = librosa.yin(y, fmin=50, fmax=300)
    mean_pitch = np.mean(pitch)
    pitch_var = np.std(pitch)

    # Jitter & Shimmer
    jitter = np.std(pitch) / mean_pitch if mean_pitch > 0 else 0
    shimmer = np.std(y) / np.mean(np.abs(y))

    # Gabungan fitur
    features = np.hstack([mfccs, mean_pitch, pitch_var, jitter, shimmer])
    return features, pitch, mean_pitch, pitch_var, jitter, shimmer

# Fungsi simpan log
def save_log(file_name, prediction, prob_non_fatigue, prob_fatigue):
    log_file = "history.csv"
    new_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "prediction": "Fatigue" if prediction == 1 else "Non-Fatigue",
        "confidence": round(float(prob_fatigue if prediction == 1 else prob_non_fatigue), 3)
    }
    if not os.path.exists(log_file):
        pd.DataFrame([new_data]).to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(log_file, index=False)
    return new_data

# UI Streamlit
st.title("🧠 Deteksi Mental Fatigue Mahasiswa via Suara")
st.write("Upload rekaman suara (WAV/MP3) untuk analisis kondisi mental (Fatigue vs Non-Fatigue).")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "mp3"])

if uploaded_file is not None:
    # Simpan file sementara
    temp_file = "temp.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())

    # Tampilkan audio
    st.audio(temp_file)

    # Ekstrak fitur
    features, pitch_values, mean_pitch, pitch_var, jitter, shimmer = extract_features(temp_file)
    features = features.reshape(1, -1)

    # Prediksi
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    prob_non_fatigue, prob_fatigue = prob[0], prob[1]

    # Tampilkan hasil prediksi
    if prediction == 1:
        st.error(f"⚠️ Mahasiswa terdeteksi **Fatigue** (Confidence {prob_fatigue*100:.2f}%)")
    else:
        st.success(f"✅ Mahasiswa terdeteksi **Non-Fatigue** (Confidence {prob_non_fatigue*100:.2f}%)")

    # === Visualisasi Probabilitas ===
    fig, ax = plt.subplots()
    ax.bar(["Non-Fatigue", "Fatigue"], [prob_non_fatigue, prob_fatigue], color=["green", "red"])
    ax.set_ylabel("Probabilitas")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # === Histogram distribusi pitch (sesuai jurnal Figure 2) ===
    fig2, ax2 = plt.subplots()
    ax2.hist(pitch_values, bins=30, color="blue", alpha=0.7)
    ax2.set_title("Distribusi Pitch")
    ax2.set_xlabel("Pitch (Hz)")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)

    # === Radar chart fitur akustik (sesuai jurnal Figure 1) ===
    categories = ["Mean Pitch", "Pitch Variability", "Jitter", "Shimmer"]
    values = [mean_pitch, pitch_var, jitter, shimmer]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig3, ax3 = plt.subplots(subplot_kw=dict(polar=True))
    ax3.plot(angles, values, linewidth=2, linestyle='solid')
    ax3.fill(angles, values, 'skyblue', alpha=0.4)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_title("Profil Fitur Akustik")
    st.pyplot(fig3)

    # Simpan hasil ke CSV
    save_log(uploaded_file.name, prediction, prob_non_fatigue, prob_fatigue)
    st.write("📂 Hasil tersimpan ke **history.csv**")

# === Tampilkan riwayat ===
if os.path.exists("history.csv"):
    st.subheader("Riwayat Analisis")
    df_history = pd.read_csv("history.csv")
    st.dataframe(df_history)

    if st.button("🗑️ Hapus Riwayat"):
        os.remove("history.csv")
        st.warning("Riwayat berhasil dihapus! Silakan refresh halaman.")
