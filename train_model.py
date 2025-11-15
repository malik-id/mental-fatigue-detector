import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# === Fungsi ekstraksi fitur sesuai jurnal ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100, mono=True)

    # 13 MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    # Pitch (mean + variability)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    mean_pitch = np.mean(pitch)
    pitch_var = np.std(pitch)

    # Jitter & Shimmer
    jitter = np.std(pitch) / mean_pitch if mean_pitch > 0 else 0
    shimmer = np.std(y) / np.mean(np.abs(y))

    # Gabungkan fitur
    features = np.hstack([mfccs, mean_pitch, pitch_var, jitter, shimmer])
    return features

# === Load dataset ===
data_folder = "data/"
X, y = [], []

# Label 1 = Fatigue (jenuh), 0 = Non-Fatigue (tidak jenuh)
for label, folder in [(1, "jenuh"), (0, "tidak_jenuh")]:
    folder_path = os.path.join(data_folder, folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".wav", ".mp3")):
            file_path = os.path.join(folder_path, file_name)
            X.append(extract_features(file_path))
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Jumlah data:", len(y))
print("Distribusi label:", {0: list(y).count(0), 1: list(y).count(1)})

# === Split data (70:30) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Random Forest dengan GridSearchCV ===
param_grid = {
    "randomforestclassifier__n_estimators": [100, 200, 300],
    "randomforestclassifier__max_depth": [None, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 5, 10]
}

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# === Evaluasi Model ===
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nEvaluasi Model (Random Forest):")
print("Accuracy :", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall   :", round(rec, 3))
print("F1-Score :", round(f1, 3))
print("AUC      :", round(auc, 3))

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
print("CV Mean Accuracy:", round(np.mean(cv_scores), 3))

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# === Simpan model ===
joblib.dump(best_model, "model.pkl")
print("✅ Model disimpan ke model.pkl")
