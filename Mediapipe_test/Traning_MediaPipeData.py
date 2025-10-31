import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ---------------- Paths ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAIN_CSV      = SCRIPT_DIR / "gesture_data_new.csv"
ENCODER_PATH  = PROJECT_ROOT / "gesture_label_encoder.pkl"
SCALER_PATH   = PROJECT_ROOT / "gesture_input_scaler.pkl"        # NEW: save scaler for inference
TFLITE_PATH   = PROJECT_ROOT / "gesture_model_new.tflite"
H5_PATH       = PROJECT_ROOT / "gesture_model.h5"
PRED_CSV_PATH = SCRIPT_DIR / "gesture_data_with_confidence.csv"

# ---------------- Load CSV ----------------
if not MAIN_CSV.exists():
    raise FileNotFoundError(f"Could not find CSV: {MAIN_CSV}")

df = pd.read_csv(MAIN_CSV)
if "label" not in df.columns:
    raise ValueError("CSV must contain a 'label' column.")

# Drop duplicates, shuffle
df = df.drop_duplicates().copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# KEEP ONLY the 4 classes we want  ------------------------  NEW / CHANGED
allowed = {"rock", "paper", "scissor", "game"}
df = df[df["label"].isin(allowed)].copy()
if df.empty:
    raise ValueError("After filtering to rock/paper/scissor/game, no rows remained.")
print("Classes present (post-filter):", sorted(df["label"].unique()))

# ---------------- Prepare Data ----------------
# Expected landmark columns: x0..x20, y0..y20, z0..z20 (63 total)
landmark_columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
missing = [c for c in landmark_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected landmark columns: {missing[:5]}{'...' if len(missing)>5 else ''}")

X = df[landmark_columns].astype("float32").values
y = df["label"].values

# Encode labels --------------------------------------------------------------- NEW / CHANGED
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
joblib.dump(encoder, ENCODER_PATH)
print(f"✅ Saved LabelEncoder → {ENCODER_PATH}")
print("Label classes (order used by the model):", list(encoder.classes_))
assert set(encoder.classes_) == allowed, f"Unexpected classes: {encoder.classes_}"

# Feature scaling (per-feature standardization) ------------------------------ NEW / CHANGED
scaler = StandardScaler()
X = scaler.fit_transform(X).astype("float32")
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Saved StandardScaler → {SCALER_PATH}")

# Split
X_train, X_test, y_train_idx, y_test_idx = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# One-hot
y_train = tf.keras.utils.to_categorical(y_train_idx, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test_idx,  num_classes)

# ---------------- Build Model ----------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),   # 63 features
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=64)

# ---------------- Evaluate ----------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test accuracy: {acc:.4f}")

# Classification report
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_idx = np.argmax(y_pred_probs, axis=1)
print("\nClassification report:\n",
      classification_report(y_test_idx, y_pred_idx, target_names=encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test_idx, y_pred_idx)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------- Add predictions to CSV (full data) ----------------
full_probs = model.predict(X, verbose=0)
full_pred_idx = np.argmax(full_probs, axis=1)
confidences = np.max(full_probs, axis=1)
predicted_labels = encoder.inverse_transform(full_pred_idx)

df_out = df.copy()
df_out["predicted_label"] = predicted_labels
df_out["confidence"] = confidences
df_out.to_csv(PRED_CSV_PATH, index=False)
print(f"✅ CSV with predictions + confidence → {PRED_CSV_PATH}")

# ---------------- Save models ----------------
model.save(H5_PATH)
print(f"✅ Saved Keras model → {H5_PATH}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"✅ Saved TFLite model → {TFLITE_PATH} ({round(TFLITE_PATH.stat().st_size/1024, 2)} KB)")
