import os
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# TensorFlow version
print("TensorFlow version:", tf.__version__)

# === Load and Merge CSVs ===
true_labels_file = "gesture_data_with_true_labels.csv"
main_file = "gesture_data_new.csv"

df_true = pd.read_csv(true_labels_file)
df_main = pd.read_csv(main_file)

df_combined = pd.concat([df_main, df_true], ignore_index=True)
df_combined.drop_duplicates(inplace=True)
df_combined = df_combined[df_combined["label"] != "okay"]  # ✅ Remove "okay"
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
df_combined.to_csv(main_file, index=False)
print("✅ Merged and removed 'okay' label.")

# === Prepare Data ===
df = df_combined
landmark_columns = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
x = df[landmark_columns].astype("float32").values
y = df['label'].values

# Encode Labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

# ✅ Save encoder for reuse in inference
joblib.dump(encoder, "gesture_label_encoder.pkl")
print("✅ Saved LabelEncoder as 'gesture_label_encoder.pkl'")
print("Label classes:", encoder.classes_)  # Debug

# Normalize input data
x /= np.max(x)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.2, random_state=42)

# One-hot encode
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# === Build Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Evaluate
model.evaluate(x_test, y_test)

# === Add Predictions to CSV ===
full_probs = model.predict(x)
predicted_indices = np.argmax(full_probs, axis=1)
confidences = np.max(full_probs, axis=1)
predicted_labels = encoder.inverse_transform(predicted_indices)

df['predicted_label'] = predicted_labels
df['confidence'] = confidences
df.to_csv("gesture_data_with_confidence.csv", index=False)
print("✅ CSV with predictions + confidence saved.")

# Save model
model.save("gesture_model.h5")
print("✅ Saved model as 'gesture_model.h5'")

# === Confusion Matrix ===
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# === TFLite Conversion ===
TF_LITE_MODEL_FILE_NAME = "../gesture_model_new.tflite"
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TF_LITE_MODEL_FILE_NAME, "wb") as f:
    f.write(tflite_model)
print(f"✅ Saved TFLite model: {TF_LITE_MODEL_FILE_NAME} ({round(os.path.getsize(TF_LITE_MODEL_FILE_NAME)/1024, 2)} KB)")
print("✅ Model training and TFLite conversion complete.")
