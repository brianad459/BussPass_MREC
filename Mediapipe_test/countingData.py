import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the updated CSV with predictions and confidence
df = pd.read_csv("gesture_data_with_confidence.csv")

# Count each predicted label
label_counts = df['predicted_label'].value_counts()

# Average confidence for each predicted label
average_confidences = df.groupby('predicted_label')['confidence'].mean()

# Print counts
print("Gesture Counts:")
print(label_counts)

# Manually count each gesture (optional, but redundant with .value_counts())
count_rock = 0
count_paper = 0
count_scissors = 0
count_game = 0
label = None
for label in df['predicted_label']:
    if label == "rock":
        count_rock += 1
    elif label == "paper":
        count_paper += 1
    elif label == "scissor":
        count_scissors += 1
    elif label == "game":
        count_game += 1

print(f"Rock: {count_rock}")
print(f"Paper: {count_paper}")
print(f"Scissor: {count_scissors}")
print(f"Game: {count_game}")

# Print average confidence
print("\nAverage Confidence per Gesture:")
print(average_confidences)

average_confidences.plot(kind='barh', color='pink')

# Load CSV with true and predicted labels
df = pd.read_csv("gesture_data_with_confidence.csv")

# Calculate accuracy
correct = (df["true_label"] == df["predicted_label"]).sum()
total = len(df)
accuracy = correct / total

print(f"Accuracy: {accuracy:.2%}")

plt.title('Average Confidence per Gesture')
plt.xlabel('Confidence Score')
plt.ylabel('Gesture')
plt.xlim(0, 1)  # assuming confidence is between 0 and 1
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

