import os
import pandas as pd

# Folder path
data_dir = "C:/Users/Hari/OneDrive/Desktop/PROJECT_FINAL_YEAR/datasets/cremad"
files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

# Emotion mapping
emotion_map = {
    "NEU": 0,
    "HAP": 1,
    "SAD": 2,
    "ANG": 3,
    "FEA": 4,
    "DIS": 5
}

data = []
for f in files:
    parts = f.split("_")
    emotion_code = parts[2]  # 3rd part is emotion code
    label = emotion_map[emotion_code]
    data.append({"file": os.path.join(data_dir, f), "label": label})

# Convert to DataFrame
df = pd.DataFrame(data)

# Split train/test (e.g., 80/20)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Save CSV
train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
