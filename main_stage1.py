import os
import torch
import librosa
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
from glob import glob

# =============================
# CONFIG
# =============================
MODEL_NAME = "OthmaneJ/distil-wav2vec2"
SAMPLE_RATE = 16000
MAX_LEN = SAMPLE_RATE * 5  # 5 seconds max
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUG_PROB = 0.5
TOP_DB = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================
# EMOTION MAP
# =============================
EMOTION_MAP = {
    "NEU": 0,
    "HAP": 1,
    "SAD": 2,
    "ANG": 3,
    "FEA": 4,
    "DIS": 5,
}

ID2LABEL = {v: k for k, v in EMOTION_MAP.items()}
NUM_CLASSES = len(EMOTION_MAP)

# =============================
# AUDIO UTILS
# =============================
def remove_silence(audio):
    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)
    return audio

def normalize_audio(audio):
    m = np.max(np.abs(audio))
    return audio if m == 0 else audio / m

def augment_audio(audio, sr):
    n_steps = random.randint(-2, 2)
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    noise = 0.005 * np.random.randn(len(audio))
    return audio + noise

# =============================
# LOAD LOCAL CREMA-D
# =============================
def load_cremad(folder="datasets/cremad"):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]
    examples = []
    for path in files:
        fname = os.path.basename(path)
        emotion_code = fname.split("_")[2]
        if emotion_code in EMOTION_MAP:
            examples.append({"audio_path": path, "label": EMOTION_MAP[emotion_code]})
    return examples

print("⬇️ Loading CREMA-D...")
cremad = load_cremad()
print(f"CREMA-D samples: {len(cremad)}")

# =============================
# LOAD LOCAL RAVDESS
# =============================
def load_ravdess(local_dir="datasets/RAVDESS"):
    audio_paths = glob(os.path.join(local_dir, "**/*.wav"), recursive=True)
    data = []
    for path in audio_paths:
        label = int(os.path.basename(path).split("-")[2]) - 1
        if label in EMOTION_MAP.values():
            data.append({"audio_path": path, "label": label})
    return data

print("⬇️ Loading RAVDESS...")
ravdess = load_ravdess()
print(f"RAVDESS samples: {len(ravdess)}")

# =============================
# COMBINE DATA
# =============================
dataset = cremad + ravdess
random.shuffle(dataset)

split_idx = int(0.8 * len(dataset))
train_data = dataset[:split_idx]
test_data  = dataset[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Test samples : {len(test_data)}")

# =============================
# MODEL
# =============================
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=EMOTION_MAP
).to(DEVICE)

# =============================
# TORCH DATASET
# =============================
class SERDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]["audio_path"]
        label = self.data[idx]["label"]

        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio = remove_silence(audio)
        audio = normalize_audio(audio)

        if self.augment and random.random() < AUG_PROB:
            audio = augment_audio(audio, SAMPLE_RATE)

        audio = audio[:MAX_LEN]
        return audio, label

def collate_fn(batch):
    audios, labels = zip(*batch)
    inputs = processor(
        list(audios),
        sampling_rate=SAMPLE_RATE,
        padding=True,    # pad to batch max length
        return_tensors="pt"
    )
    
    input_values = inputs["input_values"]
    attention_mask = inputs.get("attention_mask")  # safe extraction
    
    return input_values, attention_mask, torch.tensor(labels)


train_loader = DataLoader(
    SERDataset(train_data, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    SERDataset(test_data, augment=False),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# =============================
# TRAINING
# =============================
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS * len(train_loader)
)

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for x, mask, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        if mask is not None:
            mask = mask.to(DEVICE)
            out = model(input_values=x, attention_mask=mask, labels=y)
        else:
            out = model(input_values=x, labels=y)
        
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


        losses.append(loss.item())
    print(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f}")

# =============================
# EVALUATION
# =============================
model.eval()
preds, true = [], []

with torch.no_grad():
    for x, mask, y in test_loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        out = model(input_values=x, attention_mask=mask)
        p = torch.argmax(out.logits, dim=-1)
        preds.extend(p.cpu().numpy())
        true.extend(y.numpy())

print("\n✅ Accuracy:", accuracy_score(true, preds))
print("Precision:", precision_score(true, preds, average="macro"))
print("Recall   :", recall_score(true, preds, average="macro"))
print("F1-score :", f1_score(true, preds, average="macro"))

print("\n📊 Classification Report:")
print(classification_report(true, preds, target_names=ID2LABEL.values()))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(true, preds))
