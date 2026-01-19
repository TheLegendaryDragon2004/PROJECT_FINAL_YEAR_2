import os
import torch
from torch.utils.data import Dataset
from project_utils.label_mapping import RAVDESS_EMOTION_MAP, LABEL2ID
from data.preprocess import extract_features

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir, config, train=True):
        self.files = []
        for root, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".wav"):
                    self.files.append(os.path.join(root, f))

        self.config = config
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        emotion_code = file_path.split("-")[2]
        emotion = RAVDESS_EMOTION_MAP[emotion_code]
        label = LABEL2ID[emotion]

        features = extract_features(
            file_path,
            sr=self.config["sample_rate"],
            n_mfcc=self.config["n_mfcc"],
            max_len=self.config["max_len"],
            augment=self.train   # 🔥 FIX
        )

        return torch.from_numpy(features), label
