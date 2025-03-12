import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =================================================================
# 1. Feature Extraction and Dynamic Fusion
# =================================================================
def extract_mfcc(audio_path, n_mfcc=22, n_mels=40, sr=8000):
    """Extract MFCC features"""
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)
    return mfcc.T  # (time, n_mfcc)

def extract_mel_spectrogram(audio_path, n_mels=40, sr=8000):
    """Extract Mel spectrogram features"""
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return mel_spec.T  # (time, n_mels)

class FeatureFusion(nn.Module):
    """Dynamic fusion of MFCC and Mel spectrogram features"""
    def __init__(self, n_mfcc, n_mels):
        super(FeatureFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_mfcc + n_mels, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, mfcc, mel_spec):
        combined = torch.cat((mfcc, mel_spec), dim=-1)  # (batch, time, n_mfcc + n_mels)
        weights = self.attention(combined)  # (batch, time, 2)
        mfcc_weight = weights[:, :, 0].unsqueeze(-1)  # (batch, time, 1)
        mel_weight = weights[:, :, 1].unsqueeze(-1)  # (batch, time, 1)
        fused = mfcc * mfcc_weight + mel_spec * mel_weight  # (batch, time, n_mfcc or n_mels)
        return fused

# =================================================================
# 2. Conformer Module
# =================================================================
class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor=4, conv_expansion_factor=2, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion_factor, dim),
        )
        self.ln3 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_expansion_factor, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim * conv_expansion_factor, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )
        self.ln4 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + attn_output
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        residual = x
        x = self.ln3(x).permute(0, 2, 1)  # (batch, dim, time)
        x = self.conv(x).permute(0, 2, 1)  # (batch, time, dim)
        x = self.ln4(x + residual)
        return x

class Conformer(nn.Module):
    def __init__(self, input_dim, num_layers=3, dim=256, num_heads=4, dropout=0.1):
        super(Conformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)  # (batch, time, dim)
        for block in self.blocks:
            x = block(x)
        return x

# =================================================================
# 3. ECAPA-TDNN Module with Multi-Scale Convolution
# =================================================================
class MultiScaleBottle2neck(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7]):
        super(MultiScaleBottle2neck, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in scales
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(scales))
        self.relu = nn.ReLU()

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)  # (batch, out_channels * len(scales), time)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=256, C=512):
        super(ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, C, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(C)
        self.relu = nn.ReLU()
        self.layer1 = MultiScaleBottle2neck(C, C // 3)
        self.layer2 = MultiScaleBottle2neck(C, C // 3)
        self.layer3 = MultiScaleBottle2neck(C, C // 3)
        self.fc = nn.Conv1d(C * 3, C, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.fc(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        return x.mean(dim=2)  # (batch, C)

# =================================================================
# 4. Complete Model
# =================================================================
class SpeakerClassifier(nn.Module):
    def __init__(self, n_mfcc=22, n_mels=40, num_classes=40, dim=256):
        super(SpeakerClassifier, self).__init__()
        self.fusion = FeatureFusion(n_mfcc, n_mels)
        self.conformer = Conformer(n_mfcc + n_mels, num_layers=3, dim=dim)
        self.ecapa_tdnn = ECAPA_TDNN(in_channels=dim)
        self.gru = nn.GRU(dim, dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, mfcc, mel_spec):
        fused = self.fusion(mfcc, mel_spec)  # (batch, time, n_mfcc + n_mels)
        conformer_out = self.conformer(fused)  # (batch, time, dim)
        ecapa_out = self.ecapa_tdnn(conformer_out.permute(0, 2, 1))  # (batch, dim)
        gru_out, _ = self.gru(ecapa_out.unsqueeze(1))  # (batch, 1, dim)
        logits = self.fc(gru_out.squeeze(1))  # (batch, num_classes)
        return logits

# =================================================================
# 5. Data Loading and Augmentation
# =================================================================
def load_dataset(dataset_dir, max_frames=200):
    X_mfcc, X_mel, y = [], [], []
    for speaker in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if not os.path.isdir(chapter_dir):
                continue
            for file in os.listdir(chapter_dir):
                if file.endswith(".flac"):
                    audio_path = os.path.join(chapter_dir, file)
                    mfcc = extract_mfcc(audio_path)
                    mel = extract_mel_spectrogram(audio_path)
                    if mfcc.shape[0] > max_frames:
                        mfcc = mfcc[:max_frames]
                        mel = mel[:max_frames]
                    else:
                        mfcc = np.pad(mfcc, ((0, max_frames - mfcc.shape[0]), (0, 0)), mode='constant')
                        mel = np.pad(mel, ((0, max_frames - mel.shape[0]), (0, 0)), mode='constant')
                    X_mfcc.append(mfcc)
                    X_mel.append(mel)
                    y.append(speaker)
    return np.array(X_mfcc), np.array(X_mel), np.array(y)

# =================================================================
# 6. Training and Evaluation
# =================================================================
def train_model(model, X_mfcc_train, X_mel_train, y_train, epochs=10, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i in range(0, len(X_mfcc_train), batch_size):
            batch_mfcc = X_mfcc_train[i:i + batch_size]
            batch_mel = X_mel_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            batch_mfcc = torch.tensor(batch_mfcc, dtype=torch.float32).to(device)
            batch_mel = torch.tensor(batch_mel, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)  # Numeric labels

            optimizer.zero_grad()
            outputs = model(batch_mfcc, batch_mel)
            loss_ce = criterion(outputs, batch_y)

            # Simplified triplet loss (assumes batch_size >= 3)
            anchor = outputs[0].unsqueeze(0)
            positive = outputs[1].unsqueeze(0) if len(outputs) > 1 else anchor
            negative = outputs[2].unsqueeze(0) if len(outputs) > 2 else anchor
            loss_triplet = triplet_loss(anchor, positive, negative)

            loss = loss_ce + 0.1 * loss_triplet
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / (len(X_mfcc_train) // batch_size)
        losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return model

# Main execution
if __name__ == "__main__":
    # Dataset path (replace with your actual path)
    dataset_dir = "./dev-clean/LibriSpeech/dev-clean"
    X_mfcc, X_mel, y = load_dataset(dataset_dir)

    # Convert string labels to numeric labels
    le = LabelEncoder()
    y = le.fit_transform(y)  # y becomes numpy.int64

    # Split dataset
    X_mfcc_train, X_mfcc_test, X_mel_train, X_mel_test, y_train, y_test = train_test_split(
        X_mfcc, X_mel, y, test_size=0.3, stratify=y, random_state=42
    )

    # Initialize model
    num_classes = len(np.unique(y))
    model = SpeakerClassifier(n_mfcc=22, n_mels=40, num_classes=num_classes).to(device)

    # Train model
    model = train_model(model, X_mfcc_train, X_mel_train, y_train, epochs=10, batch_size=32)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_mfcc_test_tensor = torch.tensor(X_mfcc_test, dtype=torch.float32).to(device)
        X_mel_test_tensor = torch.tensor(X_mel_test, dtype=torch.float32).to(device)
        outputs = model(X_mfcc_test_tensor, X_mel_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in le.classes_]))