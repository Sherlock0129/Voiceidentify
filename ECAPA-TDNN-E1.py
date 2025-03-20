import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 优化后的 MFCC 特征提取
def extract_mfcc_manual(audio_path, n_mfcc=13, frame_length=25, frame_shift=10, n_fft=512, n_mels=23, return_full_mfcc=False):
    """提取 MFCC 特征，可选择返回完整矩阵或平均向量"""
    y, sr = librosa.load(audio_path, sr=8000)  # 采样率 8kHz
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])  # 预加重
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)  # 加窗
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)
    log_mel_energy = np.log(mel_energy + 1e-6)  # 对数运算
    mfcc = dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]
    if return_full_mfcc:
        return mfcc.T  # 返回 (时间帧数, n_mfcc)
    return np.mean(mfcc, axis=1)

# 2. 特征归一化（Z-score）
def normalize_features(X):
    """对特征进行 Z-score 归一化"""
    X = X.astype(np.float64)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # 防止除以零
    return (X - mean) / std

# 3. 加载数据集并填充/截断
def load_dataset(dataset_dir, max_frames=200):
    """加载数据集，提取完整 MFCC 特征并填充或截断至 max_frames"""
    X, y = [], []
    for speaker in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        print(f"处理说话者: {speaker}")
        for scene in os.listdir(speaker_dir):
            scene_dir = os.path.join(speaker_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for file in os.listdir(scene_dir):
                if not file.endswith(".flac"):
                    continue
                flac_path = os.path.join(scene_dir, file)
                try:
                    mfcc = extract_mfcc_manual(flac_path, return_full_mfcc=True)
                    if mfcc is not None:
                        # 填充或截断至 max_frames
                        if mfcc.shape[0] < max_frames:
                            pad_width = max_frames - mfcc.shape[0]
                            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
                        else:
                            mfcc = mfcc[:max_frames, :]
                        X.append(mfcc)
                        y.append(speaker)
                except Exception as e:
                    print(f"加载失败: {file}, 错误: {e}")
    if len(X) == 0:
        raise ValueError("未找到有效数据！请检查数据集路径和格式。")
    X = np.array(X)  # 形状: (样本数, max_frames, n_mfcc)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# 4. E-ECAPA-TDNN 模型定义
class SEModule(nn.Module):
    """Squeeze-Excitation 模块"""
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    """改进的瓶颈层"""
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(planes / scale)
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs, bns = [], []
        num_pad = int(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out

class E_ECAPA_TDNN(nn.Module):
    """Enhanced ECAPA-TDNN 模型"""
    def __init__(self, C=512, n_class=40):
        super(E_ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(13, C, kernel_size=5, stride=1, padding=2)  # 输入通道为 n_mfcc=13
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = Bottle2neck(C, C, kernel_size=3, dilation=5, scale=8)  # 新增一层
        self.layer5 = nn.Conv1d(4 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.dropout = nn.Dropout(0.3)  # 添加 dropout
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc_out = nn.Linear(192, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x4 = self.layer4(x + x1 + x2 + x3)  # 新增层
        x = self.layer5(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.relu(x)
        t = x.size()[-1]
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.fc_out(x)
        return x

# 5. 训练函数
def train_e_ecapa_tdnn(X_train, y_train, num_classes, epochs=100, batch_size=32):
    """训练 E-ECAPA-TDNN 模型"""
    model = E_ECAPA_TDNN(C=512, n_class=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    losses = []
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue
            batch_X = torch.tensor(X_train[batch_indices], dtype=torch.float32).to(device).permute(0, 2, 1)
            batch_y = torch.tensor(y_train[batch_indices], dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / (len(X_train) // batch_size)
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        print(f'第 {epoch + 1}/{epochs} 轮, 损失: {epoch_loss:.6f}, 学习率: {optimizer.param_groups[0]["lr"]:.6f}')

        # 早停
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break

    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='blue')
    plt.title('训练损失随轮次变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.grid()
    plt.show()
    return model

# 6. 主流程
if __name__ == "__main__":
    dataset_dir = "./dev-clean/LibriSpeech/dev-clean"  # 数据集路径
    max_frames = 200

    # 加载数据集
    X, y, le = load_dataset(dataset_dir, max_frames=max_frames)
    print("特征矩阵形状 (样本数, 时间帧数, n_mfcc):", X.shape)

    # 逐样本归一化特征
    X_normalized = np.array([normalize_features(x) for x in X])

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, stratify=y)

    # 训练模型
    num_classes = len(np.unique(y))
    model = train_e_ecapa_tdnn(X_train, y_train, num_classes, epochs=100, batch_size=32)

    # 评估模型
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device).permute(0, 2, 1)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    print("准确率:", accuracy_score(y_test, y_pred))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))