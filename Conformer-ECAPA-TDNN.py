import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import math

# 检查是否有可用的 GPU 并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# =================================================================
# 1. MFCC 特征提取模块
# =================================================================
def extract_mfcc_manual(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40, return_full_mfcc=False):
    """
    提取 MFCC 特征（手动实现）。
    参数:
        audio_path (str): 音频文件路径
        n_mfcc (int): MFCC 系数的数量，默认为 22
        frame_length (int): 帧长度（单位：毫秒），默认为 25ms
        frame_shift (int): 帧移（单位：毫秒），默认为 10ms
        n_fft (int): FFT 点数，默认为 512
        n_mels (int): Mel 滤波器数量，默认为 40
        return_full_mfcc (bool): 是否返回完整的 MFCC 矩阵（时间帧数, n_mfcc），默认为 False
    返回:
        np.ndarray: 若 return_full_mfcc=False，返回平均 MFCC 向量 (n_mfcc,)；
                    若 return_full_mfcc=True，返回完整 MFCC 矩阵 (时间帧数, n_mfcc)
    """
    y, sr = librosa.load(audio_path, sr=8000)  # 加载音频，采样率为 8000Hz
    pre_emphasis = 0.97  # 预加重系数
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])  # 预加重处理
    frame_length_samples = int(sr * frame_length / 1000)  # 帧长转换为样本数
    frame_shift_samples = int(sr * frame_shift / 1000)  # 帧移转换为样本数
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)  # 分帧
    frames = frames.copy() * np.hamming(frame_length_samples).reshape(-1, 1)  # 加汉明窗
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))  # 计算幅度谱
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # 创建 Mel 滤波器组
    mel_energy = np.dot(mel_basis, mag_spec)  # 计算 Mel 能量
    log_mel_energy = np.log(mel_energy + 1e-6)  # 对数变换，避免 log(0)
    mfcc = dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]  # 计算 MFCC
    if return_full_mfcc:
        return mfcc.T  # 返回形状 (时间帧数, n_mfcc)
    return np.mean(mfcc, axis=1)  # 返回平均 MFCC 向量 (n_mfcc,)

# 可视化 MFCC 特征
def visualize_mfcc(audio_path):
    """可视化音频文件的 MFCC 特征热图"""
    mfcc_full = extract_mfcc_manual(audio_path, return_full_mfcc=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc_full.T, x_axis='time', sr=8000, hop_length=int(8000 * 10 / 1000), cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC 热图', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('MFCC 系数', fontsize=12)
    plt.tight_layout()
    plt.show()

# 可视化 Mel 频谱图
def visualize_mel_spectrogram(audio_path, frame_length=25, frame_shift=10, n_fft=512, n_mels=40):
    """可视化音频文件的 Mel 频谱图"""
    y, sr = librosa.load(audio_path, sr=8000)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy() * np.hamming(frame_length_samples).reshape(-1, 1)
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)
    log_mel_energy = np.log(mel_energy + 1e-6)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_mel_energy, x_axis='time', y_axis='mel', sr=sr, hop_length=frame_shift_samples, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel 频谱图', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('Mel 频率', fontsize=12)
    plt.tight_layout()
    plt.show()

# =================================================================
# 2. 特征归一化模块
# =================================================================
def normalize_features(X):
    """
    对特征进行归一化：减去均值并除以最大绝对值。
    参数:
        X (np.ndarray): 输入特征矩阵，形状 (时间帧数, n_mfcc) 或 (样本数, n_mfcc)
    返回:
        np.ndarray: 归一化后的特征矩阵，形状与输入相同
    """
    X = X.astype(np.float64)
    X -= np.mean(X, axis=0)  # 减去均值
    max_vals = np.max(np.abs(X), axis=0)  # 计算最大绝对值
    max_vals[max_vals == 0] = 1.0  # 避免除以 0
    X /= max_vals  # 归一化
    return X

# 可视化特征分布
def visualize_feature_distribution(X_before, X_after, feature_idx=0):
    """可视化特征归一化前后的分布，直方图形式"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(X_before[:, feature_idx], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'特征 {feature_idx} 归一化前', fontsize=14)
    plt.xlabel('值', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.subplot(1, 2, 2)
    plt.hist(X_after[:, feature_idx], bins=50, color='salmon', edgecolor='black')
    plt.title(f'特征 {feature_idx} 归一化后', fontsize=14)
    plt.xlabel('值', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.tight_layout()
    plt.show()

# =================================================================
# 3. 数据加载模块
# =================================================================
def load_dataset(dataset_dir, max_frames=200):
    """
    加载数据集并提取完整 MFCC 特征，填充或截断至 max_frames。
    参数:
        dataset_dir (str): 数据集根目录
        max_frames (int): 最大时间帧数，默认为 200
    返回:
        X (np.ndarray): 特征矩阵，形状 (样本数, max_frames, n_mfcc)
        y (np.ndarray): 标签数组，形状 (样本数,)
        le (LabelEncoder): 标签编码器，用于将说话人 ID 转换为数字
    """
    X, y = [], []
    for speaker in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            print(f"跳过非说话者目录: {speaker_dir}")
            continue
        print(f"处理说话者: {speaker}")
        for scene in os.listdir(speaker_dir):
            scene_dir = os.path.join(speaker_dir, scene)
            if not os.path.isdir(scene_dir):
                print(f"跳过非场景目录: {scene_dir}")
                continue
            print(f"  处理场景: {scene}")
            for file in os.listdir(scene_dir):
                if not file.endswith(".flac"):
                    print(f"    跳过非 FLAC 文件: {file}")
                    continue
                flac_path = os.path.join(scene_dir, file)
                try:
                    mfcc = extract_mfcc_manual(flac_path, return_full_mfcc=True)
                    if mfcc is not None:
                        if mfcc.shape[0] < max_frames:
                            pad_width = max_frames - mfcc.shape[0]
                            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')  # 填充
                        else:
                            mfcc = mfcc[:max_frames, :]  # 截断
                        X.append(mfcc)
                        y.append(speaker)
                        print(f"    已加载: {file}")
                    else:
                        print(f"    特征提取失败: {file}")
                except Exception as e:
                    print(f"    加载失败: {file}, 错误: {e}")
    if len(X) == 0:
        raise ValueError("未找到有效数据！请检查数据集路径、FLAC 格式和目录结构。")
    X = np.array(X)  # 形状 (样本数, max_frames, n_mfcc)
    le = LabelEncoder()
    y = le.fit_transform(y)  # 将说话人 ID 转换为数字标签
    return X, y, le

# =================================================================
# 4. Conformer 模块（简化版）
# =================================================================
class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion_factor, conv_expansion_factor, dropout=0.1):
        """
        Conformer 块，包含注意力、前馈网络和卷积模块。
        参数:
            dim (int): 输入和输出的维度
            num_heads (int): 多头注意力的头数
            ff_expansion_factor (int): 前馈网络的扩展因子
            conv_expansion_factor (int): 卷积模块的扩展因子
            dropout (float): Dropout 比率，默认为 0.1
        """
        super(ConformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)  # 层归一化
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)  # 多头注意力
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(  # 前馈网络
            nn.Linear(dim, dim * ff_expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion_factor, dim),
        )
        self.ln3 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(  # 卷积模块
            nn.Conv1d(dim, dim * conv_expansion_factor, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim * conv_expansion_factor, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )
        self.ln4 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        前向传播。
        输入:
            x (torch.Tensor): 输入张量，形状 (batch, time, dim)
        返回:
            torch.Tensor: 输出张量，形状 (batch, time, dim)
        """
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x)  # 自注意力
        x = residual + attn_output
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)  # 前馈网络
        residual = x
        x = self.ln3(x).permute(0, 2, 1)  # (batch, dim, time)
        x = self.conv(x).permute(0, 2, 1)  # 卷积模块，恢复 (batch, time, dim)
        x = self.ln4(x + residual)
        return x

class Conformer(nn.Module):
    def __init__(self, input_dim, num_layers, dim, num_heads, ff_expansion_factor, conv_expansion_factor, dropout=0.1):
        """
        Conformer 模型，包含多个 Conformer 块。
        参数:
            input_dim (int): 输入特征维度（如 n_mfcc=22）
            num_layers (int): Conformer 块的数量
            dim (int): 内部维度（如 1024）
            num_heads (int): 多头注意力的头数
            ff_expansion_factor (int): 前馈网络扩展因子
            conv_expansion_factor (int): 卷积模块扩展因子
            dropout (float): Dropout 比率，默认为 0.1
        """
        super(Conformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim)  # 将输入维度映射到 dim
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(dim, num_heads, ff_expansion_factor, conv_expansion_factor, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        前向传播。
        输入:
            x (torch.Tensor): 输入张量，形状 (batch, time, n_mfcc)
        返回:
            torch.Tensor: 输出张量，形状 (batch, time, dim)
        """
        x = self.embedding(x)  # (batch, time, dim)
        for block in self.conformer_blocks:
            x = block(x)
        return x

# =================================================================
# 5. ECAPA-TDNN 模块
# =================================================================
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        """SE 模块（Squeeze-and-Excitation），用于通道注意力"""
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
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, scale=8):
        """
        Bottle2neck 模块，用于 ECAPA-TDNN。
        参数:
            inplanes (int): 输入通道数
            planes (int): 输出通道数
            kernel_size (int): 卷积核大小，默认为 3
            dilation (int): 扩张率，默认为 1
            scale (int): 多尺度分支数，默认为 8
        """
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs, bns = [], []
        num_pad = math.floor(kernel_size / 2) * dilation
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

class ECAPA_TDNN(nn.Module):
    def __init__(self, C=1024, n_class=40):
        """
        ECAPA-TDNN 模型，用于说话人识别。
        参数:
            C (int): 通道数，默认为 1024
            n_class (int): 分类数（说话人数量），默认为 40
        """
        super(ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(1024, C, kernel_size=5, stride=1, padding=2)  # 输入通道与 Conformer 输出匹配
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc_out = nn.Linear(192, n_class)

    def forward(self, x):
        """
        前向传播。
        输入:
            x (torch.Tensor): 输入张量，形状 (batch, dim=1024, time)
        返回:
            torch.Tensor: 输出张量，形状 (batch, n_class)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        t = x.size()[-1]
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.fc_out(x)
        return x

# =================================================================
# 6. 结合 Conformer 和 ECAPA-TDNN 的模型
# =================================================================
class Conformer_ECAPA_TDNN(nn.Module):
    def __init__(self, input_dim, num_classes, conformer_layers=1, dim=1024, num_heads=4, ff_expansion_factor=4, conv_expansion_factor=2, dropout=0.1, C=1024):
        """
        结合 Conformer 和 ECAPA-TDNN 的模型。
        参数:
            input_dim (int): 输入特征维度（如 n_mfcc=22）
            num_classes (int): 分类数（说话人数量）
            conformer_layers (int): Conformer 层数，默认为 1
            dim (int): Conformer 输出维度，默认为 1024
            num_heads (int): 多头注意力的头数，默认为 4
            ff_expansion_factor (int): 前馈网络扩展因子，默认为 4
            conv_expansion_factor (int): 卷积模块扩展因子，默认为 2
            dropout (float): Dropout 比率，默认为 0.1
            C (int): ECAPA-TDNN 通道数，默认为 1024
        """
        super(Conformer_ECAPA_TDNN, self).__init__()
        self.conformer = Conformer(input_dim, conformer_layers, dim, num_heads, ff_expansion_factor, conv_expansion_factor, dropout)
        self.ecapa_tdnn = ECAPA_TDNN(C=C, n_class=num_classes)

    def forward(self, x):
        """
        前向传播。
        输入:
            x (torch.Tensor): 输入张量，形状 (batch, time, n_mfcc)
        返回:
            torch.Tensor: 输出张量，形状 (batch, num_classes)
        """
        x = self.conformer(x)  # (batch, time, dim=1024)
        x = x.permute(0, 2, 1)  # (batch, dim=1024, time)
        x = self.ecapa_tdnn(x)  # (batch, num_classes)
        return x

# =================================================================
# 7. 模型训练模块
# =================================================================
def train_model(model, X_train, y_train, epochs=10, batch_size=128):
    """
    训练模型。
    参数:
        model (nn.Module): 要训练的模型
        X_train (np.ndarray): 训练特征，形状 (样本数, max_frames, n_mfcc)
        y_train (np.ndarray): 训练标签，形状 (样本数,)
        epochs (int): 训练轮次，默认为 10
        batch_size (int): 批次大小，默认为 128
    返回:
        nn.Module: 训练完成的模型
    """
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam 优化器，学习率降低
    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        indices = np.random.permutation(len(X_train))  # 随机打乱索引
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / (len(X_train) // batch_size)
        losses.append(epoch_loss)
        print(f'第 {epoch + 1}/{epochs} 轮, 损失: {epoch_loss:.4f}')

    # 可视化训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', color='blue')
    plt.title('训练损失随轮次变化', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return model

# 可视化每类准确率
def visualize_per_class_accuracy(y_test, y_pred, le):
    """以条形图形式可视化每类的准确率"""
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    class_names = le.classes_
    sorted_indices = np.argsort(per_class_acc)[::-1]
    sorted_acc = per_class_acc[sorted_indices]
    sorted_names = class_names[sorted_indices]
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_names, sorted_acc, color='skyblue')
    plt.title('每类准确率', fontsize=14)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# =================================================================
# 8. 主流程与评估
# =================================================================
if __name__ == "__main__":
    dataset_dir = "./dev-clean/LibriSpeech/dev-clean"  # 数据集路径
    max_frames = 200 # 最大时间帧数

    # 加载数据集
    X, y, le = load_dataset(dataset_dir, max_frames=max_frames)
    print("特征矩阵形状 (样本数, 时间帧数, n_mfcc):", X.shape)

    # 可视化示例音频
    sample_audio_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[0],
                                     os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))[0],
                                     os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0],
                                                             os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))[0]))[0])
    visualize_mel_spectrogram(sample_audio_path)
    visualize_mfcc(sample_audio_path)

    # 特征归一化
    X_normalized = np.array([normalize_features(x) for x in X])
    visualize_feature_distribution(X[0], X_normalized[0], feature_idx=0)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, stratify=y)

    # 初始化并训练模型
    num_classes = len(np.unique(y))
    model = Conformer_ECAPA_TDNN(input_dim=22, num_classes=num_classes, conformer_layers=1, dim=1024).to(device)
    model = train_model(model, X_train, y_train, epochs=30, batch_size=128)

    # 评估模型
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    print("准确率:", accuracy_score(y_test, y_pred))
    print("\n分类报告:")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("F1 分数 (宏平均):", f1_score(y_test, y_pred, average='macro'))
    print("F1 分数 (加权平均):", f1_score(y_test, y_pred, average='weighted'))
    print("\nCohen's Kappa:", cohen_kappa_score(y_test, y_pred))
    print("Matthews 相关系数 (MCC):", matthews_corrcoef(y_test, y_pred))

    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('归一化混淆矩阵', fontsize=14)
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('真实值', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 可视化前 N 个类别的 F1 分数
    f1_scores = {class_name: report[class_name]['f1-score'] for class_name in le.classes_}
    sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
    N = 10
    top_N = sorted_f1[:N]
    plt.figure(figsize=(12, 6))
    plt.bar([x[0] for x in top_N], [x[1] for x in top_N], color='skyblue')
    plt.title(f'前 {N} 个类别的 F1 分数', fontsize=14)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('F1 分数', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 可视化每类准确率
    visualize_per_class_accuracy(y_test, y_pred, le)