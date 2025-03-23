import os
import time
import json
import librosa
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

import seaborn as sns
import math
import pandas as pd
from tqdm import tqdm
import argparse
import warnings
from sklearn.metrics import roc_curve, accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings("ignore")

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# =================================================================
# 1. 特征提取模块
# =================================================================
def extract_mfcc(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40, return_full_mfcc=False):
    """提取 MFCC 特征，可选择返回完整矩阵或平均向量

    参数:
        audio_path: 音频文件路径
        n_mfcc: MFCC系数数量
        frame_length: 帧长度(毫秒)
        frame_shift: 帧移(毫秒)
        n_fft: FFT大小
        n_mels: Mel滤波器组数量
        return_full_mfcc: 是否返回完整MFCC矩阵(而非平均值)

    返回:
        MFCC特征矩阵或平均向量
    """
    y, sr = librosa.load(audio_path, sr=8000)
    # 预加重
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 分帧
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()

    # 加窗
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)

    # 计算幅度谱
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))

    # 应用Mel滤波器组
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)

    # 取对数
    log_mel_energy = np.log(mel_energy + 1e-6)

    # DCT变换得到MFCC
    mfcc = dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]

    if return_full_mfcc:
        return mfcc.T  # 返回形状 (时间帧数, n_mfcc)
    return np.mean(mfcc, axis=1)  # 返回平均值


def extract_melspectrogram(audio_path, frame_length=25, frame_shift=10, n_fft=512, n_mels=40, return_full_mel=True):
    """提取Mel频谱图特征

    参数:
        audio_path: 音频文件路径
        frame_length: 帧长度(毫秒)
        frame_shift: 帧移(毫秒)
        n_fft: FFT大小
        n_mels: Mel滤波器组数量
        return_full_mel: 是否返回完整Mel频谱图(而非平均值)

    返回:
        Mel频谱图特征矩阵或平均向量
    """
    y, sr = librosa.load(audio_path, sr=8000)

    # 预加重
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 分帧参数
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)

    # 分帧
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()

    # 加窗
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)

    # 计算幅度谱
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))

    # 应用Mel滤波器组
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)

    # 取对数
    log_mel_energy = np.log(mel_energy + 1e-6)

    if return_full_mel:
        return log_mel_energy.T  # 返回形状 (时间帧数, n_mels)
    return np.mean(log_mel_energy, axis=1)  # 返回平均值


# =================================================================
# 2. 特征处理与数据加载
# =================================================================
def normalize_features(X):
    """对特征进行归一化：减去均值并除以最大绝对值

    参数:
        X: 输入特征矩阵

    返回:
        归一化后的特征矩阵
    """
    X = X.astype(np.float64)
    X -= np.mean(X, axis=0)
    max_vals = np.max(np.abs(X), axis=0)
    max_vals[max_vals == 0] = 1.0
    X /= max_vals
    return X


class SpeakerDataset(Dataset):
    """说话人数据集类"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(dataset_dir, feature_type='mfcc', max_frames=200, n_mfcc=22, n_mels=40):
    """加载数据集并提取特征

    参数:
        dataset_dir: 数据集目录
        feature_type: 特征类型，'mfcc'或'melspectrogram'
        max_frames: 最大帧数(用于填充或截断)
        n_mfcc: MFCC系数数量
        n_mels: Mel滤波器组数量

    返回:
        (X, y, le): 特征矩阵，标签，标签编码器
    """
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
                    # 根据特征类型提取特征
                    if feature_type == 'mfcc':
                        features = extract_mfcc(flac_path, n_mfcc=n_mfcc, return_full_mfcc=True)
                    else:  # melspectrogram
                        features = extract_melspectrogram(flac_path, n_mels=n_mels, return_full_mel=True)

                    if features is not None:
                        # 填充或截断至 max_frames
                        if features.shape[0] < max_frames:
                            pad_width = max_frames - features.shape[0]
                            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
                        else:
                            features = features[:max_frames, :]

                        X.append(features)
                        y.append(speaker)

                except Exception as e:
                    print(f"    加载失败: {file}, 错误: {e}")

    if len(X) == 0:
        raise ValueError("未找到有效数据！请检查数据集路径和目录结构。")

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le


# =================================================================
# 3. 基础模块定义 - 用于模型变体构建
# =================================================================
class SEModule(nn.Module):
    """Squeeze-and-Excitation模块

    参数:
        channels: 输入通道数
        bottleneck: 瓶颈维度大小
        se_type: SE模块类型，'standard'或'efficient'
    """

    def __init__(self, channels, bottleneck=128, se_type='standard'):
        super(SEModule, self).__init__()
        self.se_type = se_type

        if se_type == 'standard':
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        elif se_type == 'efficient':
            # 高效版本：减少计算量
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        elif se_type == 'none':
            # 消融实验：移除SE模块
            self.se = None

    def forward(self, input):
        if self.se_type == 'none':
            return input

        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    """Res2Net的Bottle2neck模块

    参数:
        inplanes: 输入通道数
        planes: 输出通道数
        kernel_size: 卷积核大小
        dilation: 膨胀率
        scale: 分支数量
        use_se: 是否使用SE模块
        se_type: SE模块类型
    """

    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, scale=8, use_se=True, se_type='standard'):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        self.width = width

        # 是否使用分支结构
        self.use_branches = scale > 1

        if self.use_branches:
            convs = []
            bns = []
            num_pad = math.floor(kernel_size / 2) * dilation

            for i in range(self.nums):
                convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
                bns.append(nn.BatchNorm1d(width))

            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()

        # 是否使用SE模块
        self.use_se = use_se
        if use_se:
            self.se = SEModule(planes, se_type=se_type)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        if self.use_branches:
            spx = torch.split(out, self.width, 1)
            for i in range(self.nums):
                if i == 0:
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp)
                sp = self.relu(sp)
                sp = self.bns[i](sp)
                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)

            # 添加最后一个分支
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        # 应用SE模块
        if self.use_se:
            out = self.se(out)

        # 残差连接
        out += residual
        return out


# =================================================================
# 4. ECAPA-TDNN模型变体定义
# =================================================================
class ECAPA_TDNN_Base(nn.Module):
    """ECAPA-TDNN基础模型"""

    def __init__(self, feature_dim=22, C=1024, n_class=40):
        super(ECAPA_TDNN_Base, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 时序统计池化
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        # 注意力加权统计池化
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


class ECAPA_TDNN_NoSE(nn.Module):
    """ECAPA-TDNN模型 - 移除SE模块"""

    def __init__(self, feature_dim=22, C=1024, n_class=40):
        super(ECAPA_TDNN_NoSE, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8, use_se=False)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8, use_se=False)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8, use_se=False)
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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 时序统计池化
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        # 注意力加权统计池化
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


class ECAPA_TDNN_NoDilation(nn.Module):
    """ECAPA-TDNN模型 - 移除膨胀卷积"""

    def __init__(self, feature_dim=22, C=1024, n_class=40):
        super(ECAPA_TDNN_NoDilation, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=1, scale=8)  # 无膨胀
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=1, scale=8)  # 无膨胀
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=1, scale=8)  # 无膨胀
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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 时序统计池化
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        # 注意力加权统计池化
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


class ECAPA_TDNN_NoRes2Net(nn.Module):
    """ECAPA-TDNN模型 - 使用普通卷积代替Res2Net结构"""

    def __init__(self, feature_dim=22, C=1024, n_class=40):
        super(ECAPA_TDNN_NoRes2Net, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        # 使用普通残差块代替Res2Net
        self.layer1 = self._make_layer(C, C, dilation=2)
        self.layer2 = self._make_layer(C, C, dilation=3)
        self.layer3 = self._make_layer(C, C, dilation=4)

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

    def _make_layer(self, inplanes, planes, dilation=1):
        """创建普通残差卷积层"""
        return nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm1d(planes),
            SEModule(planes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x) + x
        x2 = self.layer2(x1) + x1
        x3 = self.layer3(x2) + x2

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 时序统计池化
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        # 注意力加权统计池化
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


class ECAPA_TDNN_NoAttentionPooling(nn.Module):
    """ECAPA-TDNN模型 - 移除注意力加权统计池化，改用简单时序统计池化"""

    def __init__(self, feature_dim=22, C=1024, n_class=40):
        super(ECAPA_TDNN_NoAttentionPooling, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(3072)
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

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 简单时序统计池化 (无注意力机制)
        mu = torch.mean(x, dim=2)
        sg = torch.sqrt(torch.var(x, dim=2) + 1e-4)
        x = torch.cat((mu, sg), 1)

        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


class ECAPA_TDNN_SmallC(nn.Module):
    """ECAPA-TDNN模型 - 减少特征图数量(C参数)"""

    def __init__(self, feature_dim=22, C=512, n_class=40):  # C=512 instead of 1024
        super(ECAPA_TDNN_SmallC, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, C * 3 // 2, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(C * 9 // 2, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Conv1d(128, C * 3 // 2, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(C * 3)
        self.fc6 = nn.Linear(C * 3, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc_out = nn.Linear(192, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 时序统计池化
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        # 注意力加权统计池化
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


class ECAPA_TDNN_SmallScale(nn.Module):
    """ECAPA-TDNN模型 - 减小Res2Net的分支数量"""

    def __init__(self, feature_dim=22, C=1024, n_class=40):
        super(ECAPA_TDNN_SmallScale, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=4)  # scale=4 instead of 8
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=4)  # scale=4 instead of 8
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=4)  # scale=4 instead of 8
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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 时序统计池化
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        # 注意力加权统计池化
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        embedding = x  # 提取嵌入向量
        x = self.fc6(x)
        x = self.bn6(x)
        logits = self.fc_out(x)

        return logits, embedding  # 返回 logits 和 embedding


# =================================================================
# 5. 训练、评估和可视化函数
# =================================================================

def compute_eer(scores, labels):
    """计算 EER"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, model_name="base"):
    """训练模型并记录性能

    参数:
        model: 待训练模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
        device: 训练设备
        model_name: 模型名称(用于记录)

    返回:
        history: 训练历史记录(包含损失和准确率)
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            inputs = inputs.to(device).float().permute(0, 2, 1)  # [batch, n_features, time]
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(inputs)  # Extract logits
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(logits, 1)
            train_corrects += torch.sum(preds == labels.data)
            train_total += labels.size(0)

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                inputs = inputs.to(device).float().permute(0, 2, 1)  # [batch, n_features, time]
                labels = labels.to(device)

                logits, _ = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(logits, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        # 计算平均损失和准确率
        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_corrects.double() / train_total
        epoch_val_acc = val_corrects.double() / val_total

        # 记录历史
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        # 打印统计信息
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {time_elapsed:.0f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f"best_model_{model_name}.pth")
            print(f"Saved best model with val acc: {best_val_acc:.4f}")

    return history

def evaluate_model(model, test_loader, criterion, device, le):
    """评估模型性能

    参数:
        model: 待评估模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        le: 标签编码器

    返回:
        结果字典，包含各项指标（包括 EER）
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device).float().permute(0, 2, 1)
            labels = labels.to(device)

            logits, embeddings = model(inputs)  # 获取 logits 和 embedding
            loss = criterion(logits, labels)

            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    # 计算平均损失
    test_loss = test_loss / len(test_loader.dataset)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 计算F1分数
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算 EER
    embeddings = np.concatenate(all_embeddings, axis=0)  # 合并所有嵌入
    unique_speakers = np.unique(all_labels)
    scores = []
    labels_eer = []

    # 模拟说话人验证：每个说话人第一个样本为注册样本，其余为测试样本
    for speaker in unique_speakers:
        speaker_indices = [i for i, lbl in enumerate(all_labels) if lbl == speaker]
        if len(speaker_indices) < 2:
            continue  # 跳过只有一个样本的说话人
        enroll_idx = speaker_indices[0]  # 注册样本
        test_indices = speaker_indices[1:]  # 测试样本

        enroll_embedding = embeddings[enroll_idx]
        # 正样本对
        for test_idx in test_indices:
            test_embedding = embeddings[test_idx]
            score = np.dot(enroll_embedding, test_embedding) / (
                np.linalg.norm(enroll_embedding) * np.linalg.norm(test_embedding)
            )  # 余弦相似度
            scores.append(score)
            labels_eer.append(1)  # 正样本

        # 负样本对：与其他说话人的注册样本比较
        other_speakers = [s for s in unique_speakers if s != speaker]
        for other in other_speakers[:5]:  # 限制负样本数量以减少计算开销
            other_indices = [i for i, lbl in enumerate(all_labels) if lbl == other]
            if other_indices:
                other_embedding = embeddings[other_indices[0]]
                score = np.dot(enroll_embedding, other_embedding) / (
                    np.linalg.norm(enroll_embedding) * np.linalg.norm(other_embedding)
                )
                scores.append(score)
                labels_eer.append(0)  # 负样本

    eer = compute_eer(scores, labels_eer) if scores else 0.0

    return {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'eer': eer
    }
# def evaluate_model(model, test_loader, criterion, device, le):
#     """评估模型性能
#
#     参数:
#         model: 待评估模型
#         test_loader: 测试数据加载器
#         criterion: 损失函数
#         device: 设备
#         le: 标签编码器
#
#     返回:
#         结果字典，包含各项指标
#     """
#     model.eval()
#     test_loss = 0.0
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Evaluating"):
#             inputs = inputs.to(device).float().permute(0, 2, 1)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             test_loss += loss.item() * inputs.size(0)
#
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     # 计算平均损失
#     test_loss = test_loss / len(test_loader.dataset)
#
#     # 计算准确率
#     accuracy = accuracy_score(all_labels, all_preds)
#
#     # 计算F1分数
#     f1_macro = f1_score(all_labels, all_preds, average='macro')
#     f1_weighted = f1_score(all_labels, all_preds, average='weighted')
#
#     # 生成分类报告
#     report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)
#
#     # 混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#
#     return {
#         'test_loss': test_loss,
#         'accuracy': accuracy,
#         'f1_macro': f1_macro,
#         'f1_weighted': f1_weighted,
#         'report': report,
#         'confusion_matrix': cm,
#         'predictions': all_preds,
#         'labels': all_labels
#     }


def visualize_training_history(histories, names):
    """可视化多个模型的训练历史

    参数:
        histories: 训练历史记录列表
        names: 模型名称列表
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 绘制训练损失
    for history, name in zip(histories, names):
        axs[0, 0].plot(history['train_loss'], label=name)
    axs[0, 0].set_title('训练损失')
    axs[0, 0].set_xlabel('轮次')
    axs[0, 0].set_ylabel('损失')
    axs[0, 0].legend()

    # 绘制验证损失
    for history, name in zip(histories, names):
        axs[0, 1].plot(history['val_loss'], label=name)
    axs[0, 1].set_title('验证损失')
    axs[0, 1].set_xlabel('轮次')
    axs[0, 1].set_ylabel('损失')
    axs[0, 1].legend()

    # 绘制训练准确率
    for history, name in zip(histories, names):
        axs[1, 0].plot(history['train_acc'], label=name)
    axs[1, 0].set_title('训练准确率')
    axs[1, 0].set_xlabel('轮次')
    axs[1, 0].set_ylabel('准确率')
    axs[1, 0].legend()

    # 绘制验证准确率
    for history, name in zip(histories, names):
        axs[1, 1].plot(history['val_acc'], label=name)
    axs[1, 1].set_title('验证准确率')
    axs[1, 1].set_xlabel('轮次')
    axs[1, 1].set_ylabel('准确率')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_history_comparison.png')
    plt.show()


def visualize_model_comparison(results, names):
    """可视化多个模型的性能比较

    参数:
        results: 评估结果列表
        names: 模型名称列表
    """
    # 提取各模型的准确率和F1分数
    accuracies = [result['accuracy'] for result in results]
    f1_macros = [result['f1_macro'] for result in results]
    f1_weighteds = [result['f1_weighted'] for result in results]

    # 创建条形图
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, accuracies, width, label='准确率')
    ax.bar(x, f1_macros, width, label='F1 (宏平均)')
    ax.bar(x + width, f1_weighteds, width, label='F1 (加权平均)')
    ax.bar(x + 2 * width, [1 - result['eer'] for result in results], width, label='准确率 (1-EER)')

    ax.set_ylabel('分数')
    ax.set_title('模型性能比较')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    # 创建表格进行详细比较
    comparison_data = []
    for i, name in enumerate(names):
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{accuracies[i]:.4f}",
            'F1 (Macro)': f"{f1_macros[i]:.4f}",
            'F1 (Weighted)': f"{f1_weighteds[i]:.4f}",
            'Test Loss': f"{results[i]['test_loss']:.4f}",
            'EER': f"{results[i]['eer']:.4f}"
        })

    df = pd.DataFrame(comparison_data)
    print("模型性能比较:")
    print(df.to_string(index=False))

    # 保存比较结果到CSV
    df.to_csv('model_comparison.csv', index=False)


# =================================================================
# 6. 主函数 - 运行消融实验
# =================================================================
def main(args):
    """主函数 - 运行消融实验

    参数:
        args: 命令行参数
    """
    print("=" * 50)
    print("开始ECAPA-TDNN模型消融实验")
    print("=" * 50)

    # 加载数据集
    print(f"加载数据集: {args.dataset_dir}")
    X, y, le = load_dataset(
        args.dataset_dir,
        feature_type=args.feature_type,
        max_frames=args.max_frames,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels
    )

    print(f"特征矩阵形状: {X.shape}")
    print(f"说话人类别数: {len(np.unique(y))}")

    # 特征归一化
    X_normalized = np.array([normalize_features(x) for x in X])

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, stratify=y, random_state=42)

    # 创建数据加载器
    train_dataset = SpeakerDataset(X_train, y_train)
    test_dataset = SpeakerDataset(X_test, y_test)

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 设置实验模型列表
    num_classes = len(np.unique(y))
    feature_dim = args.n_mfcc if args.feature_type == 'mfcc' else args.n_mels

    models = {
        'base': ECAPA_TDNN_Base(feature_dim=feature_dim, C=args.C, n_class=num_classes),
        'no_se': ECAPA_TDNN_NoSE(feature_dim=feature_dim, C=args.C, n_class=num_classes),
        'no_dilation': ECAPA_TDNN_NoDilation(feature_dim=feature_dim, C=args.C, n_class=num_classes),
        'no_res2net': ECAPA_TDNN_NoRes2Net(feature_dim=feature_dim, C=args.C, n_class=num_classes),
        'no_attention_pooling': ECAPA_TDNN_NoAttentionPooling(feature_dim=feature_dim, C=args.C, n_class=num_classes),
        'small_c': ECAPA_TDNN_SmallC(feature_dim=feature_dim, C=args.C // 2, n_class=num_classes),
        'small_scale': ECAPA_TDNN_SmallScale(feature_dim=feature_dim, C=args.C, n_class=num_classes)
    }

    # 选择要训练的模型
    if args.models == 'all':
        selected_models = list(models.keys())
    else:
        selected_models = args.models.split(',')

    # 训练历史记录和评估结果
    histories = []
    results = []
    names = []

    # 对每个选定的模型进行训练和评估
    for model_name in selected_models:
        if model_name not in models:
            print(f"警告: 未知模型 '{model_name}'，跳过")
            continue

        print(f"\n{'=' * 20} 训练模型: {model_name} {'=' * 20}")

        model = models[model_name].to(device)

        # 模型大小分析
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 优化器和学习率调度器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            device=device,
            model_name=model_name
        )

        # 评估模型
        print(f"\n{'=' * 20} 评估模型: {model_name} {'=' * 20}")
        result = evaluate_model(model, test_loader, criterion, device, le)

        # 打印评估结果
        print(f"测试损失: {result['test_loss']:.4f}")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"F1分数 (宏平均): {result['f1_macro']:.4f}")
        print(f"F1分数 (加权平均): {result['f1_weighted']:.4f}")
        print(f"EER: {result['eer']:.4f}")
        # 保存结果
        histories.append(history)
        results.append(result)
        names.append(model_name)

        # 保存模型和结果
        if args.save_results:
            torch.save(model.state_dict(), f"{model_name}_final.pth")

            # 保存评估结果
            with open(f"{model_name}_results.json", 'w') as f:
                result_copy = result.copy()
                result_copy.pop('confusion_matrix')  # 移除无法JSON序列化的NumPy数组
                result_copy.pop('predictions')
                result_copy.pop('labels')
                json.dump(result_copy, f, indent=4)

    # 可视化比较结果
    if len(histories) > 1:
        visualize_training_history(histories, names)
        visualize_model_comparison(results, names)

        # 额外实验：特征重要性分析
        if args.feature_importance and 'base' in selected_models:
            print("\n特征重要性分析...")
            # TODO: 实现特征重要性分析
            # 此处可以添加特征重要性分析代码


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECAPA-TDNN模型消融实验")

    # 数据集参数
    parser.add_argument('--dataset_dir', type=str, default="./dev-clean/LibriSpeech/dev-clean", help="数据集目录")
    parser.add_argument('--feature_type', type=str, default="mfcc", choices=['mfcc', 'melspectrogram'], help="特征类型")
    parser.add_argument('--max_frames', type=int, default=200, help="最大帧数(用于填充或截断)")
    parser.add_argument('--n_mfcc', type=int, default=22, help="MFCC系数数量")
    parser.add_argument('--n_mels', type=int, default=40, help="Mel滤波器组数量")

    # 模型参数
    parser.add_argument('--C', type=int, default=1024, help="ECAPA-TDNN中的通道数")
    parser.add_argument('--models', type=str, default="all", help="要训练的模型(逗号分隔)或'all'")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help="批量大小")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮数")
    parser.add_argument('--lr', type=float, default=0.001, help="学习率")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="权重衰减")
    parser.add_argument('--lr_step', type=int, default=10, help="学习率调整步长")
    parser.add_argument('--lr_gamma', type=float, default=0.5, help="学习率衰减因子")

    # 其他参数
    parser.add_argument('--save_results', action='store_true', help="是否保存结果")
    parser.add_argument('--feature_importance', action='store_true', help="是否执行特征重要性分析")

    args = parser.parse_args()
    main(args)