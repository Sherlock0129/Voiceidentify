import os
import librosa
import numpy as np
import torch
import torch.nn as nn
# 检查CUDA是否可用
print("CUDA可用:", torch.cuda.is_available())  # 预期输出 True
print("GPU数量:", torch.cuda.device_count())   # 预期输出 ≥1
print(torch.__version__)          # 查看 PyTorch 版本
print(torch.cuda.is_available())  # 必须输出 True 才能用 GP
import torch.nn.functional as F
from scipy.fftpack import  dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import math
import matplotlib
matplotlib.use('TkAgg')  # 强制使用Tkinter后端
# 中文显示
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =================================================================
# 1. MFCC 特征提取
# =================================================================
def extract_mfcc_manual(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40, return_full_mfcc=False):
    """提取 MFCC 特征，可选择返回完整矩阵或平均向量"""
    y, sr = librosa.load(audio_path, sr=8000)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)
    log_mel_energy = np.log(mel_energy + 1e-6)
    mfcc = dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]
    if return_full_mfcc:
        return mfcc.T  # 返回形状 (时间帧数, n_mfcc)
    return np.mean(mfcc, axis=1)

# MFCC 可视化
def visualize_mfcc(audio_path):
    """可视化音频文件的 MFCC 特征"""
    mfcc_full = extract_mfcc_manual(audio_path, return_full_mfcc=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc_full.T, x_axis='time', sr=8000, hop_length=int(8000 * 10 / 1000), cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC Heatmap', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('MFCC Coefficients', fontsize=12)
    plt.tight_layout()
    plt.show()

# Mel 频谱图可视化
def visualize_mel_spectrogram(audio_path, frame_length=25, frame_shift=10, n_fft=512, n_mels=40):
    """可视化音频文件的 Mel 频谱图"""
    y, sr = librosa.load(audio_path, sr=8000)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)
    log_mel_energy = np.log(mel_energy + 1e-6)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_mel_energy, x_axis='time', y_axis='mel', sr=sr, hop_length=frame_shift_samples, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Mel Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

# =================================================================
# 2. 特征归一化
# =================================================================
def normalize_features(X):
    """对特征进行归一化：减去均值并除以最大绝对值"""
    X = X.astype(np.float64)
    X -= np.mean(X, axis=0)
    max_vals = np.max(np.abs(X), axis=0)
    max_vals[max_vals == 0] = 1.0
    X /= max_vals
    return X

# 特征分布可视化
def visualize_feature_distribution(X_before, X_after, feature_idx=0):
    """可视化特征归一化前后的分布"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(X_before[:, feature_idx], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Feature {feature_idx} Before Normalization', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.subplot(1, 2, 2)
    plt.hist(X_after[:, feature_idx], bins=50, color='salmon', edgecolor='black')
    plt.title(f'Feature {feature_idx} After Normalization', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

# =================================================================
# 3. 加载数据集并进行填充
# =================================================================
def load_dataset(dataset_dir, max_frames=200):
    """加载数据集并提取完整 MFCC 特征，填充或截断至 max_frames"""
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
                    # 提取完整 MFCC 矩阵
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
                        print(f"    已加载: {file}")
                    else:
                        print(f"    特征提取失败: {file}")
                except Exception as e:
                    print(f"    加载失败: {file}, 错误: {e}")
    if len(X) == 0:
        raise ValueError("未找到有效数据！请检查数据集路径、FLAC 格式和目录结构。")
    X = np.array(X)  # 现在 X 的形状应为 (样本数, max_frames, n_mfcc)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# =================================================================
# 4. ECAPA-TDNN 模型定义
# =================================================================
class SEModule(nn.Module):
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
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
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
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
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
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out

class ECAPA_TDNN(nn.Module):
    def __init__(self, C=1024, n_class=40):  # 根据数据集调整 n_class
        super(ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(22, C, kernel_size=5, stride=1, padding=2)  # 输入通道数 = n_mfcc
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
        self.fc_out = nn.Linear(192, n_class)  # 分类输出层

    def forward(self, x):
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
# 5. 训练 ECAPA-TDNN 模型
# =================================================================
def train_ecapa_tdnn(X_train, y_train, num_classes, epochs=10, batch_size=32):
    """训练 ECAPA-TDNN 模型"""
    model = ECAPA_TDNN(C=1024, n_class=num_classes).cuda(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue  # 跳过不完整的批次
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            batch_X = torch.tensor(batch_X, dtype=torch.float32).cuda().permute(0, 2, 1)  # [batch, n_mfcc, time]
            batch_y = torch.tensor(batch_y, dtype=torch.long).cuda()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / (len(X_train) // batch_size)
        losses.append(epoch_loss)
        print(f'第 {epoch + 1}/{epochs} 轮, 损失: {epoch_loss:.4f}')

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', color='blue')
    plt.title('训练损失随轮次变化', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return model

# 每类准确率可视化
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
# 6. 主流程与可视化
# =================================================================
if __name__ == "__main__":
    dataset_dir = ".\\dev-clean\\LibriSpeech\\dev-clean"
    max_frames = 200  # 设置固定的帧数用于填充

    # 加载数据集并进行填充
    X, y, le = load_dataset(dataset_dir, max_frames=max_frames)
    print("特征矩阵形状 (样本数, 时间帧数, n_mfcc):", X.shape)

    # 可视化示例音频的 Mel 频谱图和 MFCC
    sample_audio_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[0],
                                     os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))[0],
                                     os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0],
                                                             os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))[0]))[0])
    visualize_mel_spectrogram(sample_audio_path)
    visualize_mfcc(sample_audio_path)

    # 逐样本归一化特征
    X_normalized = np.array([normalize_features(x) for x in X])

    # 可视化第一个样本的第一个特征的分布
    visualize_feature_distribution(X[0], X_normalized[0], feature_idx=0)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.05, stratify=y)

    # 训练 ECAPA-TDNN
    num_classes = len(np.unique(y))
    model = train_ecapa_tdnn(X_train, y_train, num_classes, epochs=10, batch_size=32)

    # 在评估模型部分添加随机扰动
    noise_level = 0.3  # 可以调整扰动的强度
    X_test_noisy = X_test + noise_level * np.random.randn(*X_test.shape)

    model.eval()
    with torch.no_grad():
        # 原始测试集评估
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda().permute(0, 2, 1)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

        print("原始测试集准确率:", accuracy_score(y_test, y_pred))
        print("\n原始测试集分类报告:")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        print("原始测试集F1 分数 (宏平均):", f1_score(y_test, y_pred, average='macro'))
        print("原始测试集F1 分数 (加权平均):", f1_score(y_test, y_pred, average='weighted'))
        print("\n原始测试集Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))
        print("原始测试集Matthews 相关系数 (MCC):", matthews_corrcoef(y_test, y_pred))

        # 添加扰动后的测试集评估
        X_test_noisy_tensor = torch.tensor(X_test_noisy, dtype=torch.float32).cuda().permute(0, 2, 1)
        outputs_noisy = model(X_test_noisy_tensor)
        y_pred_noisy = outputs_noisy.argmax(dim=1).cpu().numpy()

        print("\n添加扰动后测试集准确率:", accuracy_score(y_test, y_pred_noisy))
        print("\n添加扰动后测试集分类报告:")
        report_noisy = classification_report(y_test, y_pred_noisy, target_names=le.classes_, output_dict=True)
        print(classification_report(y_test, y_pred_noisy, target_names=le.classes_))
        print("添加扰动后测试集F1 分数 (宏平均):", f1_score(y_test, y_pred_noisy, average='macro'))
        print("添加扰动后测试集F1 分数 (加权平均):", f1_score(y_test, y_pred_noisy, average='weighted'))
        print("\n添加扰动后测试集Cohen's Kappa:", cohen_kappa_score(y_test, y_pred_noisy))
        print("添加扰动后测试集Matthews 相关系数 (MCC):", matthews_corrcoef(y_test, y_pred_noisy))

        # 计算鲁棒性指标（以准确率为例）
        robustness_accuracy = accuracy_score(y_test, y_pred) - accuracy_score(y_test, y_pred_noisy)
        print(f"模型鲁棒性（准确率差值）: {robustness_accuracy}")



    # # 可视化归一化混淆矩阵
    # cm = confusion_matrix(y_test, y_pred)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    # plt.title('归一化混淆矩阵', fontsize=14)
    # plt.xlabel('预测值', fontsize=12)
    # plt.ylabel('真实值', fontsize=12)
    # plt.tight_layout()
    # plt.show()
    #
    # # 可视化前 N 个类别的 F1 分数
    # f1_scores = {class_name: report[class_name]['f1-score'] for class_name in le.classes_}
    # sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
    # N = 10
    # top_N = sorted_f1[:N]
    # plt.figure(figsize=(12, 6))
    # plt.bar([x[0] for x in top_N], [x[1] for x in top_N], color='skyblue')
    # plt.title(f'前 {N} 个类别的 F1 分数', fontsize=14)
    # plt.xlabel('类别', fontsize=12)
    # plt.ylabel('F1 分数', fontsize=12)
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()
    #
    # # 可视化每类准确率
    # visualize_per_class_accuracy(y_test, y_pred, le)
    visualize_per_class_accuracy(y_test, y_pred_noisy, le)