import os
import librosa
import numpy as np
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# =================================================================
# 1. MFCC特征提取（按论文参数实现）
# =================================================================
# =================================================================
# 1. 修复MFCC提取和归一化
# =================================================================
def extract_mfcc_manual(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40):
    """提取MFCC并展平为1D特征向量"""
    y, sr = librosa.load(audio_path, sr=8000)

    # 预加重
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 分帧和加窗
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)

    # 傅里叶变换和Mel滤波器组
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)

    # 对数能量和DCT
    log_mel_energy = np.log(mel_energy + 1e-6)
    mfcc = dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]

    # 展平为1D向量（时间轴取均值）
    return np.mean(mfcc, axis=1)  # 形状变为 (n_mfcc,)

# =================================================================
# 2. 特征归一化（按论文方法）
# =================================================================
def normalize_train_features(X_train_raw):
    """训练集归一化：计算均值和最大值，并返回归一化后的特征及参数"""
    # 减去均值
    mean_train = np.mean(X_train_raw, axis=0)
    X_train = X_train_raw - mean_train

    # 计算最大绝对值（避免零）
    max_vals_train = np.max(np.abs(X_train), axis=0)
    max_vals_train[max_vals_train == 0] = 1.0  # 防止除以零

    # 归一化
    X_train_normalized = X_train / max_vals_train
    return X_train_normalized, mean_train, max_vals_train

def normalize_test_features(X_test_raw, mean_train, max_vals_train):
    """测试集归一化：使用训练集的均值和最大值"""
    X_test = X_test_raw - mean_train
    X_test_normalized = X_test / max_vals_train
    return X_test_normalized
# =================================================================
# 2. 加载数据集并修复维度问题
# =================================================================
def load_dataset(dataset_dir):
    X, y = [], []
    for speaker in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            print(f"跳过非说话人目录: {speaker_dir}")
            continue

        print(f"处理说话人: {speaker}")

        # 遍历场景（Scene）文件夹
        for scene in os.listdir(speaker_dir):
            scene_dir = os.path.join(speaker_dir, scene)
            if not os.path.isdir(scene_dir):
                print(f"跳过非场景目录: {scene_dir}")
                continue

            print(f"  处理场景: {scene}")

            # 遍历场景中的FLAC文件
            for file in os.listdir(scene_dir):
                if not file.endswith(".flac"):
                    print(f"    跳过非FLAC文件: {file}")
                    continue

                flac_path = os.path.join(scene_dir, file)
                try:
                    mfcc = extract_mfcc_manual(flac_path)
                    if mfcc is not None:
                        X.append(mfcc)
                        y.append(speaker)
                        print(f"    成功加载: {file}")
                    else:
                        print(f"    特征提取失败: {file}")
                except Exception as e:
                    print(f"    加载失败: {file}, 错误: {e}")

    if len(X) == 0:
        raise ValueError("未找到有效数据！请检查：\n"
                         "1. 数据集路径是否正确\n"
                         "2. 文件是否为FLAC格式\n"
                         "3. 目录结构是否为 speaker/scene/files.flac")

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# =================================================================
# 3. 训练SVM模型（使用ERBF核）
# =================================================================
def train_svm(X_train, y_train):
    """
    训练SVM模型（论文中使用ERBF核，此处用RBF近似）
    :param X_train: 训练数据 (n_samples, n_features)
    :param y_train: 标签 (n_samples,)
    :return: 训练好的SVM模型
    """
    # 论文参数：ERBF核（此处用RBF替代）
    model = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# =================================================================
# 4. 主流程（模拟数据示例）
# =================================================================
if __name__ == "__main__":
    dataset_dir_train = "./dev-clean/LibriSpeech/dev-clean"
    X, y, le = load_dataset(dataset_dir_train)
    print("特征矩阵形状:", X.shape)

    # 按说话人划分数据集
    unique_speakers = np.unique(y)
    # 随机选择70%的说话人作为训练集，30%作为测试集
    train_speakers, test_speakers = train_test_split(
        unique_speakers,
        test_size=0.3,
        random_state=42
    )
    # 创建训练和测试集的掩码
    train_mask = np.isin(y, train_speakers)
    test_mask = np.isin(y, test_speakers)
    X_train_raw, y_train = X[train_mask], y[train_mask]
    X_test_raw, y_test = X[test_mask], y[test_mask]

    # 训练集归一化（计算参数）
    X_train, mean_train, max_vals_train = normalize_train_features(X_train_raw)

    # 测试集归一化（使用训练集的参数）
    X_test = normalize_test_features(X_test_raw, mean_train, max_vals_train)

    # 训练SVM模型
    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    print("准确率:", accuracy_score(y_test, y_pred))

    # 检查预测结果分布
    print("预测标签分布:", np.unique(y_pred, return_counts=True))
    print("真实标签分布:", np.unique(y_test, return_counts=True))

    # 生成分类报告（仅包含测试集中的说话人）
    test_speaker_names = le.classes_[test_speakers]
    print("\n分类报告:")
    print(classification_report(
        y_test,
        y_pred,
        labels=test_speakers,
        target_names=test_speaker_names,
        zero_division=0
    ))