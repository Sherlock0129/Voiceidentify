import os
import librosa
import numpy as np
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
def normalize_features(features):
    """
    归一化：每列减去均值，除以最大值
    :param features: (n_samples, n_features)
    :return: 归一化后的特征
    """
    # 减去均值
    features -= np.mean(features, axis=0)
    # 除以最大值
    features /= np.max(np.abs(features), axis=0)
    return features

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

    # 归一化
    X = normalize_features(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 训练SVM
    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)

    # 添加随机扰动
    noise_level = 0.3  # 可以调整扰动的强度
    X_test_noisy = X_test + noise_level * np.random.randn(*X_test.shape)

    # 评估原始测试集
    y_pred = model.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred)
    print("原始测试集 Accuracy:", accuracy_original)
    print(classification_report(y_test, y_pred))

    # 评估添加扰动后的测试集
    y_pred_noisy = model.predict(X_test_noisy)
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
    print("添加扰动后测试集 Accuracy:", accuracy_noisy)
    print(classification_report(y_test, y_pred_noisy))

    # 计算鲁棒性
    robustness = accuracy_original - accuracy_noisy
    print(f"模型鲁棒性（准确率差值）: {robustness}")
