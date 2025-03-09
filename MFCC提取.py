import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =================================================================
# 1. MFCC特征提取（按论文参数实现）
# =================================================================
def extract_mfcc_manual(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40):
    """
    按论文参数提取22阶MFCC
    :param audio_path: 音频路径
    :param n_mfcc: MFCC阶数（论文中为22）
    :param frame_length: 帧长（ms）
    :param frame_shift: 帧移（ms）
    :param n_fft: FFT点数
    :param n_mels: Mel滤波器数量
    :return: MFCC特征矩阵 (n_mfcc, T)
    """
    # 加载音频
    y, sr = librosa.load(audio_path, sr=8000)  # 论文中使用8kHz采样率

    # 1. 预加重（Pre-emphasis）
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 2. 分帧（Framing）
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)

    # 3. 加汉明窗（Hamming Window）
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)

    # 4. 傅里叶变换（DFT）
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))

    # 5. Mel滤波器组（Mel Filter Bank）
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)

    # 6. 对数能量
    log_mel_energy = np.log(mel_energy + 1e-6)

    # 7. DCT（离散余弦变换）
    mfcc = librosa.util.dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]

    return mfcc

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
    # 假设已加载数据（需替换为实际数据，如Aurora-2）
    # 示例：生成模拟数据（10个说话人，每人20个样本）
    n_speakers = 10
    n_samples_per_speaker = 20
    X = []
    y = []

    for spk_id in range(n_speakers):
        for _ in range(n_samples_per_speaker):
            # 模拟MFCC特征（22阶，100帧）
            mfcc = np.random.randn(22, 100).flatten()  # 实际应从音频提取
            X.append(mfcc)
            y.append(spk_id)

    X = np.array(X)
    y = np.array(y)

    # 特征归一化
    X = normalize_features(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练SVM
    model = train_svm(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))