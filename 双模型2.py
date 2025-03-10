import os

import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from scipy.fftpack import dct

def extract_mfcc_manual(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40, min_duration=1.0,debug=True):
    """提取22阶MFCC（论文参数：帧长25ms，帧移10ms，40个Mel滤波器）"""
    y, sr = librosa.load(audio_path, sr=8000)  # 强制8kHz采样率

    # 检查音频长度是否足够
    if len(y) < int(sr * min_duration):
        print(f"警告：{audio_path} 过短（{len(y)/sr:.2f}s），已跳过")
        return None

    # 预加重 (Pre-emphasis)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 分帧 (Framing)
    frame_length_samples = int(sr * frame_length / 1000)
    frame_shift_samples = int(sr * frame_shift / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=frame_shift_samples)
    frames = frames.copy()

    # 加窗 (Hamming Window)
    frames *= np.hamming(frame_length_samples).reshape(-1, 1)

    # 傅里叶变换 (FFT)
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))

    # Mel滤波器组 (40个滤波器)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_basis, mag_spec)

    # 对数能量 & DCT (取前22阶)
    log_mel_energy = np.log(mel_energy + 1e-6)  # 避免log(0)
    mfcc = dct(log_mel_energy, axis=0, norm='ortho')[:n_mfcc]

    # 最终确保返回固定维度
    if mfcc.shape[0] != n_mfcc:
        print(f"错误：{audio_path} 特征维度异常 ({mfcc.shape})")
        return None
    # 沿时间轴取均值（论文方法）
    return np.mean(mfcc, axis=1)
# =================================================================
# 修改后的数据加载函数（按说话人组织数据）
# =================================================================
def load_speaker_data(dataset_dir):
    """
    返回字典格式：{speaker_id: [mfcc_features]}
    确保每个MFCC样本为二维数组（n_frames × n_coeffs）
    """
    speaker_data = {}
    for speaker in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        mfcc_list = []
        # 遍历场景文件夹
        for scene in os.listdir(speaker_dir):
            scene_dir = os.path.join(speaker_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            # 处理每个FLAC文件
            for file in os.listdir(scene_dir):
                if file.endswith(".flac"):
                    flac_path = os.path.join(scene_dir, file)
                    try:
                        mfcc = extract_mfcc_manual(flac_path)
                        if mfcc is not None:
                            # 确保MFCC是二维数组
                            if mfcc.ndim == 1:
                                mfcc = mfcc.reshape(1, -1)  # 单帧情况
                            mfcc_list.extend(mfcc)  # 使用extend展开帧
                    except Exception as e:
                        print(f"Error processing {flac_path}: {str(e)}")

        if mfcc_list:
            # 转换为二维numpy数组
            speaker_data[speaker] = np.array(mfcc_list)
            print(f"Loaded {len(mfcc_list)} frames from speaker {speaker}")
        else:
            print(f"Warning: No MFCC data for speaker {speaker}")

    # 过滤空数据
    return {k: v for k, v in speaker_data.items() if len(v) > 0}

# =================================================================
# 修改后的归一化方法（修复维度问题）
# =================================================================

def paper_normalization(train_data, test_data=None):
    """
    论文中的归一化方法：
    1. 确保输入数据为二维数组
    2. 减去每列均值
    3. 除以每列绝对最大值
    """
    # 确保数据维度正确
    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)

    # 训练阶段计算参数
    mu = np.mean(train_data, axis=0)
    max_vals = np.max(np.abs(train_data), axis=0)

    # 处理可能的零值（添加微小值避免除零）
    max_vals = np.where(max_vals == 0, 1e-8, max_vals)

    # 应用归一化
    norm_train = (train_data - mu) / max_vals

    if test_data is not None:
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        norm_test = (test_data - mu) / max_vals
        return norm_train, norm_test, mu, max_vals
    return norm_train, mu, max_vals


# =================================================================
# EER计算函数
# =================================================================
def compute_eer(y_true, scores):
    """
    计算等错误率(EER)
    """
    # 生成候选阈值
    thresholds = np.sort(scores)

    far_list = []
    frr_list = []

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0

        far_list.append(far)
        frr_list.append(frr)

    # 找到最接近EER的点
    eer_threshold = thresholds[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))]
    eer = (far_list[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))] +
           frr_list[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))]) / 2
    return eer, eer_threshold


# =================================================================
# 主流程（按论文方法训练每个说话人的模型）
# =================================================================
if __name__ == "__main__":
    # 加载数据（组织成字典形式）
    speaker_data = load_speaker_data("./dev-clean/LibriSpeech/dev-clean")

    # 遍历每个说话人进行训练
    for target_speaker in speaker_data.keys():
        print(f"\n=== Training model for speaker: {target_speaker} ===")

        # 获取正样本（目标说话人）
        positive_samples = speaker_data[target_speaker]

        # 获取负样本（其他所有说话人）
        negative_samples = []
        for speaker, data in speaker_data.items():
            if speaker != target_speaker:
                negative_samples.append(data)
        negative_samples = np.concatenate(negative_samples) if negative_samples else np.empty(
            (0, positive_samples.shape[1]))

        # 创建标签
        y_positive = np.ones(len(positive_samples))
        y_negative = np.zeros(len(negative_samples))

        # 合并数据集
        X = np.concatenate([positive_samples, negative_samples])
        y = np.concatenate([y_positive, y_negative])

        # 打乱数据
        X, y = shuffle(X, y, random_state=42)

        # 划分训练测试集（保持类别平衡）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 应用论文中的归一化方法
        X_train_norm, X_test_norm, mu, max_vals = paper_normalization(X_train, X_test)

        # 训练SVM（使用论文参数）
        model = SVC(
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',  # 处理类别不平衡
            probability=True  # 需要概率输出来计算EER
        )
        model.fit(X_train_norm, y_train)

        # 获取决策分数
        scores = model.decision_function(X_test_norm)

        # 计算EER
        eer, eer_threshold = compute_eer(y_test, scores)
        print(f"EER: {eer * 100:.2f}% (Threshold: {eer_threshold:.2f})")

        # 常规评估指标
        y_pred = model.predict(X_test_norm)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # 保存模型（可选）
        # joblib