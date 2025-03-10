import os
import librosa
import numpy as np
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 强制使用Tkinter后端


# =================================================================
# 1. MFCC特征提取（严格按论文参数）
# =================================================================
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
# 2. 数据加载与预处理（按说话人划分）
# =================================================================
def load_speaker_data(speaker_dir, num_samples=5, required_dim=22, debug=True):
    """加载数据并强制统一维度"""
    mfcc_list = []
    for root, _, files in os.walk(speaker_dir):
        for file in files:
            if not file.endswith(".flac"):
                continue

            path = os.path.join(root, file)
            mfcc = extract_mfcc_manual(path)

            # 维度校验与修正
            if mfcc is not None:
                if len(mfcc) != required_dim:
                    print(f"强制修正维度: {path} ({len(mfcc)} -> {required_dim})")
                    mfcc = np.pad(mfcc, (0, required_dim - len(mfcc)))[:required_dim]
                mfcc_list.append(mfcc)

            if len(mfcc_list) >= num_samples:
                break

    return np.array(mfcc_list) if mfcc_list else None


def build_dataset(dataset_dir, target_speaker):
    """终极修复：保证数据维度一致性"""
    # 加载目标说话人
    target_data = load_speaker_data(os.path.join(dataset_dir, target_speaker), num_samples=20)
    if target_data is None or len(target_data) < 8:
        raise ValueError(f"目标说话人 {target_speaker} 数据不足")

    # 划分目标数据
    X_target_train = target_data[:5]  # (5,22)
    X_target_test = target_data[5:8]  # (3,22)

    # 加载冒名者数据
    X_impostor_train, X_impostor_test = [], []
    for speaker in os.listdir(dataset_dir):
        if speaker == target_speaker:
            continue

        data = load_speaker_data(os.path.join(dataset_dir, speaker), num_samples=10)
        if data is not None and len(data) >= 8:
            X_impostor_train.append(data[:5])  # (5,22)
            X_impostor_test.append(data[5:8])  # (3,22)

    # 安全合并
    def safe_concat(arrays):
        """自动填充/截断为统一维度"""
        max_dim = 22  # 论文指定维度
        processed = []
        for arr in arrays:
            if arr.shape[1] < max_dim:
                arr = np.pad(arr, ((0, 0), (0, max_dim - arr.shape[1])))
            elif arr.shape[1] > max_dim:
                arr = arr[:, :max_dim]
            processed.append(arr)
        return np.concatenate(processed, axis=0)

    try:
        X_train = safe_concat([X_target_train] + X_impostor_train)
        y_train = np.hstack([np.ones(len(X_target_train)), -np.ones(len(X_train) - len(X_target_train))])

        X_test = safe_concat([X_target_test] + X_impostor_test)
        y_test = np.hstack([np.ones(len(X_target_test)), -np.ones(len(X_test) - len(X_target_test))])
    except Exception as e:
        print(f"数据合并失败: {str(e)}")
        exit()

    # 合并训练数据
    X_train = np.vstack([X_target_train] + X_impostor_train)
    y_train = np.hstack([  # 使用hstack合并一维数组
        np.ones(len(X_target_train), dtype=np.int32),
        -np.ones(X_train.shape[0] - len(X_target_train), dtype=np.int32)
    ])

    # 合并测试数据
    X_test = np.vstack([X_target_test] + X_impostor_test)
    y_test = np.hstack([
        np.ones(len(X_target_test), dtype=np.int32),
        -np.ones(X_test.shape[0] - len(X_target_test), dtype=np.int32)
    ])

    return (X_train, y_train), (X_test, y_test)
# =================================================================
# 3. 归一化与模型训练（按论文方法）
# =================================================================
def train_speaker_model(X_train, y_train):
    # 计算类别权重
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == -1)
    class_weight = {1: n_neg / n_pos, -1: 1.0}  # 正样本权重提高

    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight=class_weight,  # 关键修改
        probability=True
    )
    model.fit(X_train, y_train)
    return model


# =================================================================
# 4. 评估指标计算（EER/FAR/FRR）
# =================================================================
def compute_eer(y_true, scores):
    """计算等错误率 (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold, fpr, fnr, thresholds


def plot_roc_curve(fpr, tpr, eer):
    """绘制ROC曲线并标记EER点"""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(eer, 1 - eer, color='red', label=f'EER = {eer:.2%}')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "FAR": far,
        "FRR": frr,
        "Accuracy": acc,
        "Threshold": threshold
    }
# =================================================================
# 5. 主流程（以LibriSpeech为例）
# =================================================================
if __name__ == "__main__":
    # 数据集路径（假设结构为：dataset/speaker_id/scene/*.flac）
    dataset_dir = "./dev-clean/LibriSpeech/dev-clean"

    # 随机选择一个目标说话人
    all_speakers = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    target_speaker = np.random.choice(all_speakers)
    print(f"目标说话人: {target_speaker}")


    # 构建数据集
    try:
        (X_train, y_train), (X_test, y_test) = build_dataset(dataset_dir, target_speaker)
        print(f"\n数据维度验证:")
        print(f"训练集: X.shape={X_train.shape}, y.shape={y_train.shape}")
        print(f"测试集: X.shape={X_test.shape}, y.shape={y_test.shape}")
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        exit()


    # 校验标签是否为纯一维数组
    def validate_labels(y):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim != 1:
            raise ValueError(f"标签维度错误: 期望1维，实际为{y.ndim}维")
        return y.astype(np.int32)


    y_train = validate_labels(y_train)
    y_test = validate_labels(y_test)

    # 打印最终校验结果
    print("标签维度验证:")
    print(f"y_train: shape={y_train.shape}, dtype={y_train.dtype}")
    print(f"y_test: shape={y_test.shape}, dtype={y_test.dtype}")
    # 构建数据集
    X, y = build_dataset(dataset_dir, target_speaker)
    # y = np.array(y, dtype=np.float32)
    print("y type:", type(y))
    # print("y shape:", np.shape(y))  # 避免直接调用 y.shape 导致 AttributeError
    print("y example:", y[:5])
    y = np.array(y[0], dtype=np.int32)

    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"Example X_train: {X_train[:2]}")
    print(f"Example y_train: {y_train[:10]}")
    # 划分训练集和测试集（按样本而非说话人）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    print("\n数据分布检查:")
    print(f"训练集 - 正样本: {np.sum(y_train == 1)}, 负样本: {np.sum(y_train == -1)}")
    print(f"测试集 - 正样本: {np.sum(y_test == 1)}, 负样本: {np.sum(y_test == -1)}")

    if np.sum(y_test == 1) < 5 or np.sum(y_test == -1) < 20:
        raise ValueError("测试集样本不足，请检查数据划分！")
    # 训练模型
    model, scaler, max_vals = train_speaker_model(X_train, y_train)

    # 测试集归一化
    X_test_normalized = scaler.transform(X_test) / max_vals

    # 预测得分（使用decision_function获取到超平面的距离）
    scores = model.decision_function(X_test_normalized)

    # 在计算EER前添加
    print("测试集分布：")
    print(f"正样本数: {np.sum(y_test == 1)}")
    print(f"负样本数: {np.sum(y_test == -1)}")

    # 计算EER
    eer, eer_thresh, fpr, fnr, thresholds = compute_eer(y_test, scores)
    print(f"EER: {eer * 100:.2f}%")

    # 计算FAR和FRR（在EER阈值下）
    y_pred = (scores >= eer_thresh).astype(int)
    far = np.sum((y_pred == 1) & (y_test == -1)) / np.sum(y_test == -1)
    frr = np.sum((y_pred == 0) & (y_test == 1)) / np.sum(y_test == 1)
    print(f"FAR: {far * 100:.2f}%  FRR: {frr * 100:.2f}%")

    # 使用示例
    # metrics = compute_metrics(y_test, scores, eer_threshold)
    # print(f"综合评估:\n"
    #       f"- 准确率: {metrics['Accuracy']:.2%}\n"
    #       f"- 接受率: {np.mean(scores >= eer_threshold):.2%}")

    # 绘制ROC曲线
    plot_roc_curve(fpr, 1 - fnr, eer)