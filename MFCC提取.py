import os
import librosa
import numpy as np
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

# =================================================================
# 1. MFCC Feature Extraction
# =================================================================
def extract_mfcc_manual(audio_path, n_mfcc=22, frame_length=25, frame_shift=10, n_fft=512, n_mels=40, return_full_mfcc=False):
    """Extract MFCC features, optionally returning the full matrix or a flattened vector."""
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
        return mfcc.T
    return np.mean(mfcc, axis=1)

# Visualization for MFCC
def visualize_mfcc(audio_path):
    """Visualize the MFCC features of an audio file."""
    mfcc_full = extract_mfcc_manual(audio_path, return_full_mfcc=True)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc_full.T, x_axis='time', sr=8000, hop_length=int(8000 * 10 / 1000), cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC Heatmap', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('MFCC Coefficients', fontsize=12)
    plt.tight_layout()
    plt.show()

# Visualization for Mel Spectrogram
def visualize_mel_spectrogram(audio_path, frame_length=25, frame_shift=10, n_fft=512, n_mels=40):
    """Visualize the Mel spectrogram of an audio file."""
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
# 2. Feature Normalization
# =================================================================
def normalize_features(X):
    """Normalize features by subtracting the mean and dividing by the maximum absolute value."""
    X = X.astype(np.float64)
    X -= np.mean(X, axis=0)
    max_vals = np.max(np.abs(X), axis=0)
    max_vals[max_vals == 0] = 1.0
    X /= max_vals
    return X

# Visualization for Feature Distribution
def visualize_feature_distribution(X_before, X_after, feature_idx=0):
    """Visualize feature distribution before and after normalization."""
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
# 3. Load Dataset
# =================================================================
def load_dataset(dataset_dir):
    """Load dataset and extract MFCC features."""
    X, y = [], []
    for speaker in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker)
        if not os.path.isdir(speaker_dir):
            print(f"Skipping non-speaker directory: {speaker_dir}")
            continue
        print(f"Processing speaker: {speaker}")
        for scene in os.listdir(speaker_dir):
            scene_dir = os.path.join(speaker_dir, scene)
            if not os.path.isdir(scene_dir):
                print(f"Skipping non-scene directory: {scene_dir}")
                continue
            print(f"  Processing scene: {scene}")
            for file in os.listdir(scene_dir):
                if not file.endswith(".flac"):
                    print(f"    Skipping non-FLAC file: {file}")
                    continue
                flac_path = os.path.join(scene_dir, file)
                try:
                    mfcc = extract_mfcc_manual(flac_path)
                    if mfcc is not None:
                        X.append(mfcc)
                        y.append(speaker)
                        print(f"    Loaded: {file}")
                    else:
                        print(f"    Feature extraction failed: {file}")
                except Exception as e:
                    print(f"    Load failed: {file}, Error: {e}")
    if len(X) == 0:
        raise ValueError("No valid data found! Check dataset path, FLAC format, and directory structure.")
    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# =================================================================
# 4. Train SVM Model
# =================================================================
def train_svm(X_train, y_train):
    """Train an SVM model with RBF kernel."""
    model = SVC(kernel='rbf', gamma='scale', class_weight='balanced', probability=True)
    model.fit(X_train, y_train)
    return model

# Visualization for SVM Decision Boundary with Support Vectors
def visualize_svm_boundary(X_train, y_train, model, le):
    """Visualize SVM decision boundary and support vectors using PCA-reduced 2D features."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    support_indices = model.support_
    X_support_2d = X_train_2d[support_indices]
    h = 0.02
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='viridis', edgecolor='k', label='Training Points')
    plt.scatter(X_support_2d[:, 0], X_support_2d[:, 1], c='red', marker='x', s=50, label='Support Vectors')
    plt.colorbar(scatter, label='Speaker ID')
    plt.legend()
    plt.title('SVM Decision Boundary with Support Vectors (PCA 2D)', fontsize=14)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.tight_layout()
    plt.show()

# Visualization for Per-Class Accuracy
def visualize_per_class_accuracy(y_test, y_pred, le):
    """Visualize accuracy per class as a bar chart."""
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    class_names = le.classes_
    sorted_indices = np.argsort(per_class_acc)[::-1]
    sorted_acc = per_class_acc[sorted_indices]
    sorted_names = class_names[sorted_indices]
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_names, sorted_acc, color='skyblue')
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# =================================================================
# 5. Main Workflow with Visualizations
# =================================================================
if __name__ == "__main__":
    dataset_dir = "./dev-clean/LibriSpeech/dev-clean"

    # Load dataset
    X, y, le = load_dataset(dataset_dir)
    print("Feature matrix shape:", X.shape)

    # Visualize Mel Spectrogram and MFCC for a sample audio file
    sample_audio_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[0],
                                     os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))[0],
                                     os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0],
                                                             os.listdir(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))[0]))[0])
    # visualize_mel_spectrogram(sample_audio_path)
    visualize_mfcc(sample_audio_path)

    # Normalize features and visualize distribution
    X_before = X.copy()
    X = normalize_features(X)
    visualize_feature_distribution(X_before, X, feature_idx=0)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Train SVM
    model = train_svm(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("F1 Macro Average:", f1_score(y_test, y_pred, average='macro'))
    print("F1 Weighted Average:", f1_score(y_test, y_pred, average='weighted'))
    print("\nCohen's Kappa:", cohen_kappa_score(y_test, y_pred))
    print("Matthews Correlation Coefficient (MCC):", matthews_corrcoef(y_test, y_pred))

    # Visualize normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Visualize top N classes by F1-score
    f1_scores = {class_name: report[class_name]['f1-score'] for class_name in le.classes_}
    sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
    N = 10
    top_N = sorted_f1[:N]
    plt.figure(figsize=(12, 6))
    plt.bar([x[0] for x in top_N], [x[1] for x in top_N], color='skyblue')
    plt.title(f'Top {N} Classes by F1-Score', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Visualize per-class accuracy
    visualize_per_class_accuracy(y_test, y_pred, le)

    # Visualize SVM decision boundary with support vectors
    visualize_svm_boundary(X_train, y_train, model, le)