import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def plot_flac_waveform(file_path):
    # 读取FLAC文件
    data, samplerate = sf.read(file_path)

    # 计算时间轴
    time = [i / samplerate for i in range(len(data))]

    # 绘制波形图
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, label="Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("FLAC Audio Waveform")
    plt.legend()
    plt.grid()
    plt.show()

def plot_pitch(file_path):
    # 读取音频文件
    y, sr = librosa.load(file_path, sr=None)

    # 计算基频（F0）
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000)

    # 计算时间轴
    time = np.linspace(0, len(y) / sr, num=len(f0))

    # 绘制 Pitch 图
    plt.figure(figsize=(10, 4))
    plt.plot(time, f0, label="Pitch (F0)", color="blue", linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour")
    plt.ylim([50, 2000])
    plt.legend()
    plt.grid()
    plt.show()

# 示例调用
file_path = "./dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"  # 替换为你的FLAC文件路径
plot_flac_waveform(file_path)
plot_pitch(file_path)

