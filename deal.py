# load metadata
import os

import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sounddevice as sd
from sklearn import model_selection

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

meta_data = pd.read_csv("dataset/ESC-50-master/meta/esc50.csv", header=0)[['filename', 'target']]
print(meta_data)
# get data size 2000条5s音频
data_size = meta_data.shape
print(data_size)
print(np.max)
audio_dir = "dataset/ESC-50-master/audio/"


# load a wave data加载音频
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=32000)
    # x:32000HZ*5s=160000
    # print("音频size", x.shape)
    # print("音频采样率", fs)
    # fs: 320000
    return x, fs


# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    # 使用短时傅里叶变换计算stft频谱,行数为（1 + n_fft/2），列数为160000/128
    # librosa.stft() 函数用于计算音频信号的短时傅里叶变换（Short-Time Fourier Transform, STFT）。
    # 它将音频信号分解成一系列短时窗口，并计算每个窗口的频谱信息。窗口大小为 win_length，默认等于n_fft
    # 列数表示频谱的帧数（Frames）：一般情况下，每个帧代表音频信号的一个短时窗口，在计算STFT时，会滑动这个窗口以覆盖整个音频信号。
    # 确定帧数（列数）：
    # n_frames = 1 + floor(n_samples / hop_length)，1+floor(160000/128)=1251。
    # 行数表示频率的分辨率（Resolution）：每个帧的频谱被划分为一系列频率区间，每个区间包含一个频率值。
    # 计算频率的分辨率（行数），指的是频谱中能够分辨出的频率间隔：
    # 如果未使用填充（默认情况下）：n_freq_bins = win_length // 2 + 1
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    # print("STFT频谱size", stft.shape)
    # 将功率频谱图（幅度平方）转换为分贝 （dB） 单位
    log_stft = librosa.power_to_db(stft)
    # 将stft频谱转换为Mel频谱
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128, sr=32000)
    # print("MEL频谱size", melsp.shape)
    return melsp


# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.title('波形图')
    plt.xlabel('采样次数')
    plt.ylabel('采样值（振幅）')
    plt.show()


# display wave in heatmap
def show_melsp(melsp, fs):
    # 将Mel频谱转换为对数刻度的dB单位
    mel_spec_db = melsp
    # mel_spec_db = librosa.power_to_db(melsp, ref=np.max)
    # 可视化Mel频谱
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=32000, hop_length=128,
                             x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title('Mel频谱')
    plt.xlabel('Time时间')
    plt.ylabel('Mel频率（Mel Frequency）')
    plt.tight_layout()
    plt.show()


# example data
x, fs = load_wave_data(audio_dir, meta_data.loc[10, "filename"])
mel_sp = calculate_melsp(x)
print(mel_sp)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x.shape, mel_sp.shape, fs))
freq = mel_sp.shape[0]
time = mel_sp.shape[1]
show_wave(x)
show_melsp(mel_sp, fs)
sd.play(x)
sd.wait()


# data augmentation: add white noise 添加白噪音
def add_white_noise(xx, rate=0.002):
    return xx + rate * np.random.randn(len(xx))


x_wn = add_white_noise(x)
melsp = calculate_melsp(x_wn)
print(melsp)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x_wn.shape, melsp.shape, fs))
show_wave(x_wn)
show_melsp(melsp, fs)
sd.play(x_wn)
sd.wait()


# data augmentation: shift sound in timeframe 向右移动一半
def shift_sound(xx, rate=2):
    return np.roll(xx, int(len(xx) // rate))


x_ss = shift_sound(x)
melsp2 = calculate_melsp(x_ss)
print(melsp2)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x_ss.shape, melsp2.shape, fs))
show_wave(x_ss)
show_melsp(melsp2, fs)
sd.play(x_ss)
sd.wait()


# data augmentation: stretch sound拉伸声音
def stretch_sound(xx, rate=1.1):
    input_length = len(xx)
    # rate=3,变为1/3,rate=0.333,变为3倍
    x = librosa.effects.time_stretch(xx, rate=rate)
    # 长则截断，短则填充
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


x_st = stretch_sound(x)
melsp3 = calculate_melsp(x_st)
print(melsp3)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x_st.shape, melsp3.shape, fs))
show_wave(x_st)
show_melsp(melsp3, fs)
sd.play(x_st)
sd.wait()

# get training dataset and target dataset
x = list(meta_data.loc[:, "filename"])
y = list(meta_data.loc[:, "target"])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, stratify=y)
print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train),
                                                                len(y_train),
                                                                len(x_test),
                                                                len(y_test)))
print("训练集\n", x_train, "\n", y_train)
print("测试集\n", x_test, "\n", y_test)


# save wave data in npz, with augmentation 保存为npz
def save_np_data(filename, x, y, freq, time, aug=None, rates=None):
    np_data = np.zeros(freq * time * len(x))
    np_data = np.reshape(np_data, (len(x), freq, time))
    # print(np_data.shape)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data(audio_dir, x[i])
        if aug is not None:
            # 添加白噪音
            _x = aug(xx=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        # print(_x.shape)
        np_data[i] = _x
        np_targets[i] = y[i]
    np.savez(filename, x=np_data, y=np_targets)


print(f"input size: ({freq},{time})")

# save raw training dataset
if not os.path.exists("data/esc_melsp_all_train_raw.npz"):
    save_np_data("data/esc_melsp_all_train_raw.npz", x_train, y_train, freq, time)

# save test dataset
if not os.path.exists("data/esc_melsp_all_test.npz"):
    save_np_data("data/esc_melsp_all_test.npz", x_test, y_test, freq, time)

# save training dataset with white noise 随机添加白噪音
if not os.path.exists("data/esc_melsp_train_white_noise.npz"):
    rates = np.random.randint(1, 50, len(x_train)) / 10000
    save_np_data("data/esc_melsp_train_white_noise.npz", x_train, y_train, freq, time, aug=add_white_noise, rates=rates)

# save training dataset with sound shift 随机添加平移,向右移动1/2-1/6
if not os.path.exists("data/esc_melsp_train_shift_sound.npz"):
    rates = np.random.choice(np.arange(2, 6), len(y_train))
    save_np_data("data/esc_melsp_train_shift_sound.npz", x_train, y_train, freq, time, aug=shift_sound, rates=rates)

# save training dataset with stretch 随机添加拉伸，遍为原来的0.8-1.2
if not os.path.exists("data/esc_melsp_train_stretch_sound.npz"):
    rates = np.random.choice(np.arange(80, 120), len(y_train)) / 100
    save_np_data("data/esc_melsp_train_stretch_sound.npz", x_train, y_train, freq, time, aug=stretch_sound, rates=rates)

# save training dataset with combination of white noise and shift or stretch
if not os.path.exists("data/esc_melsp_train_combination.npz"):
    np_data = np.zeros(freq * time * len(x_train))
    np_data = np.reshape(np_data, (len(x_train), freq, time))
    np_targets = np.zeros(len(y_train))
    for i in range(len(y_train)):
        x, fs = load_wave_data(audio_dir, x_train[i])
        x = add_white_noise(xx=x, rate=np.random.randint(1, 50) / 1000)
        if np.random.choice((True, False)):
            x = shift_sound(xx=x, rate=np.random.choice(np.arange(2, 6)))
        else:
            x = stretch_sound(xx=x, rate=np.random.choice(np.arange(80, 120)) / 100)
        x = calculate_melsp(x)
        np_data[i] = x
        np_targets[i] = y_train[i]
    np.savez("data/esc_melsp_train_combination.npz", x=np_data, y=np_targets)
