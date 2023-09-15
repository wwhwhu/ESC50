import os

import librosa
import numpy as np
import pandas as pd
from keras.models import load_model

data = pd.read_csv("dataset/predict/test.csv", header=0)[['target', 'category']]
print(data)
label_mapping = data.set_index('target')['category'].to_dict()
print(label_mapping)
# label_mapping = {0: 'dog', 14: 'chirping_birds', 36: 'vacuum_cleaner', 19: 'thunderstorm', 30: 'door_wood_knock',
#                  34: 'can_opening', 9: 'crow', 22: 'clapping', 48: 'fireworks', 41: 'chainsaw', 47: 'airplane',
#                  31: 'mouse_click', 17: 'pouring_water', 45: 'train', 8: 'sheep', 15: 'water_drops', 46: 'church_bells',
#                  37: 'clock_alarm', 32: 'keyboard_typing', 16: 'wind', 25: 'footsteps', 4: 'frog', 3: 'cow',
#                  27: 'brushing_teeth', 43: 'car_horn', 12: 'crackling_fire', 40: 'helicopter', 29: 'drinking_sipping',
#                  10: 'rain', 7: 'insects', 26: 'laughing', 6: 'hen', 44: 'engine', 23: 'breathing', 20: 'crying_baby',
#                  49: 'hand_saw', 24: 'coughing', 39: 'glass_breaking', 28: 'snoring', 18: 'toilet_flush', 2: 'pig',
#                  35: 'washing_machine', 38: 'clock_tick', 21: 'sneezing', 1: 'rooster', 11: 'sea_waves', 42: 'siren',
#                  5: 'cat', 33: 'door_wood_creaks', 13: 'crickets'}

freq = 128
time = 1251
esc_dir = "dataset/"
predict_audio_dir = os.path.join(esc_dir, "predict/audio")

predict_data = pd.read_csv("dataset/predict/test.csv", header=0)[['filename']]
print("预测音频文件名", predict_data)
print("预测音频条数", predict_data.shape[0])

predict = list(predict_data.loc[:, "filename"])


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


# save wave data in npz, with augmentation
def save_np_data(filename, x, aug=None, rates=None):
    np_data = np.zeros(freq * time * len(x))
    np_data = np.reshape(np_data, (len(x), freq, time))
    for i in range(len(predict)):
        _x, fs = load_wave_data(predict_audio_dir, x[i])
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
    np.savez(filename, x=np_data)


# save raw training dataset
if not os.path.exists("dataset/predict/esc_melsp_predict_raw.npz"):
    save_np_data("dataset/predict/esc_melsp_predict_raw.npz", predict)

predict_file = "dataset/predict/esc_melsp_predict_raw.npz"
# load test dataset
predict = np.load(predict_file)
x_predict = predict["x"]
x_predict = np.reshape(x_predict, (predict_data.shape[0], freq, time, 1))

model = load_model("CNN_models/best.hdf5")
pred1 = model.predict(x_predict)

model2 = load_model("CRNN_models/best.hdf5")
pred2 = model2.predict(x_predict)
# pred = None
# for model_path in ["CNN_models/best.hdf5", "CRNN_models/best.hdf5"]:
#     model = load_model(model_path)
#     if pred is None:
#         pred = model.predict(x_predict)
#     else:
#         pred += model.predict(x_predict)

print(pred1.shape, pred2.shape)
print("CNN预测结果的向量表示：", pred1)
print("CRNN预测结果的向量表示：", pred2)
res1 = np.argmax(pred1, axis=1)
res2 = np.argmax(pred2, axis=1)
print(res1.shape, res2.shape)
print("CNN预测结果的最终表示：", res1)
print("CRNN预测结果的最终表示：", res2)

# 根据字典生成类别数组
labels1 = np.vectorize(label_mapping.get)(res1)
print("CNN预测类的最终表示：", labels1)
labels2 = np.vectorize(label_mapping.get)(res2)
print("CRNN预测类的最终表示：", labels2)

df = pd.DataFrame({"img_path": predict_data["filename"], "tags": res1, "class": labels1})
df.to_csv("dataset/predict/res/CNN.csv", index=False, header=True)

df = pd.DataFrame({"img_path": predict_data["filename"], "tags": res2, "class": labels2})
df.to_csv("dataset/predict/res/CRNN.csv", index=False, header=True)
