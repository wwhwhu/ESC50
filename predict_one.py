import librosa
import numpy as np
import pandas as pd
from keras.models import load_model

data = pd.read_csv("dataset/predict/test.csv", header=0)[['target', 'category']]
print(data)
label_mapping = data.set_index('target')['category'].to_dict()
print(label_mapping)

freq = 128
time = 1251
n_fft = 1024
hop_length = 128

x, fs = librosa.load("wwh.wav", sr=32000)
print(x)
stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
print(stft.shape)
log_stft = librosa.power_to_db(stft)
# 将stft频谱转换为Mel频谱
melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128, sr=32000)
# melsp = np.reshape(melsp, (1, freq, time))
melsp = np.expand_dims(melsp, axis=0)
print("Input_size:", melsp.shape)

model = load_model("CNN_models/best.hdf5")
pred1 = model.predict(melsp)

model2 = load_model("CRNN_models/best.hdf5")
pred2 = model2.predict(melsp)

print(pred1.shape, pred2.shape)
print("CNN预测结果的向量表示：", pred1)
print("CRNN预测结果的向量表示：", pred2)
res1 = np.argmax(pred1, axis=1)
res2 = np.argmax(pred2, axis=1)
print(res1.shape, res2.shape)
print("CNN预测结果的最终表示：", res1[0])
print("CRNN预测结果的最终表示：", res2[0])

# 根据字典生成类别数组
labels1 = np.vectorize(label_mapping.get)(res1)
print("CNN预测类的最终表示：", labels1[0])
labels2 = np.vectorize(label_mapping.get)(res2)
print("CRNN预测类的最终表示：", labels2[0])