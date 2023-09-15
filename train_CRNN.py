# dataset files
import csv
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

from model.model import CRNN_model

train_files = ["data/esc_melsp_all_train_raw.npz", "data/esc_melsp_train_white_noise.npz",
               "data/esc_melsp_train_shift_sound.npz", "data/esc_melsp_train_stretch_sound.npz",
               "data/esc_melsp_train_combination.npz"]
# train_files = ["data/esc_melsp_all_train_raw.npz"]
test_file = "data/esc_melsp_all_test.npz"

# 单个npz的训练文件与测试文件大小
train_num = 1600
test_num = 400
print("train_num", train_num)
print("test_num", test_num)

# mel频谱的频域与时域信息
freq = 128
time = 1251

# define dataset placeholders
x_train = np.zeros(freq * time * train_num * len(train_files))
# 1600 * 128 * 1251
x_train = np.reshape(x_train, (train_num * len(train_files), freq, time))
y_train = np.zeros(train_num * len(train_files))

# load dataset
for i in range(len(train_files)):
    data = np.load(train_files[i])
    x_train[i * train_num:(i + 1) * train_num] = data["x"]
    y_train[i * train_num:(i + 1) * train_num] = data["y"]

# load test dataset
test_data = np.load(test_file)
x_test = test_data["x"]
y_test = test_data["y"]

print("训练集x大小", x_train.shape)
print("训练集y大小", y_train.shape)
print("测试集x大小", x_test.shape)
print("测试集y大小", y_test.shape)

# redefine target data into one hot vector one-hot编码
classes = 50
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

# reshape training dataset
x_train = np.reshape(x_train, (train_num * len(train_files), freq, time, 1))
x_test = np.reshape(x_test, (test_num, freq, time, 1))
print("\n\n训练集x大小", x_train.shape)
print("训练集y大小", y_train.shape)
print("测试集x大小", x_test.shape)
print("测试集y大小", y_test.shape)
print("model输入shape:", x_train.shape[1:], "model输出shape:", y_train.shape[1:][0])

# get CRNN model
model = CRNN_model(x_train.shape[1:], class_num=int(y_train.shape[1:][0]))
model.summary()

# directory for model checkpoints
model_dir = "CRNN_models\\"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# early stopping and model checkpoint# early
# 10轮验证集loss停止增长即停止训练
es_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1, mode='min')
chkpt = os.path.join(model_dir, 'esc50_.{epoch:02d}_{val_loss:.4f}_{val_accuracy:.4f}.hdf5')
# 保存模型文件，每轮都保存
cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=False, mode='min',
                        save_freq='epoch')

chkpt2 = os.path.join(model_dir, 'best.hdf5')
# 保存最佳模型文件
cp_cb2 = ModelCheckpoint(filepath=chkpt2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# 连续三轮未改善学习率降为原来0.6倍
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=3, verbose=1, mode="min")


# 保存每轮的loss,val-loss,acc,val-acc
# 定义自定义回调函数
class LogMetrics(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super(LogMetrics, self).__init__()
        self.log_file = log_file
        self.metrics = []

    # 在每个 epoch 结束时记录指标
    def on_epoch_end(self, epoch, logs=None):
        metrics_dict = {
            'epoch': epoch + 1,
            'loss': logs['loss'],
            'accuracy': logs['accuracy'],
            'val_loss': logs['val_loss'],
            'val_accuracy': logs['val_accuracy']
        }
        self.metrics.append(metrics_dict)

    # 在训练结束时保存指标为 CSV 文件
    def on_train_end(self, logs=None):
        with open(self.log_file, 'w', newline='') as file:
            fieldnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)


# 创建回调函数实例
log_metrics_cb = LogMetrics('CRNN_models\\metrics.csv')


# between class data generator
class MixupGenerator():
    def __init__(self, x_train, y_train, batch_size=16, alpha=0.2, shuffle=True):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(x_train)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                # 每次取出两组
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                x, y = self.__data_generation(batch_ids)
                yield x, y

    # __get_exploration_order用于获得数据的遍历顺序
    def __get_exploration_order(self):
        # index为1600
        indexes = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    # __data_generation 方法用于生成混合样本的输入和标签。
    # 首先，从训练集中选取出两组输入和标签，分别为 x1、x2、y1、y2。
    # 然后，从 Beta 分布中随机采样出一个混合比例 l。
    # 将 l 转换为与输入 x 和标签 y 的形状相匹配的数组。使用混合比例和两组输入、标签生成混合样本 x 和 y。
    # 最后，通过 yield 返回混合样本的输入 x 和标签 y
    def __data_generation(self, batch_ids):
        _, h, w, c = self.x_train.shape
        _, class_num = self.y_train.shape
        x1 = self.x_train[batch_ids[:self.batch_size]]
        x2 = self.x_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        x_l = np.reshape(l, (self.batch_size, 1, 1, 1))
        y_l = np.reshape(l, (self.batch_size, 1))
        x = x1 * x_l + x2 * (1 - x_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return x, y


# train model
batch_size = 16
alpha = 0.2
epochs = 1000
shuffle = True

training_generator = MixupGenerator(x_train, y_train, batch_size, alpha, shuffle)()
# model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size,
#           callbacks=[es_cb, cp_cb, cp_cb2, reduce_lr, log_metrics_cb], epochs=epochs, verbose=1, shuffle=True)
history = model.fit(training_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    callbacks=[es_cb, cp_cb, cp_cb2, reduce_lr, log_metrics_cb])

model.load_weights("CRNN_models\\best.hdf5")

score = model.evaluate(x_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# plot learning curves

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("CRNN_models\\acc.png")
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("CRNN_models\\loss.png")
plt.show()