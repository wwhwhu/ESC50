# 训练后量化：https://www.tensorflow.org/lite/performance/post_training_quantization?hl=zh-cn
# 量化：https://www.tensorflow.org/lite/performance/model_optimization?hl=zh-cn
# Keras 中的修剪示例：https://tensorflow.google.cn/model_optimization/guide/pruning/pruning_with_keras
# tfmot sparsity：https://tensorflow.google.cn/model_optimization/api_docs/python/tfmot/sparsity
import os
import tempfile

import numpy as np
import tensorflow as tf

# 训练后量化是一种转换技术，它可以在改善 CPU 和硬件加速器延迟的同时缩减模型大小，且几乎不会降低模型准确率。
# 使用 TensorFlow Lite 转换器将已训练的浮点 TensorFlow 模型转换为 TensorFlow Lite 格式后，可以对该模型进行量化。

# 方法1: 动态量化 4x smaller, 2x-3x speedup CPU
# 训练后量化最简单的形式是仅将权重从浮点静态量化为整数（具有 8 位精度）
# 推断时，权重从 8 位精度转换为浮点，并使用浮点内核进行计算。此转换会完成一次并缓存，以减少延迟。
# 为了进一步改善延迟，“动态范围”算子会根据激活的范围将其动态量化为 8 位，并使用 8 位权重和激活执行计算。
# 此优化提供的延迟接近全定点推断。但是，输出仍使用浮点进行存储，因此使用动态范围算子的加速小于全定点计算。
from tensorflow_model_optimization import sparsity
import tensorflow_model_optimization as tfmot


def quant1(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant1_model = converter.convert()
    # 保存 TensorFlow Lite 模型
    with open('compression/CNN/quant1.tflite', 'wb') as f:
        f.write(tflite_quant1_model)


# 方法2：全整数量化 4x smaller, 3x+ speedup CPU, Edge TPU, Microcontrollers
# 通过确保所有模型数学均为整数量化，进一步改善延迟，减少峰值内存用量，以及兼容仅支持整数的硬件设备或加速器。
# 对于全整数量化，需要校准或估算模型中所有浮点张量的范围，即 (min, max)。
# 与权重和偏差等常量张量不同，模型输入、激活（中间层的输出）和模型输出等变量张量不能校准，除非我们运行几个推断周期。
# 因此，转换器需要一个有代表性的数据集来校准它们。这个数据集可以是训练数据或验证数据的一个小子集（大约 100-500 个样本）。
# 对模型进行全整数量化，但在模型没有整数实现时使用浮点算子（以确保转换顺利进行）
# 为了与原始的全浮点模型具有相同的接口，此 tflite_quant_model 不兼容仅支持整数的设备（如 8 位微控制器）和加速器（如 Coral Edge TPU），
# 因为输入和输出仍为浮点。
def quant2(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
    tflite_quant2_model = converter.convert()
    # 保存 TensorFlow Lite 模型
    with open('compression/CNN/quant2.0.tflite', 'wb') as f:
        f.write(tflite_quant2_model)


# 方法3：纯整数，对于适用于微控制器的 TensorFlow Lite 和 Coral Edge TPU，创建全整数模型是常见的用例。
# 为了确保兼容仅支持整数的设备（如 8 位微控制器）和加速器（如 Coral Edge TPU），可以使用以下步骤对包括输入和输出在内的所有算子强制执行全整数量化：
# 如果遇到当前无法量化的运算，转换器会引发错误。
def quant2_x(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant2x_model = converter.convert()
    # 保存 TensorFlow Lite 模型
    with open('compression/CNN/quant2.1.tflite', 'wb') as f:
        f.write(tflite_quant2x_model)


# 方法4：Float16量化 2x smaller, GPU acceleration	CPU, GPU
# 将权重量化为 float16（16 位浮点数的 IEEE 标准）来缩减浮点模型的大小
# float16 量化的优点如下：
#
# 将模型的大小缩减一半（因为所有权重都变成其原始大小的一半）。
# 实现最小的准确率损失。
# 支持可直接对 float16 数据进行运算的部分委托（例如 GPU 委托），从而使执行速度比 float32 计算更快。
# float16 量化的缺点如下：
#
# 它不像对定点数学进行量化那样减少那么多延迟。
# 默认情况下，float16 量化模型在 CPU 上运行时会将权重值“反量化”为 float32。（请注意，GPU 委托不会执行此反量化，因为它可以对 float16 数据进行运算。）
def quant3(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant3_model = converter.convert()
    # 保存 TensorFlow Lite 模型
    with open('compression/CNN/quant3.tflite', 'wb') as f:
        f.write(tflite_quant3_model)


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 加载 HDF5 模型
h5_model = tf.keras.models.load_model("CNN_models/best.hdf5")
h5_model.summary()
# 一些示例数据，表示代表性数据集，需要无标签的代表性样本
predict = np.load("data/esc_melsp_all_train_raw.npz")
data = predict["x"]
data = np.expand_dims(data, axis=-1)
labels = predict["y"]
labels = tf.keras.utils.to_categorical(labels, 50)
print(data.shape, labels.shape)


# # 创建 TensorFlow 数据集
# dataset = tf.data.Dataset.from_tensor_slices((data, labels))
#
# # 将数据集保存为 TFRecord 文件，用于整数量化训练
# tf.data.experimental.save(dataset, "train_dataset.tfrecord")

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(data).batch(1).take(1200):
        yield [tf.dtypes.cast(input_value, tf.float32)]


# 转换成 TensorFlow Lite 模型, 保存原始模型
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
tflite_model = converter.convert()
# 保存 TensorFlow Lite 模型
with open('compression/CNN/original.tflite', 'wb') as f:
    f.write(tflite_model)


# 量化
# quant1(converter)
# quant2(converter)
# quant2_x(converter)
# quant3(converter)

def get_gzipped_model_size(file, save_dir):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    with zipfile.ZipFile(save_dir, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(save_dir)


# 剪枝
# 剪枝操作Conv，使用 PolynomialDecay 函数修剪计划。
def prune1(model, train_data, train_labels, test_data, test_labels):
    # Compute end step to finish pruning after 2 epochs.
    x1 = get_gzipped_model_size("CNN_models/best.hdf5", "CNN_models/best_h5.zip")
    x2 = get_gzipped_model_size("compression/CNN/original.tflite", "compression/CNN/original_tf.zip")
    batch_size = 16
    epochs = 20
    print("train_data.shape", train_data.shape)
    print("train_label.shape", train_labels.shape)
    num_data = train_data.shape[0]
    end_step = np.ceil(num_data / batch_size).astype(np.int32) * epochs
    print("end_step: ", end_step)
    # Define model for pruning.
    pruning_params = {'pruning_schedule': sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                         final_sparsity=0.80,
                                                                         begin_step=0,
                                                                         end_step=end_step),
                      }

    # 定义剪枝方法Conv2D与batch剪枝
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Conv2D):
            print("find prune layer:", layer.name)
            return sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    # 模型转化为剪枝模型
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense
    )
    # model_for_pruning = sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.summary()
    # model_for_pruning.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-8, amsgrad=True)
    # 编译剪枝模型
    model_for_pruning.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    logdir = tempfile.mkdtemp()
    baseline_model_accuracy = model.evaluate(test_data, test_labels, verbose=0)
    # 剪枝微调训练
    model_for_pruning.fit(x=train_data, y=train_labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(test_data, test_labels),
                          callbacks=[sparsity.keras.UpdatePruningStep(),
                                     sparsity.keras.PruningSummaries(log_dir=logdir), ])

    # 准确性比较
    # Baseline test loss: 0.5280600190162659
    # Pruned test loss: 0.9467154145240784
    # Baseline test accuracy: 0.8525000214576721
    # Pruned test accuracy: 0.7850000262260437
    model_for_pruning_accuracy = model_for_pruning.evaluate(test_data, test_labels, verbose=0)
    print('Baseline test loss:', baseline_model_accuracy[0])
    print('Pruned test loss:', model_for_pruning_accuracy[0])
    print('Baseline test accuracy:', baseline_model_accuracy[1])
    print('Pruned test accuracy:', model_for_pruning_accuracy[1])

    # strip_pruning之后才能保存
    model_for_export = sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.summary()
    tf.keras.models.save_model(model_for_export, "compression/CNN/prune1_ex.hdf5", include_optimizer=False)
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter2.experimental_enable_resource_variables = True
    converter2.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    tflite_model2 = converter2.convert()
    with open("compression/CNN/prune1_ex.tflite", "wb") as f:
        f.write(tflite_model2)
    x3 = get_gzipped_model_size("compression/CNN/prune1_ex.hdf5", "compression/CNN/prune1_ex_h5.zip")
    x4 = get_gzipped_model_size("compression/CNN/prune1_ex.tflite", "compression/CNN/prune1_ex_tf.zip")
    print("Size of gzipped baseline Keras model: %.2f bytes" % x1) # 3529.18K
    print("Size of gzipped baseline TFlite model: %.2f bytes" % x2) # 902.63K
    print("Size of gzipped pruned Keras model: %.2f bytes" % x3) # 798.16K
    print("Size of gzipped pruned TFlite model: %.2f bytes" % x4) # 781.39K

def prune2(model, train_data, train_labels, test_data, test_labels):
    # Compute end step to finish pruning after 2 epochs.
    x1 = get_gzipped_model_size("CNN_models/best.hdf5", "CNN_models/best_h5.zip")
    x2 = get_gzipped_model_size("compression/CNN/original.tflite", "compression/CNN/original_tf.zip")
    batch_size = 16
    epochs = 20
    print("train_data.shape", train_data.shape)
    print("train_label.shape", train_labels.shape)
    num_data = train_data.shape[0]
    end_step = np.ceil(num_data / batch_size).astype(np.int32) * epochs
    print("end_step: ", end_step)
    # Define model for pruning.
    pruning_params = {'pruning_schedule': sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                         final_sparsity=0.80,
                                                                         begin_step=0,
                                                                         end_step=end_step),
                      }

    # 定义剪枝方法Conv2D与batch剪枝
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            print("find prune layer:", layer.name)
            return sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    # 模型转化为剪枝模型
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense
    )
    # model_for_pruning = sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.summary()
    # model_for_pruning.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-8, amsgrad=True)
    # 编译剪枝模型
    model_for_pruning.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    logdir = tempfile.mkdtemp()
    baseline_model_accuracy = model.evaluate(test_data, test_labels, verbose=0)
    # 剪枝微调训练
    model_for_pruning.fit(x=train_data, y=train_labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(test_data, test_labels),
                          callbacks=[sparsity.keras.UpdatePruningStep(),
                                     sparsity.keras.PruningSummaries(log_dir=logdir), ])

    # 准确性比较
    model_for_pruning_accuracy = model_for_pruning.evaluate(test_data, test_labels, verbose=0)
    # Baseline test loss: 0.5280612111091614
    # Pruned test loss: 0.9473150372505188
    # Baseline test accuracy: 0.8525000214576721
    # Pruned test accuracy: 0.7549999952316284
    print('Baseline test loss:', baseline_model_accuracy[0])
    print('Pruned test loss:', model_for_pruning_accuracy[0])
    print('Baseline test accuracy:', baseline_model_accuracy[1])
    print('Pruned test accuracy:', model_for_pruning_accuracy[1])

    # strip_pruning之后才能保存
    model_for_export = sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.summary()
    tf.keras.models.save_model(model_for_export, "compression/CNN/prune2_ex.hdf5", include_optimizer=False)
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter2.experimental_enable_resource_variables = True
    converter2.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    tflite_model2 = converter2.convert()
    with open("compression/CNN/prune2_ex.tflite", "wb") as f:
        f.write(tflite_model2)
    x3 = get_gzipped_model_size("compression/CNN/prune2_ex.hdf5", "compression/CNN/prune2_ex_h5.zip")
    x4 = get_gzipped_model_size("compression/CNN/prune2_ex.tflite", "compression/CNN/prune2_ex_tf.zip")
    print("Size of gzipped baseline Keras model: %.2f bytes" % x1) # 3529.18K
    print("Size of gzipped baseline TFlite model: %.2f bytes" % x2) # 902.63K
    print("Size of gzipped pruned Keras model: %.2f bytes" % x3) # 410.96K
    print("Size of gzipped pruned TFlite model: %.2f bytes" % x4) # 356.32K

test_d = np.load("data/esc_melsp_all_test.npz")
test_data = test_d["x"]
test_data = np.expand_dims(test_data, axis=-1)
test_labels = test_d["y"]
test_labels = tf.keras.utils.to_categorical(test_labels, 50)
print(test_data.shape, test_labels.shape)

# 剪枝conv2d+batchnormalization
# prune1(h5_model, data, labels, test_data, test_labels)

# 剪枝dense
prune2(h5_model, data, labels, test_data, test_labels)