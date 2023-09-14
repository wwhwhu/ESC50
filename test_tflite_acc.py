import os

import numpy as np
import tensorflow as tf


def test_tflite_models(tflite_models, test_data):
    # 加载测试数据
    data = np.load(test_data)
    test_audio = data['x']
    test_labels = data['y']

    # 定义准确性计数器
    accuracy_scores = []

    # 遍历每个TFLite模型
    for model_path in tflite_models:
        # 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # 获取输入和输出张量
        input_details = interpreter.get_input_details()
        print('input_detail:', input_details)
        output_details = interpreter.get_output_details()
        print('output_detail:', output_details)

        # 预测并计算准确性
        correct = 0
        total = 0

        for i in range(len(test_audio)):
            # test_audio[i]: (128, 1251)->(1, 128, 1251, 1)
            # 设置输入
            input_data = np.expand_dims(test_audio[i], axis=0)
            input_data = np.expand_dims(input_data, axis=-1).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # 执行推理
            interpreter.invoke()

            # 获取输出
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # 计算准确性
            predicted_label = np.argmax(output_data)
            true_label = test_labels[i]
            if predicted_label == true_label:
                correct += 1
            total += 1

        accuracy = correct / total
        accuracy_scores.append(accuracy)

        print(f"Model: {model_path}")
        print(f"Accuracy: {accuracy}\n")

    return accuracy_scores


# tflite_models = ["compression/CNN/original.tflite", "compression/CNN/prune1_ex.tflite",
#                  "compression/CNN/prune2_ex.tflite", "compression/CNN/quant1.tflite", "compression/CNN/quant2.0.tflite",
#                  "compression/CNN/quant2.1.tflite", "compression/CNN/quant3.tflite"]
# test_data = "data/esc_melsp_all_test.npz"
#
# accuracy_scores = test_tflite_models(tflite_models, test_data)
#
# CNN_name = ['original.tflite', "prune1_ex.tflite", "prune2_ex.tflite", "quant1.tflite", "quant2.0.tflite",
#             "quant2.1.tflite", "quant3.tflite"]
# # 打印准确性得分
# # 0.8525 0.785 0.755 0.85 0.855 0.02 0.8525
# # CNN Model original.tflite Accuracy: 0.8525
# # CNN Model prune1_ex.tflite Accuracy: 0.785
# # CNN Model prune2_ex.tflite Accuracy: 0.755
# # CNN Model quant1.tflite Accuracy: 0.85
# # CNN Model quant2.0.tflite Accuracy: 0.855
# # CNN Model quant2.1.tflite Accuracy: 0.02
# # CNN Model quant3.tflite Accuracy: 0.8525
# for i, accuracy in enumerate(accuracy_scores):
#     print(f"CNN Model {CNN_name[i]} Accuracy: {accuracy}")

# 不屏蔽GPU会报错，目前不知道为什么
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tflite_models = ["compression/CRNN/original.tflite", "compression/CRNN/prune1_ex.tflite",
                 "compression/CRNN/prune2_ex.tflite", "compression/CRNN/quant1.tflite", "compression/CRNN/quant2.0.tflite",
                 "compression/CRNN/quant2.1.tflite", "compression/CRNN/quant3.tflite"]
test_data = "data/esc_melsp_all_test.npz"

accuracy_scores = test_tflite_models(tflite_models, test_data)

CRNN_name = ['original.tflite', "prune1_ex.tflite", "prune2_ex.tflite", "quant1.tflite", "quant2.0.tflite",
            "quant2.1.tflite", "quant3.tflite"]
# 打印准确性得分
# 0.68 0.625 0.5975 0.675 0.6675 0.02 0.68
# CRNN Model original.tflite Accuracy: 0.68
# CRNN Model prune1_ex.tflite Accuracy: 0.625
# CRNN Model prune2_ex.tflite Accuracy: 0.5975
# CRNN Model quant1.tflite Accuracy: 0.675
# CRNN Model quant2.0.tflite Accuracy: 0.6675
# CRNN Model quant2.1.tflite Accuracy: 0.02
# CRNN Model quant3.tflite Accuracy: 0.68
for i, accuracy in enumerate(accuracy_scores):
    print(f"CRNN Model {CRNN_name[i]} Accuracy: {accuracy}")
