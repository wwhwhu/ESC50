import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Reshape, LSTM, Permute, Dense, Flatten, \
    Activation


def CRNN_model(shape, class_num=50):
    model = tf.keras.Sequential()
    # 添加卷积层1
    # 1*128*1251->30*126*1249
    model.add(Conv2D(30, kernel_size=(3, 3), padding='valid', activation='relu', name='conv1', input_shape=shape))
    # 添加BatchNormalize1
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 30*126*1249->30*42*624
    model.add(MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool1'))
    model.add(Dropout(0.1, name='dropout1'))

    # 添加卷积层2
    # 30*42*624->60*40*622
    model.add(Conv2D(60, kernel_size=(3, 3), padding='valid', activation='relu', name='conv2'))
    # 添加BatchNormalize2
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 60*40*622->60*20*311
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))
    model.add(Dropout(0.1, name='dropout2'))

    # 添加卷积层3
    # 60*20*309->60*18*309
    model.add(Conv2D(60, kernel_size=(3, 3), padding='valid', activation='relu', name='conv3'))
    # 添加BatchNormalize1
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 60*18*309->60*6*154
    model.add(MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool3'))
    model.add(Dropout(0.1, name='dropout3'))

    # 添加卷积层4
    # 60*6*154->60*4*152
    model.add(Conv2D(60, kernel_size=(3, 3), padding='valid', activation='relu', name='conv4'))
    # 添加BatchNormalize1
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 60*4*152->60*1*50
    model.add(MaxPooling2D(pool_size=(4, 3), strides=(4, 3), name='pool4'))
    model.add(Dropout(0.1, name='dropout4'))
    # 添加LSTM层
    # 变成channel*height*width
    # 60*1*50->50*60*1
    model.add(Permute((3, 1, 2)))
    # 50*60*1->50*60(17对应time信息)
    model.add(Reshape((50, 60)))
    # 50*60->50*60
    model.add(LSTM(60, return_sequences=True, name='LSTM1'))
    # 50*60->60(最后一个时间步)
    model.add(LSTM(60, return_sequences=False, name='LSTM2'))
    model.add(Dropout(0.3, name='dropout5'))
    # 添加线性层
    # 60->50
    model.add(Dense(class_num, name='Linear'))
    model.add(Activation('softmax'))
    # initiate Adam optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-8, amsgrad=True)

    # Let's train the model using Adam with amsgrad
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def CNN_k2c2_model(shape, class_num=50):
    model = tf.keras.Sequential()
    # 添加卷积层1
    # 1*128*1251->20*126*1249
    model.add(Conv2D(20, kernel_size=(3, 3), padding='valid', activation='relu', name='conv1', input_shape=shape))
    # 添加BatchNormalize1
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 20*126*1249->20*63*312
    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 4), name='pool1'))
    model.add(Dropout(0.1, name='dropout1'))

    # 添加卷积层2
    # 20*63*312->41*61*310
    model.add(Conv2D(41, kernel_size=(3, 3), padding='valid', activation='relu', name='conv2'))
    # 添加BatchNormalize2
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 41*61*310->41*30*77
    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 4), name='pool2'))
    model.add(Dropout(0.1, name='dropout2'))

    # 添加卷积层3
    # 41*30*77->41*28*75
    model.add(Conv2D(41, kernel_size=(3, 3), padding='valid', activation='relu', name='conv3'))
    # 添加BatchNormalize1
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 41*28*75->41*14*18
    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 4), name='pool3'))
    model.add(Dropout(0.1, name='dropout3'))

    # 添加卷积层4
    # 41*14*18->62*12*16
    model.add(Conv2D(62, kernel_size=(3, 3), padding='valid', activation='relu', name='conv4'))
    # 添加BatchNormalize1
    model.add(BatchNormalization())
    # 添加最大池化, 默认valid, 丢弃边缘信息
    # 62*12*16->62*3*4
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4'))
    model.add(Dropout(0.1, name='dropout4'))

    # 添加Flatten层
    # 62*3*4->744
    model.add(Flatten())

    # 添加线性层
    # 744->256
    model.add(Dense(256, name='Linear1'))
    # 256->50
    model.add(Dense(class_num, name='Linear2'))

    model.add(Activation('softmax'))
    # initiate Adam optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-8, amsgrad=True)

    # Let's train the model using Adam with amsgrad
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
























