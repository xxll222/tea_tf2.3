#  _*_coding:utf-8_*_
# Author          : liuwenhui
# Creation time   : 2024/3/25
# Document        : MobieNetV3small.py
# IDE             : PyCharm
# 导入所需库
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os
from time import time


# 数据集加载函数
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 数据增强设置
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 加载并增强训练数据
    train_ds = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # 加载测试数据
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # 获取类别名称
    class_names = train_ds.class_indices.keys()

    return train_ds, val_ds, list(class_names)


# 构建 MobileNetV3 Small 模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=None):
    # 加载预训练的 MobileNetV3 Small 模型
    base_model = MobileNetV3Small(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    # 冻结基础模型的层
    base_model.trainable = False

    # 构建新的模型，在 MobileNetV3 Small 之上添加全局平均池化层和分类层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(class_num, activation='softmax')(x)

    # 构建最终模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 输出模型概述
    model.summary()

    # 编译模型
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 训练模型
def train(epochs, data_dir, test_data_dir, img_height, img_width, batch_size):
    train_ds, val_ds, class_names = data_load(data_dir, test_data_dir, img_height, img_width, batch_size)
    num_classes = len(class_names)

    model = model_load(IMG_SHAPE=(img_height, img_width, 3), class_num=num_classes)

    # TensorBoard 回调
   # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 训练模型
    #  history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback])
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    # 保存模型
    model.save("models/mobilenetv3small_model.h5")

    return history


if __name__ == "__main__":
    EPOCHS = 10
    DATA_DIR = 'path/to/train_data'
    TEST_DATA_DIR = 'path/to/test_data'
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32

    history = train(EPOCHS, DATA_DIR, TEST_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    # 可视化训练结果（此部分可根据需要添加）
    # show_loss_acc(history)
