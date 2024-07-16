#  _*_coding:utf-8_*_
# Author          : liuwenhui
# Creation time   : 2024/3/17
# Document        : MobilenetV2.py
# IDE             : PyCharm
import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建mobilenet模型
# 模型加载，指定图片处理的大小和是否进行迁移学习
# 构建并微调mobilenet模型
def model_load_and_finetune(IMG_SHAPE=(224, 224, 3), class_num=12, fine_tune_layers=5):
    # 加载预训练的mobilenet模型
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # 获取要微调的层索引
    fine_tune_at = len(base_model.layers) - fine_tune_layers

    # 冻结前面大部分层
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 解冻最后几层进行微调
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model = tf.keras.models.Sequential([
        # 进行归一化的处理
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()

    # 编译模型时可以适当降低学习率，防止一开始就对预训练权重造成过大扰动
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_mobilenet.png', dpi=100)


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()

    train_ds, val_ds, class_names = data_load(r"D:\Desktop\A\tea_tf2.3\tea_tf2.3-master\new_data\train",
                                              r"D:\Desktop\A\tea_tf2.3\tea_tf2.3-master\new_data\val", 224, 224, 16)

    print(class_names)
    # 加载模型
    model = model_load_and_finetune(class_num=len(class_names), fine_tune_layers=5)
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save("models/mobilenet_tea.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=40)