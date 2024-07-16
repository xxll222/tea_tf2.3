#  _*_coding:utf-8_*_
# Author          : liuwenhui
# Creation time   : 2024/3/17
# Document        : app.py
# IDE             : PyCharm
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

# 初始化Flask应用
app = Flask(__name__)

# 加载模型
model_mobilenet = tf.keras.models.load_model('models/mobilenet_tea.h5')

# 定义预处理图片的函数
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # 从请求中获取图片
    image_file = request.files['file']

    # 预处理图片
    image = Image.open(BytesIO(image_file.read()))
    processed_image = preprocess_image(image, target_size=(224, 224))

    # 进行预测
    prediction = model_mobilenet.predict(processed_image).tolist()

    # 获取类别名称
    class_names = ['炭疽病','藻斑病', '鸟眼斑', '褐斑病', '茶轮斑病', '健康', '红叶斑', '白星病']

    # 找到最高概率的类别
    predicted_class = class_names[np.argmax(prediction)]

    # 以JSON格式返回结果
    response = {
        'prediction': {
            'class': predicted_class,
            'confidence': np.max(prediction)
        }
    }
    return jsonify(response)


if __name__ == "__main__":
    # 运行Flask应用
    app.run(debug=True)

