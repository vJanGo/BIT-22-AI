# sklearn官网教程：https://scikit-learn.org/stable/modules/classes.html
from sklearn.neural_network import MLPClassifier
import pickle
from PIL import Image
import numpy as np
# 从数据中加载图像
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 导入mnist.pkl数据
x_train, y_train, x_test, y_test = pickle.load(open("mnist.pkl", 'rb'))
# print(x_train[0])
# print(y_test[0])
# img_show(x_train[0])
# 转换数据格式，并只取6000条数据进行训练
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
x_train, y_train = x_train[:6000], y_train[:6000]

# 采用sklearn库中的MLPClassifier方法构建模型,
# 建议参数hidden_layer_sizes=(100,), activation='logistic',solver='adam', learning_rate_init=0.001, max_iter=1300, batch_size=800
# 实现后，请给出模型和模型的评分

# 构建MLP模型
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', 
                    learning_rate_init=0.001, max_iter=1300, batch_size=800, random_state=1120221586)

# 训练模型
mlp.fit(x_train, y_train)
print("训练完成！")
# 输出模型评分
score = mlp.score(x_test, y_test)
print("模型评分:", score)

