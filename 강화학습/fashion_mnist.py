import tensorflow as tf
from tensorflow import keras

from numpy import argmax, max, expand_dims
from matplotlib.pyplot import figure, bar, imshow, yticks, colorbar, grid, xticks, show, xlabel, cm, subplot, ylim

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

figure()
imshow(train_images[0])
colorbar()
grid(False)
show()

train_images = train_images / 255.0

test_images = test_images / 255.0

figure(figsize=(10, 10))
for i in range(25):
    subplot(5, 5, i + 1)
    xticks([])
    yticks([])
    grid(False)
    imshow(train_images[i], cmap=cm.binary)
    xlabel(class_names[train_labels[i]])
show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

predictions = model.predict(test_images)

print(predictions[0])

argmax(predictions[0])

print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    grid(False)
    xticks([])
    yticks([])

    imshow(img, cmap=cm.binary)

    predicted_label = argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                     100 * max(predictions_array),
                                     class_names[true_label]),
           color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    grid(False)
    xticks([])
    yticks([])
    thisplot = bar(range(10), predictions_array, color="#777777")
    ylim([0, 1])
    predicted_label = argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 2
figure(figsize=(6, 3))
subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
show()

i = 8
figure(figsize=(6, 3))
subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
show()

# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
show()

# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[0]

print(img.shape)

# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (expand_dims(img, 0))

print(img.shape)

# 이제 이 이미지의 예측을 만듭니다
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = xticks(range(10), class_names, rotation=45)

# model.predict는 2차원 넘파이 배열을 반환하므로 첫 번째 이미지의 예측을 선택합니다.
print(argmax(predictions_single[0]))

# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
