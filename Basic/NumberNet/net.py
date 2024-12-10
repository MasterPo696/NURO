
from keras import layers, models, optimizers, preprocessing, utils, datasets

import pylab                                        # Модуль для построения графиков
import matplotlib.pyplot as plt                     # Отрисовка изображений
from PIL import Image                               # Отрисовка изображений

mnist = datasets.mnist                             # Библиотека с базой Mnist
Sequential = models.Sequential                      # Подлючение класса создания модели Sequential
Dense = layers.Dense                                # Подключение класса Dense (полносвязного слоя) 
Adam = optimizers.Adam                              # Подключение оптимизатора Adam
utils = utils                                       # Утилиты для to_categorical
image = preprocessing.image                         # Для отрисовки изображения
load_img = image.load_img                           # Метод для загрузки изображений
import numpy as np


# Способ №1
# Загрузка данных Mnist или же просто сразу работаем с даннымии
# (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()

# # Сохраняем тренировочные и тестовые данные в формате .npy
# np.save('NumberNet/data/x_train.npy', x_train_org)
# np.save('NumberNet/data/y_train.npy', y_train_org)
# np.save('NumberNet/data/x_test.npy', x_test_org)
# np.save('NumberNet/data/y_test.npy', y_test_org)
# print("Данные загружены в файлы.")

PATH = "data/"


# Закрузка файлов из папки
x_train_org = np.load(f'{PATH}x_train.npy')
y_train_org = np.load(f'{PATH}y_train.npy')
x_test_org = np.load(f'{PATH}x_test.npy')
y_test_org = np.load(f'{PATH}y_test.npy')
print("Данные загружены из файлов.")

# Проверка целостности формата случайным файлом 
n = 0
plt.imshow(Image.fromarray(x_train_org[n]).convert('RGBA'))
# plt.show() 

# Изменение формата входных картинок с 28х28 на 784х1, преобразуем в вектор с одним измериением
HEIGHT = 28
WIDTH = 28
INPUT_DIM = HEIGHT*WIDTH

x_train = x_train_org.reshape(60000, INPUT_DIM)
x_test = x_test_org.reshape(10000, INPUT_DIM)
print(f"Колличество, размерность.\nФормат данных x_train_org:{x_train_org.shape}\nФормат данных x_train: {x_train.shape}")

# Нормализация картиной в float от 0 до 1
x_train = (x_train.astype('float32')) / 255
x_test = (x_test.astype('float32') )/ 255
# print(x_train[0]) #Матрица рисует поле для НамПай

# Преобразование ответов в формат one_hot_encoding. Например [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], единица ставится на тот же индекс vector[5]
y_train = utils.to_categorical(y_train_org, 10)
y_test = utils.to_categorical(y_test_org, 10)
# print(y_train[0]) для числа 5  - [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

# Создание сети.
model = Sequential()                                                # Создание модели и присвоение класса
model.add(Dense(800, input_dim=INPUT_DIM, activation="relu"))       # Первый линейный слой нейронов, входяшая размерной, а также активатор.
model.add(Dense(400, activation="relu"))                            # Второй линейный слой нейронов и активатор. 
model.add(Dense(10, activation="softmax"))                          # Третий слой, кол-во нейронов совпадает с форматом one_hot_encoding, активатор.

# # Компиляция модель, используем эти параметры, а также вывод  структуры модели
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# print(model.summary())

# #fit - функция обучения нейронки
# #x_train, y_train - обучающая выборка, входные и выходные данные
# #batch_size - размер батча, количество примеров, которое обрабатывает нейронка перед одним изменением весов
# #epochs - количество эпох, когда нейронка обучается на всех примерах выборки
# #verbose - 0 - не визуализировать ход обучения, 1 - визуализировать
# model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)

# model.save_weights('model.weights.h5')          # Сохранение весов модели



import numpy as np
import matplotlib.pyplot as plt

# Загрузка весов модели
model.load_weights('model.weights.h5')
print("Веса модели загружены.")

# Параметры
image_folder = 'nums/2/'  # Путь к папке с изображениями
target_size = (28, 28)            # Размер изображений

count = 0
right_count = 0
for digit in range(10):
    image_path = f'{image_folder}{digit}.jpg'
    print(f"Проверяем изображение: {image_path}")

    # 1. Загрузка изображения
    img = image.load_img(image_path, target_size=target_size, color_mode='grayscale')

    # 2. Преобразование изображения в numpy-массив
    img_array = image.img_to_array(img)

    # 3. Инверсия цветов, нормализация, reshape
    img_array = 255 - img_array  # Инверсия цветов
    img_array /= 255.0           # Нормализация значений (0-1)
    img_array = img_array.reshape(1, 784)  # Приведение к форме (1, 784)

    # Отобразим обработанное изображение
    plt.imshow(img_array.reshape(28, 28), cmap='gray')  # Вернем в двумерный формат для отображения
    plt.title(f"Обработанное изображение: {digit}")
    plt.show()

    # 4. Распознавание цифры нейронной сетью
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    print(f"Ожидалось: {digit}, Распознанная цифра: {predicted_digit}\n")
    count +=1
    if digit == predicted_digit: right_count += 1
print(f"Из {count} раз, правильно ответило {right_count} раз")

