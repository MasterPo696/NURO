
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import os
# Матрица ошибок классификатора
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pylab     
from keras import layers, models, optimizers, preprocessing, utils, datasets
import time
                                   

Sequential = models.Sequential                      # Подлючение класса создания модели Sequential
Dense = layers.Dense                                # Подключение класса Dense (полносвязного слоя) 
Conv2D = layers.Conv2D
MaxPooling2D = layers.MaxPooling2D
Flatten = layers.Flatten
Dropout = layers.Dropout
BatchNormalization = layers.BatchNormalization
SpatialDropout2D = layers.SpatialDropout2D
Adam = optimizers.Adam                              # Подключение оптимизатора Adam
utils = utils                                       # Утилиты для to_categorical
image = preprocessing.image                         # Для отрисовки изображения
load_img = image.load_img                           # Метод для загрузки изображений
import numpy as np
from sklearn.model_selection import train_test_split

# Sizes and the dimensions
IMG_WIDTH           = 128                   # Ширина изображения для нейросети
IMG_HEIGHT          = 64                    # Высота изображения для нейросети
IMG_CHANNELS        = 3                     # Количество каналов (для RGB равно 3, для Grey равно 1)
# Params for the NN
EPOCHS              = 60                    # Число эпох обучения
BATCH_SIZE          = 24                    # Размер батча для обучения модели
OPTIMIZER           = Adam(0.0001)          # Оптимизатор




# Функция компиляции и обучения модели нейронной сети
# По окончанию выводит графики обучения

def compile_train_model(model,                  # модель нейронной сети
                        train_data,             # обучающие данные
                        val_data,               # проверочные данные
                        optimizer=OPTIMIZER,    # оптимизатор
                        epochs=EPOCHS,          # количество эпох обучения
                        batch_size=BATCH_SIZE,  # размер батча
                        figsize=(20, 5)):       # размер полотна для графиков

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Вывод сводки
    model.summary()

    # Обучение модели с заданными параметрами
    history = model.fit(train_data,
                        epochs=epochs,
                        # batch_size=batch_size,
                        validation_data=val_data)

    # Вывод графиков точности и ошибки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
               label='Доля верных ответов на обучающем наборе')
    ax1.plot(history.history['val_accuracy'],
               label='Доля верных ответов на проверочном наборе')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающем наборе')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочном наборе')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    plt.show()


# Функция вывода результатов оценки модели на заданных данных

def eval_model(model,
               x,                # данные для предсказания модели (вход)
               y_true,           # верные метки классов в формате OHE (выход)
               class_labels=[],  # список меток классов
               cm_round=3,       # число знаков после запятой для матрицы ошибок
               title='',         # название модели
               figsize=(10, 10)  # размер полотна для матрицы ошибок
               ):
    # Вычисление предсказания сети
    y_pred = model.predict(x)
    # Построение матрицы ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    # Округление значений матрицы ошибок
    cm = np.around(cm, cm_round)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Нейросеть {title}: матрица ошибок нормализованная', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    ax.images[-1].colorbar.remove()       # Стирание ненужной цветовой шкалы
    fig.autofmt_xdate(rotation=45)        # Наклон меток горизонтальной оси
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    plt.show()

    print('-'*100)
    print(f'Нейросеть: {title}')

    # Для каждого класса:
    for cls in range(len(class_labels)):
        # Определяется индекс класса с максимальным значением предсказания (уверенности)
        cls_pred = np.argmax(cm[cls])
        # Формируется сообщение о верности или неверности предсказания
        msg = 'ВЕРНО :-)' if cls_pred == cls else 'НЕВЕРНО :-('
        # Выводится текстовая информация о предсказанном классе и значении уверенности
        print('Класс: {:<20} {:3.0f}% сеть отнесла к классу {:<20} - {}'.format(class_labels[cls],
                                                                               100. * cm[cls, cls_pred],
                                                                               class_labels[cls_pred],
                                                                               msg))

    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))


# Совместная функция обучения и оценки модели нейронной сети

def compile_train_eval_model(model,                    # модель нейронной сети
                             train_data,               # обучающие данные
                             val_data,                 # проверочные данные
                             test_data,                # тестовые данные
                             class_labels,  # список меток классов
                             title='',                 # название модели
                             optimizer=OPTIMIZER,      # оптимизатор
                             epochs=EPOCHS,            # количество эпох обучения
                             batch_size=BATCH_SIZE,    # размер батча
                             graph_size=(20, 5),       # размер полотна для графиков обучения
                             cm_size=(10, 10)          # размер полотна для матрицы ошибок
                             ):

    # Компиляция и обучение модели на заданных параметрах
    # В качестве проверочных используются тестовые данные
    compile_train_model(model,
                        train_data,
                        val_data,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        figsize=graph_size)

    # Вывод результатов оценки работы модели на тестовых данных
    eval_model(model, test_data[0][0], test_data[0][1],
               class_labels=class_labels,
               title=title,
               figsize=cm_size)
    

def create_model(train_generator, validation_generator, test_generator, CLASS_COUNT, CLASS_LIST):
     # Создание последовательной модели
    model_conv = Sequential()

    # Первый сверточный слой
    model_conv.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model_conv.add(BatchNormalization())

    # Второй сверточный слой
    model_conv.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(3, 3)))

    # Третий сверточный слой
    model_conv.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model_conv.add(BatchNormalization())
    model_conv.add(Dropout(0.2))

    # Четвертый сверточный слой
    model_conv.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(3, 3)))
    model_conv.add(Dropout(0.2))

    # Пятый сверточный слой
    model_conv.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model_conv.add(BatchNormalization())

    # Шестой сверточный слой
    model_conv.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(3, 3)))
    model_conv.add(Dropout(0.2))

    # Слой преобразования многомерных данных в одномерные
    model_conv.add(Flatten())

    # Промежуточный полносвязный слой
    model_conv.add(Dense(2048, activation='relu'))

    # Промежуточный полносвязный слой
    model_conv.add(Dense(4096, activation='relu'))

    # Выходной полносвязный слой с количеством нейронов по количесту классов
    model_conv.add(Dense(CLASS_COUNT, activation='softmax'))


    # Обучение модели и вывод оценки ее работы на тестовых данных
    compile_train_eval_model(model_conv,
                            train_generator,
                            validation_generator,
                            test_generator,
                            class_labels=CLASS_LIST,
                            title='Сверточный классификатор')