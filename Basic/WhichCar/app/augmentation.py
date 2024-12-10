# Генератор аугментированных изображений
import keras
from keras import layers, models, optimizers, preprocessing, utils, datasets

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)

# from tensorflow.keras.preprocessing
import matplotlib.pyplot as plt
image = preprocessing.image 

# ImageDataGenerator = preprocessing.image.ImageDataGenerator
Adam = optimizers.Adam                              # Подключение оптимизатора Adam

# Params for the Augmentation
ROTATION_RANGE      = 8                     # Пределы поворота
WIDTH_SHIFT_RANGE   = 0.15                  # Пределы сдвига по горизонтали
HEIGHT_SHIFT_RANGE  = 0.15                  # Пределы сдвига по вертикали
ZOOM_RANGE          = 0.15                  # Пределы увеличения/уменьшения
BRIGHTNESS_RANGE    = (0.7, 1.3)            # Пределы изменения яркости
HORIZONTAL_FLIP     = True                  # Горизонтальное отражение разрешено

# Генераторы изображений

# The splits for the func
TEST_SPLIT          = 0.1                   # Доля тестовых данных в общем наборе
VAL_SPLIT           = 0.2                   # Доля проверочной выборки в обучающем наборе

# Params for the NN
EPOCHS              = 60                    # Число эпох обучения
BATCH_SIZE          = 24                    # Размер батча для обучения модели
OPTIMIZER           = Adam(0.0001)          # Оптимизатор

# PATH
DATA_PATH   = "data/"
TRAIN_PATH  = DATA_PATH + 'cars'       
TEST_PATH   = DATA_PATH + 'cars_test'  

# Sizes
IMG_WIDTH           = 128                   # Ширина изображения для нейросети
IMG_HEIGHT          = 64                    # Высота изображения для нейросети
IMG_CHANNELS        = 3                     # Количество каналов (для RGB равно 3, для Grey равно 1)





# Изображения для обучающего набора нормализуются и аугментируются согласно заданным гиперпараметрам
# Далее набор будет разделен на обучающую и проверочную выборку в соотношении VAL_SPLIT

def make_train_test_datagen():
    train_datagen = ImageDataGenerator(
                            rescale=1. / 255.,
                            rotation_range=ROTATION_RANGE,
                            width_shift_range=WIDTH_SHIFT_RANGE,
                            height_shift_range=HEIGHT_SHIFT_RANGE,
                            zoom_range=ZOOM_RANGE,
                            brightness_range=BRIGHTNESS_RANGE,
                            horizontal_flip=HORIZONTAL_FLIP,
                            validation_split=VAL_SPLIT
                        )
    # Изображения для тестового набора только нормализуются
    test_datagen = ImageDataGenerator(
                        rescale=1. / 255.
                        )
    return train_datagen, test_datagen



# Выборки генерируются

def make_train_val_test_generator(train_datagen, test_datagen, test_count):
    # Обучающая выборка генерируется из папки обучающего набора
    train_generator = train_datagen.flow_from_directory(
        # Путь к обучающим изображениям
        TRAIN_PATH,
        # Параметры требуемого размера изображения
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        # Размер батча
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        # Указание сгенерировать обучающую выборку
        subset='training'
    )

    # Проверочная выборка также генерируется из папки обучающего набора
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        # Указание сгенерировать проверочную выборку
        subset='validation'
    )

    # Тестовая выборка генерируется из папки тестового набора
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=test_count,
        class_mode='categorical',
        shuffle=True,
    )

    print(f'Формы данных тренировочной выборки: {train_generator[0][0].shape}, {train_generator[0][1].shape}, батчей: {len(train_generator)}')
    print(f'Формы данных   проверочной выборки: {validation_generator[0][0].shape}, {validation_generator[0][1].shape}, батчей: {len(validation_generator)}')
    print(f'Формы данных      тестовой выборки: {test_generator[0][0].shape}, {test_generator[0][1].shape}, батчей: {len(test_generator)}')

    print()

    # Проверка назначения меток классов
    print(f'Метки классов тренировочной выборки: {train_generator.class_indices}')
    print(f'Метки классов   проверочной выборки: {validation_generator.class_indices}')
    print(f'Метки классов      тестовой выборки: {test_generator.class_indices}')

    plt.imshow(train_generator[1][0][2])
    plt.show()

    return train_generator, validation_generator, test_generator


import numpy as np
# Функция рисования образцов изображений из заданной выборки
def show_batches(batch_name, CLASS_LIST):
    def show_batch(batch, CLASS_LIST,             # батч с примерами
                img_range=range(20),  # диапазон номеров картинок
                figsize=(25, 8),      # размер полотна для рисования одной строки таблицы
                columns=5             # число колонок в таблице
                ):

        for i in img_range:
            ix = i % columns
            if ix == 0:
                fig, ax = plt.subplots(1, columns, figsize=figsize)
            class_label = np.argmax(batch[1][i])
            ax[ix].set_title(CLASS_LIST[class_label])
            ax[ix].imshow(batch[0][i])
            ax[ix].axis('off')
            plt.tight_layout()

        plt.show()
    
    batch = batch_name[0]
    show_batch(batch, CLASS_LIST)