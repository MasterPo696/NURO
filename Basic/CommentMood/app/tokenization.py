from itertools import chain
from config import MAXWORDSCOUNT
from tensorflow.keras.preprocessing.text import Tokenizer # Токенизатор для преобразование текстов в последовательности
import numpy as np


def create_tokenizer(train_data, test_data):
    # Максимальное количество слов
    maxWordsCount = MAXWORDSCOUNT

    # Сохраним Токенайзер в одноименной переменной
    tokenizer = Tokenizer(num_words=maxWordsCount,                                        # Максимальное кол-во слов
                        filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',  # Фильтры исходного текста
                        lower=True, split=' ',                                          # Все буквы к нижнему регистру, разделение слов пробелом
                        oov_token='unknown',                                            # Один лейбл для всех незнакомых слов
                        char_level=False)                                               # Без выравнивания символов

    # Создание словаря частотности по каждой выборке
    tokenizer.fit_on_texts(train_data)
    tokenizer.fit_on_texts(test_data)

    return tokenizer


def create_sequences(tokenizer, train_data, test_data):
    train_sequence = tokenizer.texts_to_sequences(train_data)
    test_sequence = tokenizer.texts_to_sequences(test_data)

    return train_sequence, test_sequence

def split_sequence(sequence,   # Последовательность индексов
                   win_size,   # Размер окна для деления на примеры
                   hop):       # Шаг окна

    # Последовательность разбивается на части до последнего полного окна
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]

def get_samples(seq_list,    # Список последовательностей индексов
                       win_size,    # Размер окна для деления на примеры
                       hop):        # Шаг окна

    # В списке последовательности следуют в порядке их классов (их кол-во сповпадает с кол-вом классов)
    class_count = len(seq_list)

    # Списки для исходных векторов и категориальных меток класса
    x, y = [], []

    # Для каждого класса:
    for cls in range(class_count):

        # Разбиение последовательности класса cls на отрезки
        vectors = split_sequence(seq_list[cls], win_size, hop)

        # Добавление отрезков в выборку

        x += vectors

        # Для всех отрезков класса cls добавление меток класса в виде OHE
        y += [cls] * len(vectors)

    # Возврат результатов как numpy-массивов
    return np.array(x), np.array(y)



def create_x_y_train_test(train_sequence, test_sequence):
    # Длина каждого отрезка индексов
    sample_len = 100

    # Длина шага по исходному массиву индексов
    step = 10

    # Делим на выборки
    x_train, y_train = get_samples(train_sequence, sample_len, step)
    x_test, y_test = get_samples(test_sequence, sample_len, step)

        # Выведем размерности всех выборок
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, y_train, x_test, y_test

def create_bag_of_words(tokenizer, x_train, x_test):
        # Выборка для обучения
    x_train_BoW = tokenizer.sequences_to_matrix(x_train.tolist())

    # Выборка для проверки
    x_test_BoW = tokenizer.sequences_to_matrix(x_test.tolist())

    return x_train_BoW, x_test_BoW
