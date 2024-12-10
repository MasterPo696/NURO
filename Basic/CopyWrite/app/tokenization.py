from config import MAXWORDSCOUNT
from tensorflow.keras.preprocessing.text import Tokenizer # Токенизатор для преобразование текстов в последовательности
from tensorflow.keras import utils
import numpy as np
from config import SAMPLE_LEN


from itertools import chain


def create_train_test_data(texts_list):
    # Создадим пустые списки для обучающей и проверочной выборок
    train_data = []
    test_data = []

    # Циклом пройдёмся по 20 текстам
    for i in range(len(texts_list)):

        # Выделим 80% каждого текста на обучающую и 20% на проверочную
        train_len = int(len(texts_list[i])*0.8)

        # Добавим тексты в выборки функцией chain()
        train_data = list(chain(train_data, ([texts_list[i][:train_len]])))
        test_data = list(chain(test_data, ([texts_list[i][train_len:]])))

    print(f'Количество элементов в train_data: {len(train_data)}')
    print(f'Выборка train_data принадлежит к классу данных: {type(train_data)}')
    print(f'Тип данных первого элемента в train_data: {type(train_data[0])}')
    print(f'Общая длина первого текста(в токенах): {len(texts_list[0])}')
    print(f'80% от длины первого текста(в токенах), оставшихся в обучающей выборке: {len(train_data[0])}')
    print(f'Отрывок из первого текста обучающей выборки: {train_data[0][:26]}')

    print(f'Количество элементов в test_data: {len(test_data)}')
    print(f'Выборка test_data принадлежит к классу данных: {type(test_data)}')
    print(f'Тип данных первого элемента в test_data: {type(test_data[0])}')
    print(f'Общая длина первого текста(в токенах): {len(texts_list[0])}')
    print(f'20% от длины первого текста(в токенах), отделившихся в проверочную выборку: {len(test_data[0])}')
    print(f'Отрывок из первого текста: {test_data[0][1:28]}')

    return train_data, test_data


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

    print(f'Кол-во индексов первого текста обучающей выборки: {len(train_sequence[0])}')
    print(f'Кол-во индексов первого текста тестовой выборки:  {len(test_sequence[0])}')
    print(f'Первые 15 индексов первого текста обучающей выборки: {train_sequence[0][:15]}')
    print(f'Первые 15 индексов первого текста тестовой выборки: {test_sequence[0][:15]}')

    return train_sequence, test_sequence

def split_sequence(sequence,   # Последовательность индексов
                   win_size,   # Размер окна для деления на примеры
                   hop):       # Шаг окна

    # Последовательность разбивается на части до последнего полного окна
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]

def vectorize_sequence(seq_list,    # Список последовательностей индексов
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
        y += [utils.to_categorical(cls, class_count)] * len(vectors)

    # Возврат результатов как numpy-массивов
    return np.array(x), np.array(y)

def create_x_y_train_test(train_sequence, test_sequence):
    sample_len = SAMPLE_LEN

    step = 500

    x_train, y_train = vectorize_sequence(train_sequence, sample_len, step)
    x_test, y_test = vectorize_sequence(test_sequence, sample_len, step)

    # Выведем размерности всех выборок
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, y_train, x_test, y_test


def create_bag_of_words(tokenizer, x_train, x_test):
    x_train_BoW = tokenizer.sequences_to_matrix(x_train.tolist())
    x_test_BoW = tokenizer.sequences_to_matrix(x_test.tolist())

    return x_train_BoW, x_test_BoW