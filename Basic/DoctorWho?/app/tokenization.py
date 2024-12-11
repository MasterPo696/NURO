# Функция для создания и обучения токенайзера.

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
from config import VOCAB_SIZE, WIN_SIZE, WIN_HOP

def make_tokenizer(VOCAB_SIZE, #Размер словаря
                   text_train  #Обучающая выборка
                   ):

    tokenizer = Tokenizer(num_words=VOCAB_SIZE,                                           # Создаем токенайзер с размером словаря VOCAB_SIZE.
                          filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',  # Знаки препинания, скобки, спец символы убираем
                          lower=True,                                                     # Все символы переводим в прописные
                          split=' ',                                                      # Слова разделяются пробелом
                          oov_token='неизвестное_слово',                                  # Слова не вошедшие в словарь помечаем как 'неизвестное_слово'
                          char_level=False)                                               # Запрещаем токенизировать каждый символ

    # Обучим токенайзер
    tokenizer.fit_on_texts(text_train)

    # Вернем обученный токенайзер
    return tokenizer

# Функция для перевода текста в токены (индексы)

def make_train_test(tokenizer,       # Предобученный токенайзер
                    text_train,      # Обучающая выборка
                    text_test = None # Тестовая выборка
                    ):
    
    # Переведем обучающую выборку из слов в токены
    seq_train = tokenizer.texts_to_sequences(text_train)


    if text_test:
        # Переведем тестовую выборку в токены
        seq_test = tokenizer.texts_to_sequences(text_test)
    else:
       # Тестовой выборки нет - отдадим None
        seq_test = None

    # Вернем  обучающий и тестовый датасеты в виде токенов

    return seq_train, seq_test


# Функция вывода статистики по текстам


def print_text_stats(title,                  # Заголовок для блока статистики
                     texts,                  # Тексты в виде слов
                     sequences,              # Тексты в виде индексов
                     class_labels # Список классов
                     ):

    chars = 0
    words = 0

    # Выведем заголовок
    print(f'Статистика по {title} текстам:')

    # Выведем итоги по всем классам данного набора текстов и их последовательностей индексов
    for cls in range(len(class_labels)):
        print('{:<15} {:9} символов,{:8} слов'.format(class_labels[cls],      # Выводим количество имя класса, символом, слов
                                                      len(texts[cls]),
                                                      len(sequences[cls])))
        chars += len(texts[cls])                                              # Готовим сумму всех символов
        words += len(sequences[cls])                                          # Готовим сумму всех слов

    print('----')
    print('{:<15} {:9} символов,{:8} слов\n'.format('В сумме', chars, words)) # Выведем статистику по всему датасету

# Функция разбиения последовательности на отрезки скользящим окном
# На входе - последовательность индексов, размер окна, шаг окна

def split_sequence(sequence,  # Тексты в виде индексов
                   win_size,  # Размер окна
                   hop        # Шаг смещения окна
                   ):

    # Последовательность разбивается на части до последнего полного окна
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]


# Функция формирования выборок из последовательностей индексов

def vectorize_sequence(seq_list, #  Список категорий текстов
                       win_size, # Размер окна
                       hop       # Шаг смещения окна
                       ):

    print(seq_list)
    # Найдем количество классов
    class_count = len(seq_list)

    # Списки для исходных векторов и категориальных меток класса
    x, y = [], []

    # Для каждого класса:
    for cls in range(class_count):

        # Разобъем последовательности класса cls на отрезки
        vectors = split_sequence(seq_list[cls], win_size, hop)

        # Добавим отрезки в выборку
        x += vectors

        # Для всех отрезков класса cls добавим меток класса в виде OHE
        y += [utils.to_categorical(cls, class_count)] * len(vectors)

    # Вернем как numpy-массивов
    
    return np.array(x), np.array(y)


import time


class timex:
    def __enter__(self):
        # Фиксация времени старта процесса
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        # Вывод времени работы
        print('Время обработки: {:.2f} с'.format(time.time() - self.t))



def create_x_y_train_test(text_train, text_test, class_list):



    # with timex():

    tok = make_tokenizer(VOCAB_SIZE, text_train) # Получим обученный токенайзер

    # Переверем выборки из слов в токены
    seq_train, seq_test = make_train_test(tok, text_train, text_test)

        # Формируем обучающую выборку
    x_train, y_train = vectorize_sequence(seq_train, WIN_SIZE, WIN_HOP)
        # Формируем тестовую выборку
    x_test, y_test = vectorize_sequence(seq_test, WIN_SIZE, WIN_HOP)

    # Выведем итоги формирования выборок

    print_text_stats('обучающим', x_train, seq_train, class_list)
    print_text_stats('тестовым', x_test, seq_test, class_list)
    print()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)


    return x_train, y_train, x_test, y_test 

