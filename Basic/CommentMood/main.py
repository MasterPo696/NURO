import numpy as np                      # Работа с массивами данных
import pandas as pd                     # Работа с таблицами
import matplotlib.pyplot as plt         # Отрисовка графиков
import gdown                            # Загрузка датасетов из облака google
import os                               # Для работы с файлами в Colaboratory
import zipfile
from itertools import chain

from app.preproccesing import get_data, unzip_and_rename, read_text, one_list_text, get_texts_value, slice_text_for_parts
from app.net import create_model, model_fit
from app.tokenization import create_tokenizer, create_sequences, split_sequence, get_samples, create_x_y_train_test, create_bag_of_words

# Объявляем интересующие нас классы
# class_names = ["Негативный отзыв", "Позитивный отзыв"]
# # Считаем количество классов
# num_classes = len(class_names)

# Объявляем функции для чтения файла. На вход отправляем путь к файлу

zip_file = "tesla.zip"
extract_path = "data/"


def main():
    # get_data()
    # unzip_and_rename(zip_file)
    texts_list = one_list_text()
    train_len_share_list = get_texts_value(texts_list)
    train_data, test_data = slice_text_for_parts(texts_list)
    tokenizer = create_tokenizer(train_data, test_data)
    train_sequence, test_sequence = create_sequences(tokenizer, train_data, test_data)
    x_train, y_train, x_test, y_test = create_x_y_train_test(train_sequence, test_sequence)
    x_train_BoW, x_test_BoW = create_bag_of_words(tokenizer, x_train, x_test)
    modelBoW = create_model()
    model_fit(modelBoW, x_train_BoW, y_train, x_test_BoW, y_test)

main()

