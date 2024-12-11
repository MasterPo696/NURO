import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown
import os
import zipfile
import timex
from config import VOCAB_SIZE, WIN_SIZE, WIN_HOP

from app.preproccesing import download_data, unzip_with_cleanup, create_train_test_texts
from app.tokenization import make_tokenizer, make_train_test, print_text_stats, vectorize_sequence , create_x_y_train_test
from app.net import compile_train_eval_model



# Пути к файлам
DATA_PATH = "/Users/masterpo/Desktop/NURO/Basic/DoctorWho?/data/"
zip_file = "diseases.zip"
# extract_path = os.path.join(DATA_PATH)

extract_path  = DATA_PATH
import timex
print(dir(timex))  # Это покажет, какие объекты есть в модуле timex
from tensorflow.keras.layers import GRU, Bidirectional, LSTM


def main():
    download_data()
    class_list = unzip_with_cleanup(zip_file, extract_path)
    # print(class_list)
    text_train, text_test, class_list = create_train_test_texts()
    print("Text train:", text_train[0][100:])
    print("Text test:", text_test[0][100:])
    tokenizer = make_tokenizer(VOCAB_SIZE, text_train)
    seq_train, seq_test = make_train_test(tokenizer,       # Предобученный токенайзер
                    text_train,      # Обучающая выборка
                    text_test = None # Тестовая выборка
                    )

    x_train, y_train, x_test, y_test = create_x_y_train_test(text_train, text_test, class_list)
    

    # model_Conv_1 = make_mod(VOCAB_SIZE, WIN_SIZE, len(class_list))
    model4 = Sequential()
    model4.add(Embedding(VOCAB_SIZE, 50, input_length=WIN_SIZE))
    model4.add(SpatialDropout1D(0.4))
    model4.add(BatchNormalization())
    # Два двунаправленных рекуррентных слоя LSTM
    model4.add(Bidirectional(LSTM(8, return_sequences=True)))
    model4.add(Bidirectional(LSTM(8, return_sequences=True)))
    model4.add(Dropout(0.3))
    model4.add(BatchNormalization())
    # Два рекуррентных слоя GRU
    model4.add(GRU(16, return_sequences=True, reset_after=True))
    model4.add(GRU(16, reset_after=True))
    model4.add(Dropout(0.3))
    model4.add(BatchNormalization())
    # Дополнительный полносвязный слой
    model4.add(Dense(200, activation='relu'))
    model4.add(Dropout(0.3))
    model4.add(BatchNormalization())
    model4.add(Dense(len(class_list), activation='softmax'))

        # Получим обученую модель, оценку ее работы
    compile_train_eval_model(model4,
                            x_train, y_train,
                            x_test, y_test,
                            optimizer='adam',
                            epochs=30,
                            batch_size=100,
                            class_labels=class_list,
                            title='Язвы и прочие....')
    
    return

main()


