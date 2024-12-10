
from tensorflow.keras import utils      # Функции-утилиты для работы с категориальными данными
from tensorflow.keras.models import Sequential # Класс для конструирования последовательной модели нейронной сети
from tensorflow.keras.preprocessing.text import Tokenizer # Токенизатор для преобразование текстов в последовательности
from tensorflow.keras.preprocessing.sequence import pad_sequences # Заполнение последовательностей до определенной длины
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation # Основные слои
from tensorflow.keras.callbacks import EarlyStopping
from config import MAXWORDSCOUNT
import matplotlib.pyplot as plt

def create_model():
    modelBoW = Sequential()                                    # Создаём полносвязную сеть для обучения на Bag of Words
    modelBoW.add(BatchNormalization(input_dim=MAXWORDSCOUNT))  # Слой пакетной нормализации
    modelBoW.add(Dense(32, activation="relu"))                  # Полносвязный слой
    modelBoW.add(Dropout(0.4))                                 # Слой регуляризации Dropout
    modelBoW.add(Dense(1, activation='sigmoid'))               # Выходной полносвязный слой

    modelBoW.compile(optimizer='adam',                         # Компиляция модели
                loss='binary_crossentropy',
                metrics=['accuracy'])
    modelBoW.summary()                                         # Обобщение информации по модели
    
    return modelBoW


def model_fit(modelBoW, x_train_BoW, y_train, x_test_BoW, y_test):
    history = modelBoW.fit(x_train_BoW,
                        y_train,
                        epochs=30,
                        batch_size=128,
                        validation_data=(x_test_BoW, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3)])  # Добавляем Early Stopping

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе', marker='o')
    plt.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе', marker='x')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.title('График точности обучения по эпохам')
    plt.legend()
    plt.grid(True)
    plt.show()
