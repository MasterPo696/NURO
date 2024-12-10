from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAXWORDSCOUNT
import matplotlib.pyplot as plt

def create_model():
    modelBoW = Sequential()                                    # Создаём полносвязную сеть для обучения на Bag of Words
    modelBoW.add(BatchNormalization(input_dim=MAXWORDSCOUNT))  # Слой пакетной нормализации
    modelBoW.add(Dense(80,  activation="relu"))                # Полносвязный слой
    modelBoW.add(Dropout(0.6))                                 # Слой регуляризации Dropout
    modelBoW.add(Dense(20, activation="relu"))                 # Полносвязный слой
    modelBoW.add(Dense(20, activation='sigmoid'))              # Выходной полносвязный слой

    modelBoW.compile(optimizer='adam',                         # Компиляция модели
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    modelBoW.summary()                                         # Обобщение информации по модели

    return modelBoW

def model_fit(x_train_BoW, y_train, x_test_BoW, y_test):
    
    modelBoW = create_model()
    # Обучение сети Bag of Words
    history = modelBoW.fit(x_train_BoW,
                        y_train,
                        epochs=20,
                        batch_size=64,
                        validation_data=(x_test_BoW, y_test))

    # Отрисовка графика точностей обучения по эпохам
    plt.plot(history.history['accuracy'],
            label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
            label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()
    
    return history

def create_model_Emb(sample_len=1000):
    modelEmb = Sequential()                                               # Создаём последовательную модель нейросети
    modelEmb.add(Embedding(MAXWORDSCOUNT, 60, input_length=sample_len))   # Слой Embedding (c указанием размерности вектора и длины входных данных)
    modelEmb.add(BatchNormalization())                                    # Добавляем слой нормализации данных
    modelEmb.add(Dense(20))                                               # Полносвязный слой
    modelEmb.add(Dropout(0.7))                                            # Слой регуляризации Dropout
    modelEmb.add(Flatten())                                               # Выравнивающий слой
    modelEmb.add(Dense(20, activation='sigmoid'))                         # Полносвязный выходной слой на 6 нейронов

    modelEmb.compile(optimizer='rmsprop',                                 # Компиляция модели
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    modelEmb.summary()                                                    # Обобщение информации по модели
    
    return modelEmb

def model_fit_Emb(x_train, y_train, x_test, y_test):

    modelEmb = create_model_Emb()
    # Обучение сети Embedding + Dense
    history = modelEmb.fit(x_train, y_train,
                        epochs=20, batch_size=64,
                        validation_data=(x_test, y_test))

    # Отрисовка графика точностей обучения по эпохам
    plt.plot(history.history['accuracy'],
            label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
            label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    return history