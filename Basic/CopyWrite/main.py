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

from app.preproccesing import download_data, unzip_with_cleanup, get_dir_list, get_texts_80_value
from app.tokenization import create_train_test_data, create_tokenizer, create_sequences, vectorize_sequence, create_x_y_train_test, create_bag_of_words
from app.net import model_fit, model_fit_Emb


# The PATHs for the folders! 
DATA_PATH   = "data/writters/"
# TRAIN_PATH  = DATA_PATH + 'writters'       
# TEST_PATH   = DATA_PATH + 'cars_test'  
# Пути к файлам

zip_file = "20writers.zip"
extract_path = DATA_PATH  # Замените на путь для распаковки


def main():
    # download_data()
    # unzip_with_cleanup(zip_file, extract_path)
    texts_list = get_dir_list()
    train_len_shares = get_texts_80_value(texts_list)
    train_data, test_data = create_train_test_data(texts_list)
    tokenizer = create_tokenizer(train_data, test_data)
    train_sequence, test_sequence = create_sequences(tokenizer, train_data, test_data)
    x_train, y_train, x_test, y_test = create_x_y_train_test(train_sequence, test_sequence)
    x_train_BoW, x_test_BoW = create_bag_of_words(tokenizer, x_train, x_test)
    
    model_fit(x_train_BoW, y_train, x_test_BoW, y_test)
    model_fit_Emb(x_train, y_train, x_test, y_test)


main()