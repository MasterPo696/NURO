from app.preproccessing import download_dataset, unzip_to_folder, make_test_folder
from app.augmentation import make_train_test_datagen, make_train_val_test_generator, show_batches
from app.net import create_model

# PATH
DATA_PATH   = "data/"
TRAIN_PATH  = DATA_PATH + 'cars'       
TEST_PATH   = DATA_PATH + 'cars_test'  

# Пути к файлам
zip_file = "middle_fmr.zip"
extract_path = TRAIN_PATH  # Замените на путь для распаковки

def main():
    download_dataset()
    class_list = unzip_to_folder(zip_file)
    test_count = make_test_folder(class_list)
    train_datagen, test_datagen = make_train_test_datagen()
    train_gen, val_gen, test_get = make_train_val_test_generator(train_datagen, test_datagen, test_count)
    show_batches(train_gen, class_list)
    create_model(train_gen, val_gen, test_get, len(class_list), class_list)


main()