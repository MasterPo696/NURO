import gdown
import os
import zipfile


# The PATHs for the folders! 
DATA_PATH   = "data/"
TRAIN_PATH  = DATA_PATH + 'cars'       
TEST_PATH   = DATA_PATH + 'cars_test'  

# Загрузка zip-архива с датасетом из облака на диск виртуальной машины colab
def download_dataset():
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l5/middle_fmr.zip', None, quiet=True)

# Пути к файлам
zip_file = "middle_fmr.zip"
extract_path = TRAIN_PATH  # Замените на путь для распаковки

# The splits for the func
TEST_SPLIT          = 0.1                   # Доля тестовых данных в общем наборе
VAL_SPLIT           = 0.2                   # Доля проверочной выборки в обучающем наборе


def unzip_to_folder(zip_file):
    # Проверяем, существует ли zip-файл
    if os.path.exists(zip_file):
        # Открываем и разархивируем
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f'Архив {zip_file} успешно распакован в {extract_path}')
    else:
        print(f'Файл {zip_file} не найден!')
    
    CLASS_LIST = sorted(os.listdir(TRAIN_PATH))
    CLASS_COUNT = len(CLASS_LIST)
    print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')
    return CLASS_LIST


def make_test_folder(CLASS_LIST):
    try:
        os.mkdir(TEST_PATH)                                        # Создание папки для тестовых данных
    except:
        pass
    train_count = 0
    test_count = 0

    for class_name in CLASS_LIST:                              # Для всех классов по порядку номеров (их меток)
        class_path = f'{TRAIN_PATH}/{class_name}'              # Формирование полного пути к папке с изображениями класса
        test_path = f'{TEST_PATH}/{class_name}'                # Полный путь для тестовых данных класса
        class_files = os.listdir(class_path)                   # Получение списка имен файлов с изображениями текущего класса
        class_file_count = len(class_files)                    # Получение общего числа файлов класса

        try:
            os.mkdir(test_path)                                    # Создание подпапки класса для тестовых данных
        except:
            pass

        test_file_count = int(class_file_count * TEST_SPLIT)   # Определение числа тестовых файлов для класса
        test_files = class_files[-test_file_count:]            # Выделение файлов для теста от конца списка
        for f in test_files:                                   # Перемещение тестовых файлов в папку для теста
            os.rename(f'{class_path}/{f}', f'{test_path}/{f}')
        train_count += class_file_count                        # Увеличение общего счетчика файлов обучающего набора
        test_count += test_file_count                          # Увеличение общего счетчика файлов тестового набора

        print(f'Размер класса {class_name}: {class_file_count} машин, для теста выделено файлов: {test_file_count}')

    print(f'Общий размер базы: {train_count}, выделено для обучения: {train_count - test_count}, для теста: {test_count}')
    return test_count
        