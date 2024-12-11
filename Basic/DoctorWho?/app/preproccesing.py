import gdown
import os
import zipfile
from config import DATA_PATH
import time


def download_data():
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l8/diseases.zip', None, quiet=True)


def cleanup_invalid_files(folder_path):
    """
    Удаляет файлы, которые не должны быть в конечной папке (например, временные файлы).
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith('._') or file.endswith('.DS_Store'):
                os.remove(os.path.join(root, file))
                print(f"Удален файл: {file}")


def unzip_with_cleanup(zip_file, extract_path, from_enc="cp437", to_enc="cp866"):
    if not os.path.exists(zip_file):
        print(f'Файл {zip_file} не найден!')
        return []

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                try:
                    unicode_name = file_info.filename.encode(from_enc).decode(to_enc)
                except UnicodeDecodeError:
                    unicode_name = file_info.filename

                target_path = os.path.join(extract_path, unicode_name)

                if file_info.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(file_info.filename) as source, open(target_path, 'wb') as target:
                        target.write(source.read())

        print(f'Архив {zip_file} успешно распакован в {extract_path}')
    except (RuntimeError, UnicodeDecodeError) as e:
        print(f"Ошибка при распаковке: {e}")

    cleanup_invalid_files(extract_path)

    # Получаем список классов (файлы без расширения из папки `dis`)
    dis_path = os.path.join(extract_path, "dis")
    if not os.path.exists(dis_path):
        print(f"Папка {dis_path} не найдена после распаковки!")
        return []

    class_list = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(dis_path) if os.path.isfile(os.path.join(dis_path, f))]
    )
    print(f'Количество классов: {len(class_list)}, метки классов: {class_list}')
    return class_list
 



# def create_train_test_texts():

#     CLASS_LIST = []  # Список классов
#     text_train = []  # Список для оучающей выборки
#     text_test = []   # Список для тестовой выборки

#     # Зададим коэффициент разделения текста на обучающую и текстовую выборки
#     split_coef = 0.8

#     # Получим списки файлов в папке
#     file_list = os.listdir(DATA_PATH)
    
#     pos = [os.path.splitext(file)[0] for file in os.listdir(DATA_PATH) if file.endswith('.txt')]

#     print(pos)


#     for file_name in file_list:
#         m = file_name.split('.') # Разделим имя файла и расширение
#         class_name = m[0]        # Из имени файла получим название класса
#         ext = m[1]               # Выделим расширение файла

#         if ext=='txt':                                         # Если расширение txt то берем файл в работу
#             if class_name not in CLASS_LIST:                   # Проверим, есть уже такой класс в списке
#                 print(f'Добавление класса "{class_name}"')     # Выведем имя нового класса
#                 CLASS_LIST.append(class_name)                  # Добавим новый класс в списоккласса "{class_name}"')

#             cls = CLASS_LIST.index(class_name)                                        # Получим индекс (номер) нового класса
#             print(f'Добавление файла "{file_name}" в класс "{CLASS_LIST[cls]}"')      # Сообщим о появлении нового класса

#             with open(f'{DATA_PATH}/{file_name}', 'r') as f: # Откроем файл на чтение
#                 text = f.read()                                                       # Загрузка содержимого файла в строку
#                 text = text.replace('\n', ' ').split(' ')                             # Уберем символы перевода строк, получим список слов
#                 text_len=len(text)                                                    # Найдем количество прочитанных слов
#                 text_train.append(' '.join(text[:int(text_len*split_coef)]))          # Выделим часть файла в обучающую выборку
#                 text_test.append(' '.join(text[int(text_len*split_coef):]))           # Выделим часть файла в тестовую выборку

#     CLASS_COUNT = len(CLASS_LIST)

#     def test_texts_for_dis():
#             # Проверим загрузки: выведем начальные отрывки из каждого класса

#         for cls in range(CLASS_COUNT):             # Запустим цикл по числу классов
#             print(f'Класс: {CLASS_LIST[cls]}')     # Выведем имя класса
#             print(f'  train: {text_train[cls]}')   # Выведем фрагмент обучающей выборки
#             print(f'  test : {text_test[cls]}')    # Выведем фрагмент тестовой выборки
#             print()
    
#     test_texts_for_dis()

#     return text_train, text_test, CLASS_COUNT
        

def create_train_test_texts():
    CLASS_LIST = []  # Список классов
    text_train = []  # Список для обучающей выборки
    text_test = []   # Список для тестовой выборки

    # Зададим коэффициент разделения текста на обучающую и тестовую выборки
    split_coef = 0.8

    # Получим списки файлов в папке
    file_list = os.listdir(DATA_PATH)

    for file_name in file_list:
        m = file_name.split('.')  # Разделим имя файла и расширение
        if len(m) < 2:            # Защита от файлов без расширения
            continue

        class_name, ext = m[0], m[1]  # Из имени файла получим название класса и расширение

        if ext == 'txt':                                        # Если расширение txt, берем файл в работу
            if class_name not in CLASS_LIST:                   # Проверим, есть уже такой класс в списке
                print(f'Добавление класса "{class_name}"')     # Выведем имя нового класса
                CLASS_LIST.append(class_name)                  # Добавим новый класс в список

            cls = CLASS_LIST.index(class_name)                                        # Получим индекс (номер) нового класса
            print(f'Добавление файла "{file_name}" в класс "{CLASS_LIST[cls]}"')      # Сообщим о появлении нового класса

            with open(f'{DATA_PATH}/{file_name}', 'r', encoding='utf-8') as f:        # Откроем файл на чтение
                text = f.read()                                                      # Загрузка содержимого файла в строку
                text = text.replace('\n', ' ').split(' ')                             # Уберем символы перевода строк, получим список слов
                text_len = len(text)                                                 # Найдем количество прочитанных слов
                text_train.append(' '.join(text[:int(text_len * split_coef)]))        # Обучающая выборка
                text_test.append(' '.join(text[int(text_len * split_coef):]))         # Тестовая выборка

    # Проверим загрузку: выведем начальные отрывки из каждого класса
    for cls in range(len(CLASS_LIST)):             # Запустим цикл по числу классов
        print(f'Класс: {CLASS_LIST[cls]}')         # Выведем имя класса
        print(f'  train: {text_train[cls][:100]}') # Выведем фрагмент обучающей выборки
        print(f'  test : {text_test[cls][:100]}')  # Выведем фрагмент тестовой выборки
        print()

    return text_train, text_test, CLASS_LIST
