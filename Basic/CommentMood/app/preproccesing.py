import gdown
import os
import zipfile
from itertools import chain

# Пути к файлам
zip_file = "tesla.zip"
extract_path = "data/"
new_file_names = ["negative.txt", "positive.txt"]  # Новые имена для файлов

def get_data():
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l7/tesla.zip', None, quiet=True)


def unzip_and_rename(zip_file):
    # Проверяем, существует ли zip-файл
    if not os.path.exists(zip_file):
        print(f'Файл {zip_file} не найден!')
        return []

    # Открываем и разархивируем
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Получаем список файлов внутри архива
        file_list = zip_ref.namelist()
        print(f"Файлы в архиве: {file_list}")

        # Извлекаем и переименовываем
        for index, file_name in enumerate(file_list):
            # Определяем новое имя файла
            new_name = new_file_names[index] if index < len(new_file_names) else f"file_{index}.txt"

            # Извлекаем файл
            zip_ref.extract(file_name, extract_path)

            # Переименовываем извлечённый файл
            old_path = os.path.join(extract_path, file_name)
            new_path = os.path.join(extract_path, new_name)
            os.rename(old_path, new_path)
            print(f"Файл {file_name} переименован в {new_name}")

    # Итоговый список файлов
    final_file_list = sorted(os.listdir(extract_path))
    print(f"Итоговый список файлов: {final_file_list}")


PATH = "/Users/masterpo/Desktop/NURO/Basic/CommentMood/data/"

def read_text(file_name):
    read_file = open(file_name, 'r')
    text = read_file.read()
    text = text.replace("\n", " ")
    
    return text

def one_list_text():
    texts_list = []

    for j in os.listdir(PATH):
        texts_list.append(read_text(PATH + j))

        # Выводим на экран сообщение о добавлении файла
        print(j, 'добавлен в обучающую выборку')

    return texts_list



def get_texts_value(texts_list):
    # Узнаем объём каждого текста в словах и символах
    texts_len = [len(text) for text in texts_list]
    t_num = 0
    print(f'Размеры текстов по порядку (в токенах):')
    for text_len in texts_len:
        t_num += 1
        print(f'Текст №{t_num}: {text_len}')

    # Создаём список с вложенным циклом по длинам текстов, где i - 100% текста, i/5 - 20% текста
    train_len_shares = [(i - round(i/5)) for i in texts_len]
    t_num = 0
    train_len_share_list = []
    # Циклом проводим итерацию по списку с объёмами текстов равными 80% от исходных
    for train_len_share in train_len_shares:
        t_num += 1
        print(f'Доля 80% от текста №{t_num}: {train_len_share} символов')
        train_len_share_list.append(train_len_share)

    return train_len_share_list

def slice_text_for_parts(texts_list):
    # Создадим пустой список для обучающей выборки
    train_data = []
    test_data = []

    # Поделим тексты по индексу
    train_data = list(chain(train_data, (texts_list[0][:170705], texts_list[1][:107628])))
    test_data = list(chain(test_data, (texts_list[0][170705:], texts_list[1][107628:])))

    # Убедимся, что кол-во классов и типы данных всё те же
    print("train_data: ",len(train_data))
    print(type(train_data))
    print(type(train_data[0]))

    print("test_data: ",len(test_data))
    print(type(test_data))
    print(type(test_data[0]))

    return train_data, test_data


