import gdown
import os
import zipfile

def download_data():
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l7/20writers.zip', None, quiet=True) 


def unzip_with_cleanup(zip_file, extract_path, from_enc="cp437", to_enc="cp866"):
    """
    Распаковка zip-архива с корректной обработкой кодировок и очисткой "мусорных" имен.
    """
    if not os.path.exists(zip_file):
        print(f'Файл {zip_file} не найден!')
        return []

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Декодируем имя файла из исходной кодировки
                try:
                    unicode_name = file_info.filename.encode(from_enc).decode(to_enc)
                except UnicodeDecodeError:
                    unicode_name = file_info.filename  # Файлы с ошибками оставляем как есть

                target_path = os.path.join(extract_path, unicode_name)

                # Создаем папки, если нужно
                if file_info.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Извлекаем файл
                with zip_ref.open(file_info.filename) as source, open(target_path, 'wb') as target:
                    target.write(source.read())

        print(f'Архив {zip_file} успешно распакован в {extract_path}')
    except (RuntimeError, UnicodeDecodeError) as e:
        print(f"Ошибка при распаковке: {e}")

    # Удаляем "мусорные" файлы
    cleanup_invalid_files(extract_path)

    # Список классов
    CLASS_LIST = sorted(os.listdir(extract_path))
    CLASS_COUNT = len(CLASS_LIST)
    print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')
    return CLASS_LIST

def cleanup_invalid_files(folder):
    """
    Удаление файлов с некорректными именами.
    """
    for root, dirs, files in os.walk(folder):
        for name in files + dirs:
            try:
                # Попытка декодировать имя в UTF-8
                name.encode('utf-8').decode('utf-8')
            except UnicodeEncodeError:
                # Если декодирование не удается, удаляем файл/папку
                full_path = os.path.join(root, name)
                if os.path.isdir(full_path):
                    os.rmdir(full_path)
                else:
                    os.remove(full_path)
                print(f"Удален файл/папка с некорректным именем: {name}")


# Объявляем функции для чтения файла. На вход отправляем путь к файлу
def read_text(file_name):
  read_file = open(file_name, 'r')
  text = read_file.read()
  text = text.replace("\n", " ")
  return text


def get_dir_list(path="data/writters/"):
    texts_list = []
    for j in os.listdir(path):
        texts_list.append(read_text(path + j))
        print(j, 'добавлен в обучающую выборку')

    print(len(texts_list))

    return texts_list



def get_texts_80_value(texts_list):

    def get_texts_value(texts_list):
        texts_len = [len(text) for text in texts_list]
        t_num = 0
        print(f'Размеры текстов по порядку (в токенах):')
        for text_len in texts_len:
            t_num += 1
            print(f'Текст №{t_num}: {text_len}')

        return texts_len
    
    texts_len = get_texts_value(texts_list)

    train_len_shares = [(i - round(i/5)) for i in texts_len]
    t_num = 0
    for train_len_share in train_len_shares:
        t_num += 1
        print(f'Доля 80% от текста №{t_num}: {train_len_share} токенов')

    return train_len_shares




# # Объявляем интересующие нас классы
# class_names = ["Беляев", "Булгаков", "Васильев", "Гоголь", "Гончаров", "Горький", "Грибоедов",
#                "Достоевский", "Каверин", "Катаев", "Куприн", "Лермонтов", "Лесков", "Носов",
#                "Пастернак", "Пушкин", "Толстой", "Тургенев", "Чехов", "Шолохов"]

# # Считаем количество классов
# num_classes = len(class_names)