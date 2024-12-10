import requests
from PIL import Image
from io import BytesIO
import os
from urllib.parse import urlparse

# Настройки
dataset_url = "https://datasets-server.huggingface.co/rows"
dataset = "huggan%2Fwikiart"
config = "default"
split = "train"
output_dir = "wikiart_images_by_class"
length = 10  # Загружаем первые 10 записей для отладки
total_records = 10  # Ограничим до 10 записей для теста

# Создаем папку для сохранения изображений
os.makedirs(output_dir, exist_ok=True)

# Скачиваем данные частями
for offset in range(0, total_records, length):
    print(f"Processing records {offset} to {offset + length}...")
    try:
        # Запрос данных через API
        url = f"{dataset_url}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        response = requests.get(url)
        response.raise_for_status()  # Проверка успешности запроса
        print(f"Response received: {response.status_code}")

        # Печать ответа для проверки структуры данных
        data = response.json()
        print(data)  # Проверьте структуру данных в ответе

        # Обработка каждой записи
        for i, row in enumerate(data.get("rows", [])):
            try:
                # Извлекаем информацию об изображении и классе
                image_url = row["row"]["image"]
                class_name = row["row"]["style"]  # Убедитесь, что ключ 'style' есть в данных
                if not class_name:
                    class_name = "unknown"

                # Убедитесь, что class_name является строкой
                class_name = str(class_name)

                # Проверка на валидность URL
                if not image_url or not isinstance(image_url, str) or not urlparse(image_url).scheme:
                    print(f"Invalid image URL: {image_url}")
                    continue

                # Создаем папку для класса
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # Скачиваем изображение
                img_data = requests.get(image_url)
                if img_data.status_code == 200:
                    img = Image.open(BytesIO(img_data.content))

                    # Уменьшение размера и сохранение
                    img = img.resize((224, 224))  # Пример разрешения
                    img.save(os.path.join(class_dir, f"{str(offset + i)}.jpg"), quality=50)
                    print(f"Saved image {offset + i} in class '{class_name}'")
                else:
                    print(f"Failed to download image from {image_url}")
            except Exception as e:
                print(f"Error downloading/saving image {offset + i}: {e}")
    except Exception as e:
        print(f"Error processing batch starting at {offset}: {e}")
