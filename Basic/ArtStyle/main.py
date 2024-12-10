import os
import requests
import hashlib
from PIL import Image
from io import BytesIO
from config import API_KEY, API_KEY_SECRET, STYLES

ACCESS_CODE = API_KEY
SECRET_CODE = API_KEY_SECRET

styles = STYLES

def get_token():
    """Получает токен для доступа к API"""
    url = "http://www.wikiart.org/en/Api/2/login"
    params = {"accessCode": ACCESS_CODE, "secretCode": SECRET_CODE}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("SessionKey")

def is_valid_image(image_data):
    """Проверка на валидность изображения"""
    try:
        img = Image.open(BytesIO(image_data))
        img.verify()  # Проверка целостности
        return True
    except Exception:
        return False

def get_image_hash(image_data):
    """Возвращает хэш изображения для проверки уникальности"""
    hash_object = hashlib.md5(image_data)
    return hash_object.hexdigest()

def download_unique_paintings_by_style(token, style, save_dir="dataset", max_images=100):
    """Скачивает уникальные картины по стилю"""
    style_dir = os.path.join(save_dir, style)
    os.makedirs(style_dir, exist_ok=True)
    
    # Множество для хранения хешей изображений
    unique_image_hashes = set()
    
    page_number = 1
    images_downloaded = 0

    while images_downloaded < max_images:
        url = "http://www.wikiart.org/en/search/Any/1"
        params = {"json": 2, "style": style, "PageSize": 100, "page": page_number}
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        paintings = response.json()
        
        if not paintings:  # Если картин больше нет на странице, выходим
            break
        
        for painting in paintings:
            if images_downloaded >= max_images:
                break
            
            image_url = painting.get("image")
            title = painting.get("title", "unknown").replace(" ", "_")
            file_path = os.path.join(style_dir, f"{title}.jpg")
            
            if image_url and not os.path.exists(file_path):
                image_response = requests.get(image_url)
                
                if image_response.status_code == 200:
                    if is_valid_image(image_response.content):
                        image_hash = get_image_hash(image_response.content)
                        
                        if image_hash not in unique_image_hashes:  # Проверяем на уникальность
                            with open(file_path, "wb") as f:
                                f.write(image_response.content)
                            unique_image_hashes.add(image_hash)  # Добавляем хэш в множество
                            images_downloaded += 1
                            print(f"Downloaded: {title}")
                        else:
                            print(f"Duplicate image skipped: {title}")
                    else:
                        print(f"Corrupted image: {title}")
                else:
                    print(f"Failed to download: {title}")
        
        page_number += 1  # Переходим к следующей странице

def download_all_styles(token, styles, save_dir="dataset"):
    """Скачивает картины для всех стилей"""
    for style in styles:
        print(f"Downloading {style} style...")
        download_unique_paintings_by_style(token, style, save_dir)
        
# Использование
try:
    token = get_token()
    download_all_styles(token, styles)
except requests.RequestException as e:
    print(f"Error: {e}")
