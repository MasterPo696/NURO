import cv2
import pytesseract
from PIL import Image

# API_AutoCode = "https://b2bapi.avtocod.ru/b2b/api/v1/"

# Укажите путь к Tesseract, если он не добавлен в PATH
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # для macOS

# Загрузка изображения с номером
image = cv2.imread("CcontrolL/data/x_data/samples/sample2.png")

# Получаем размеры изображения, а также закрашиваем тот кусочек, что не нужен
height, width, _ = image.shape
x_start = int(width * 10 / 13)  # Начало по оси X (1/4 от правого края)
y_start = int(height* 2 / 3)     # Начало по оси Y (1/2 от нижнего края)
x_end = width
y_end = height
image[y_start:y_end, x_start:x_end] = (255, 255, 255)  # BGR: для белого цвета

# Отображаем результат
# cv2.imshow("Image with Colored Area", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Преобразуем в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применяем бинаризацию (чтобы улучшить распознавание)
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

# Используем pytesseract для извлечения текста
extracted_text = pytesseract.image_to_string(binary_image, config='--psm 8')

# Выводим извлеченный текст
print("Извлеченный текст:", extracted_text)
