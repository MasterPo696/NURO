
from keras import layers, models, optimizers, preprocessing, utils, datasets
import cv2
import pylab                                        # Модуль для построения графиков
import matplotlib.pyplot as plt                     # Отрисовка изображений
from PIL import Image                               # Отрисовка изображений

mnist = datasets.mnist                             # Библиотека с базой Mnist
Sequential = models.Sequential                      # Подлючение класса создания модели Sequential
Dense = layers.Dense                                # Подключение класса Dense (полносвязного слоя) 
Adam = optimizers.Adam                              # Подключение оптимизатора Adam
utils = utils                                       # Утилиты для to_categorical
image = preprocessing.image                         # Для отрисовки изображения
load_img = image.load_img                           # Метод для загрузки изображений
import numpy as np



# Загружаем классификатор для обнаружения машин
car_cascade = cv2.CascadeClassifier('CcontrolL/data/haarcascade/haarcascade_fullbody.xml')



IMAGE_PATH = "CcontrolL/data/x_data/test/"
IMAGE_NAME = IMAGE_PATH + "CAR.jpeg"
image = cv2.imread("CcontrolL/data/x_data/test/test.png")

# Преобразуем изображение в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаруживаем машины на изображении
cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Если машины найдены, обводим их зеленым прямоугольником
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Отображаем результат
cv2.imshow("Detected Cars", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Применяем адаптивный порог для улучшения контуров
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Находим контуры
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
    
#     if len(approx) == 4:  # Если найден прямоугольник
#         # Преобразуем в координаты области
#         plate_contour = approx
#         break

#     # Рисуем прямоугольник вокруг найденного номера
# if len(plate_contour) == 4:
#     cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
#     cv2.imshow("Detected License Plate", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()