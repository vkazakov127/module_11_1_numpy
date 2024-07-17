# -*- coding: utf-8 -*-
# module_11_1_numpy.py
import numpy as np
from scipy.datasets import face
import matplotlib.pyplot as plt

# Изображение как массив из "scipy.datasets"
img = face()
print(f'Изображение как массив. type(img) = {type(img)}; img.shape= {img.shape}')
# Переведём в чёрно-белый
img_array = img / 255
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
print(f'Чёрно-белый массив. type(img_gray) = {type(img_gray)}; img_gray.shape= {img_gray.shape}')
# Покажем "img_gray"
plt.imshow(img_gray, cmap="gray")
plt.show()
# -------- Преобразования матриц, что собственно и требуется в Задании
# ----- Поворот матрицы влево
print(f'----- Поворот матрицы влево')
img_gray2 = np.rot90(img_gray)
# Показать "img_gray2"
plt.imshow(img_gray2, cmap="gray")
plt.show()
# ----- Транспонирование
print(f'----- Транспонирование')
img_gray3 = img_gray.transpose()
# Показать "img_gray3"
plt.imshow(img_gray3, cmap="gray")
plt.show()
# ----- Умножение на единичную матрицу
print(f'----- Умножение на единичную матрицу')
# ----- Изготовим "Единичную квадратную матрицу"
# Определим требуемый размер матрицы, под размер "img_gray"
img_gray_shape = img_gray.shape
print(f'Размеры матрицы-источника = {img_gray_shape}')
# Единичная квадратная матрица
ones = np.eye(img_gray_shape[0], dtype="int8")
print(f'Единичная квадратная матрица "ones". type(ones) = {type(ones)}; ones.shape = {ones.shape}')
# Квадратная матрица - первый операнд
img_gray4 = img_gray[:img_gray_shape[0], :img_gray_shape[0]]
print(f'Квадратная матрица - Операнд №1. "img_gray4". type(img_gray4) = {type(img_gray4)}; img_gray4.shape = {img_gray4.shape}')
# Собственно умножение на Единичную матрицу
print(f'Собственно умножение Операнд №1 на Единичную квадратную матрицу')
img_gray5 = img_gray4 @ ones
# Результат: ничего не должно измениться в сравнении с Операндом 1
print(f'Результат: ничего не должно измениться в сравнении с Операндом 1. Показываем картинку.')
# Показать "img_gray5"
plt.imshow(img_gray5, cmap="gray")
plt.show()

print('----------- The End -----------')