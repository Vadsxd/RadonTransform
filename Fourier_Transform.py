import numpy as np

'''
Вычиляет квадратную матрицу N x N с помощью быстрого преобразования Фурье
Идем по массиву М и присваиваем элементу в ячейке значение из формулы в 14 строке,
используя вектор k и n.
'''


def fourier_transform(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
