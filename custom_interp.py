import numpy as np
from PIL import ImageMath
from scipy.fftpack import fft, ifft, fftfreq


def radon_transform(image):
    """Реализация преобразования радона"""

    # Нужен массив numpy для записи значений изображения
    npImage = np.array(image)
    # Также нужна float копия изображения по логистическим соображениям
    floatImage = ImageMath.eval("float(a)", a=image)

    steps = image.size[0] # ? количетво строк в изображении
    # Пустой массив для хранения вновь созданной синограммы
    radon = np.zeros((steps, len(npImage)), dtype='float64')

    # Для каждого шага нам нужно поворачивать изображение и суммировать вдоль вертикальных линий, это DRT
    for step in range(steps):
        rotation = floatImage.rotate(-step * 180 / steps) # поворот изображения по часовой стрелке на очередной угол
        npRotate = np.array(rotation)
        radon[:, step] = sum(npRotate) # суммируем элементы каждой строки в npRotate, записываем полученные значения в столбец массива radon
    return radon


def interp(angle, s, radon_filter):
    c = 1
    S_C = [0] * len(s)
    for i in range(len(s)):
       S_C[i] = s[i] / c

    len_s = len(s)
    # строим матрицу Вандермонда
    vand_matr = np.zeros((len_s, len_s))
    for i in range(len_s):
        for j in range(len_s):
            vand_matr[i][j] = S_C[i] ** j
        # раскладываем ее LU разложением
    L = np.identity(len_s)
    U = np.zeros((len_s, len_s))

    for i in range(1, len_s + 1):
        for j in range(1, len_s + 1):
            if i <= j:
                sum = 0
                for k in range(1, i):
                    sum += L[i - 1][k - 1] * U[k - 1][j - 1]
                U[i - 1][j - 1] = vand_matr[i - 1][j - 1] - sum

            if i > j:
                sum = 0
                for k in range(1, j):
                    sum += L[i - 1][k - 1] * U[k - 1][j - 1]
                L[i - 1][j - 1] = (vand_matr[i - 1][j - 1] - sum) / U[j - 1][j - 1]

    # ищем коэффициенты А из уравнения A = X^-1 * Y в два этапа
    # L * Z = F
    Z = [0]*len_s
    for i in range(1, len_s + 1):
        sum = 0
        for k in range(1, i):
            sum += Z[k - 1] * L[i - 1][k - 1]
        Z[i - 1] = radon_filter[i - 1] - sum

    # U * A = Z
    T = [0] * len_s
    for i in range(len_s, 0, -1):
        sum = 0
        for k in range(i + 1, len_s + 1):
            sum += T[k - 1] * U[i - 1][k - 1]
        T[i - 1] = (Z[i - 1] - sum)/U[i - 1][i - 1]

    A = [0] * len_s
    for i in range(len(s)):
       A[i] = T[i] / c ** i
    # из полученного многочлена вычисляем значение projection подставляя angle
    projection = np.zeros((len_s, len_s))
    for i in range(len_s):
        for j in range(len_s):
            for k in range(len_s):
                projection[i][j] += A[len_s - 1 - k] * (angle[i][j] ** (len_s - k - 1))

    return projection


def inverse_radon_transform(sinogram, limit):

    """Обратное преобразование радона, восстанавливает синограмму"""
    if type(sinogram) != "<class 'numpy.ndarray'>":
        sinogram = np.array(sinogram)
    size = len(sinogram)

    # одномерный массив длинны  len(sinogram) значения в котором равномерно распределенны внутри полуоткрытого интервала [0, limit) и умножены на p/180.
    theta = np.linspace(0, limit, len(sinogram), endpoint=False) * (np.pi / 180.0)

    # Во-первых, нам нужно дополнить изображение

    # максимальный размер проекции равен ближайшей степени 2 от размера синограммы
    # минимум должен быть 64, так как это стандарт при расчете
    max_projeciton_size = max(64, int(2 ** np.ceil(np.log2(2 * len(sinogram)))))

    # Дополним изображение цифрами 0
    pad_width = ((0, max_projeciton_size - len(sinogram)), (0, 0))
    padded_sinogram = np.pad(sinogram, pad_width, mode="constant", constant_values=0)

    # Теперь нам нужно развернуть фильтры, они основаны на преобразованиях Фурье
    # Во-первых, получим частоты
    filter = fftfreq(max_projeciton_size).reshape(-1, 1)

    # Одним из лучших фильтров для применения является фильтр Ram-Lak
    # Применим преобразование Фурье
    ram_lak = 2 * np.abs(filter)
    fourier_projection = fft(padded_sinogram, axis=0) * ram_lak

    # Нам нужны только действительные части обратного преобразования Фурье
    radon_filter = np.real(ifft(fourier_projection, axis=0)) # вычисляет одномерное обратное дискретное преобразование Фурье
    radon_filter = radon_filter[:sinogram.shape[0], :] # обрезаем по размеру синограммы

    # Подготовим место для размещения восстановленного изображения
    reconstructed_image = np.zeros((size, size))

    middle = size // 2

    # начало обратной проекции
    [X, Y] = np.mgrid[0:size, 0:size]
    xprojection = X - int(size) // 2
    yprojection = Y - int(size) // 2

    s = np.arange(radon_filter.shape[0]) - middle
    for i in range(len(theta)):
        angle = yprojection * np.cos(theta[i]) - xprojection * np.sin(theta[i])

        # линейная интерполяция
        projection = np.interp(angle, s, radon_filter[:, i])
        reconstructed_image += projection

    # Удалим пиксели, которые выходят за пределы круга восстановления
    radius = size // 2
    circle = (xprojection ** 2 + yprojection ** 2) <= radius ** 2
    reconstructed_image[~circle] = 0
    return reconstructed_image  # * np.pi / (360)
