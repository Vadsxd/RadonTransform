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
    # это упрощает вычисления и гарантирует, что мы ничего не потеряем
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

    for i in range(len(theta)):
        angle = yprojection * np.cos(theta[i]) - xprojection * np.sin(theta[i])
        s = np.arange(radon_filter.shape[0]) - middle

        # линейная интерполяция
        projection = np.interp(angle, s, radon_filter[:, i], left=0, right=0)
        reconstructed_image += projection

    # Удалим пиксели, которые выходят за пределы круга восстановления
    radius = size // 2
    circle = (xprojection ** 2 + yprojection ** 2) <= radius ** 2
    reconstructed_image[~circle] = 0
    return reconstructed_image  # * np.pi / (360)
