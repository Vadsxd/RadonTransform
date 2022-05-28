import numpy as np
from PIL import ImageMath

from Filter import filter_func
from Fourier_Transform import fourier_transform
from Inverse_Fourier_Transform import inverse_fourier_transform
from Filter_Pict import make_pict_unfiltered, make_pict_filtered


def np_interp(angle, s, radon_filter):
    c = 1
    S_C = [0] * len(s)
    for i in range(len(s)):
        S_C[i] = s[i] / c

    len_s = len(s)
    vand_matr = [[0] * len_s for _ in range(len_s)]
    for i in range(len_s):
        for j in range(len_s):
            vand_matr[i][j] = S_C[i] ** j

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

    Z = [0] * len_s
    for i in range(1, len_s + 1):
        sum = 0
        for k in range(1, i):
            sum += Z[k - 1] * L[i - 1][k - 1]
        Z[i - 1] = radon_filter[i - 1] - sum

    T = [0] * len_s
    for i in range(len_s, 0, -1):
        sum = 0
        for k in range(i + 1, len_s + 1):
            sum += T[k - 1] * U[i - 1][k - 1]
        T[i - 1] = (Z[i - 1] - sum) / U[i - 1][i - 1]

    A = [0] * len_s
    for i in range(len(s)):
        A[i] = T[i] / (c ** i)

    projection = np.zeros((len_s, len_s))
    for i in range(len_s):
        for j in range(len_s):
            for k in range(len_s):
                projection[i][j] += A[len_s - 1 - k] * (angle[i][j] ** (len_s - k - 1))

    return projection


def radon_transform(image):
    npImage = np.array(image)
    floatImage = ImageMath.eval("float(a)", a=image)

    steps = image.size[0]
    radon = np.zeros((steps, len(npImage)), dtype='float64')

    for step in range(steps):
        rotation = floatImage.rotate(-step * 180 / steps)
        npRotate = np.array(rotation)
        radon[:, step] = sum(npRotate)
        # plt.imshow(rotation, cmap="gray")
        # plt.show()
    return radon


def inverse_radon_transform(sinogram, limit):
    """Обратное преобразование радона, восстанавливает синограмму"""
    if type(sinogram) != "<class 'numpy.ndarray'>":
        sinogram = np.array(sinogram)
    size = len(sinogram)

    theta = np.linspace(0, limit, len(sinogram), endpoint=False) * (np.pi / 180.0)

    max_projeciton_size = max(64, int(2 ** np.ceil(np.log2(2 * len(sinogram)))))

    pad_width = ((0, max_projeciton_size - len(sinogram)), (0, 0))
    padded_sinogram = np.pad(sinogram, pad_width, mode="constant", constant_values=0)

    filter = filter_func(max_projeciton_size)
    make_pict_unfiltered(filter.reshape(1, -1))

    filter = filter_func(max_projeciton_size).reshape(-1, 1)
    ram_lak = 2 * np.abs(filter)

    make_pict_filtered(ram_lak.reshape(1, -1))
    fourier_projection = fourier_transform(padded_sinogram) * ram_lak

    res = inverse_fourier_transform(fourier_projection)

    radon_filter = np.real(res)
    radon_filter = radon_filter[:sinogram.shape[0], :]

    reconstructed_image = np.zeros((size, size))

    middle = size // 2

    [X, Y] = np.mgrid[0:size, 0:size]
    xprojection = X - int(size) // 2
    yprojection = Y - int(size) // 2

    for i in range(len(theta)):
        angle = yprojection * np.cos(theta[i]) - xprojection * np.sin(theta[i])
        s = np.arange(radon_filter.shape[0]) - middle

        projection = np.interp(angle, s, radon_filter[:, i])
        reconstructed_image += projection

    radius = size // 2
    circle = (xprojection ** 2 + yprojection ** 2) <= radius ** 2
    reconstructed_image[~circle] = 0
    return reconstructed_image
