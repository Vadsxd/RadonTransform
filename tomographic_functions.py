import numpy as np
from PIL import ImageMath
from scipy.fftpack import fft, ifft, fftfreq


def radon_transform(image):
    """Implementation of the radon transform"""

    # Need a numpy array to represent the image values
    npImage = np.array(image)
    # Also need a float copy of the image for logistical reasons
    floatImage = ImageMath.eval("float(a)", a=image)

    steps = image.size[0]
    # Empty array to store the newly created sinogram
    radon = np.zeros((steps, len(npImage)), dtype='float64')

    # For each step, we need to roate the image and sum along the vertical lines, this is the DRT
    for step in range(steps):
        rotation = floatImage.rotate(-step * 180 / steps)
        npRotate = np.array(rotation)
        radon[:, step] = sum(npRotate)
    return radon


def inverse_radon_transform(sinogram, limit):
    """Inverse of the radon transform, reconstructs a sinogram"""
    if type(sinogram) != "<class 'numpy.ndarray'>":
        sinogram = np.array(sinogram)
    size = len(sinogram)
    theta = np.linspace(0, limit, len(sinogram), endpoint=False) * (np.pi / 180.0)

    # First, we need to pad the image

    # the maximum projection size is the nearest power of 2 fromt the sinogram size
    # the mimimum must be 64, as this is the standard in calculation
    max_projeciton_size = max(64, int(2 ** np.ceil(np.log2(2 * len(sinogram)))))

    # Pad the image with 0's
    # this makes computation easier and ensures we do not lose anything
    pad_width = ((0, max_projeciton_size - len(sinogram)), (0, 0))
    padded_sinogram = np.pad(sinogram, pad_width, mode="constant", constant_values=0)

    # Now we need to deploy the filters, these are based on the Fourier Transforms
    # First, get the frequencies
    filter = fftfreq(max_projeciton_size).reshape(-1, 1)

    # One of the best filters to apply is the Ram-Lak filter
    # Apply the fourier transform
    ram_lak = 2 * np.abs(filter)
    fourier_projection = fft(padded_sinogram, axis=0) * ram_lak

    # We only want the real parts of the inverse fourier transform
    radon_filter = np.real(ifft(fourier_projection, axis=0))
    radon_filter = radon_filter[:sinogram.shape[0], :]

    # Prepare a place for the reconstrcted image to go
    reconstructed_image = np.zeros((size, size))

    middle = size // 2

    # backprojection begins
    [X, Y] = np.mgrid[0:size, 0:size]
    xprojection = X - int(size) // 2
    yprojection = Y - int(size) // 2

    for i in range(len(theta)):
        angle = yprojection * np.cos(theta[i]) - xprojection * np.sin(theta[i])
        s = np.arange(radon_filter.shape[0]) - middle

        # linear interpolation
        projection = np.interp(angle, s, radon_filter[:, i], left=0, right=0)
        reconstructed_image += projection

    # Eliminate the pixels that fall ouside the reconstruction cirlce
    radius = size // 2
    circle = (xprojection ** 2 + yprojection ** 2) <= radius ** 2
    reconstructed_image[~circle] = 0
    return reconstructed_image  # * np.pi / (360)
