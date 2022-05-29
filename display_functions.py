import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tomographic_functions import *


def animate_transform(original_image):
    fig = plt.figure()

    npImage = np.array(original_image)

    floatImage = ImageMath.eval("float(a)", a=original_image)

    steps = original_image.size[0]

    radon = np.zeros((steps, len(npImage)), dtype='float64')

    images = []
    for step in range(steps):
        rotation = floatImage.rotate(-step * 180 / steps)
        npRotate = np.array(rotation)
        radon[:, step] = sum(npRotate)
        image = plt.imshow(radon, cmap="gray", animated=True)
        images.append([image])

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=False, repeat=False)
    writergif = animation.PillowWriter(fps=30)
    ani.save("TransformAnimation.gif", writer=writergif)
    plt.show()


def animate_reconstruction(original_image):
    fig = plt.figure()

    radon_image = radon_transform(original_image)

    images = []
    for i in range(181):
        reconstructed_image = inverse_radon_transform(radon_image, i)
        image = plt.imshow(reconstructed_image, cmap="gray", animated=True)
        images.append([image])
        # print(i)

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=False, repeat=False)
    writergif = animation.PillowWriter(fps=30)
    ani.save("ReconstructionAnimation.gif", writer=writergif)
    plt.show()


def show_all_images(original_image, steps=180):
    radon_image = radon_transform(original_image)
    reconstructed_image = inverse_radon_transform(radon_image, steps)
    result_visualisation(reconstructed_image.reshape(1, -1))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(original_image, cmap="gray")
    ax1.set_title("Original Image")
    ax2.imshow(radon_image, cmap="gray")
    ax2.set_title("Sinogram")
    ax3.imshow(reconstructed_image, cmap="gray")
    ax3.set_title("Reconstructed Image")
    plt.show()
