import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tomographic_functions import *

#
def animate_reconstruction(original_image):
    # создание окна
    fig = plt.figure()
    # создание синограммы
    radon_image = radon_transform(original_image)

    images = []
    # создание слайдов анимации (на каждой итерации выполняем обратное преобразование только до i градусов)
    for i in range(181):
        # восстановление исходного изображения
        reconstructed_image = inverse_radon_transform(radon_image, i)
        image = plt.imshow(reconstructed_image, cmap="gray", animated=True)
        images.append([image])

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=False, repeat=False)
    writergif = animation.PillowWriter(fps=30)
    ani.save("Animation.gif", writer=writergif)
    plt.show()


def show_all_images(original_image, steps=180):
    radon_image = radon_transform(original_image)
    reconstructed_image = inverse_radon_transform(radon_image, steps)

    # создание вторрого окна с демонстрацией исходного изображения, синограммы и восстановленного изображения
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(original_image, cmap="gray")
    ax1.set_title("Original Image")
    ax2.imshow(radon_image, cmap="gray")
    ax2.set_title("Sinogram")
    ax3.imshow(reconstructed_image, cmap="gray")
    ax3.set_title("Reconstructed Image")

    plt.show()
