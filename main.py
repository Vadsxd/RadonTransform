from PIL import Image
from display_functions import *

if __name__ == '__main__':
    image = Image.open('pictures/альт8.jpg')
    # animate_transform(image)
    animate_reconstruction(image)
    show_all_images(image)
