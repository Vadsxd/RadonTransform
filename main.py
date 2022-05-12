from PIL import Image
from display_functions import *

if __name__ == '__main__':
    image = Image.open('/home/vadim/PycharmProjects/MathStaff/pictures/PictureExamp.png')
    animate_reconstruction(image)
    show_all_images(image)

