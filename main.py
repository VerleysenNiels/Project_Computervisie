import cv2
import platform
import numpy as np
import os


def main():
    print('python=={}'.format(platform.python_version()))
    print('opencv-contrib-python=={}'.format(cv2.__version__))
    print('numpy=={}'.format(np.__version__))


if __name__ == "__main__":
    main()
    images = []
    im_folders = os.listdir("/images")
    for folder in im_folders:
        ims_in_folder = os.listdir("/images/" + folder)
        for im in ims_in_folder:
            image = cv2.imread()
            images.append(image)
