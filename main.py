import cv2
import platform
import numpy as np
import os
import argparse


def main():
    print('python=={}'.format(platform.python_version()))
    print('opencv-contrib-python=={}'.format(cv2.__version__))
    print('numpy=={}'.format(np.__version__))


if __name__ == "__main__":
    main()

    parser = argparse.ArgumentParser(description='Locate a painting in the MSK')
    parser.add_argument('--input', help='Path to input image.',
                        default='images/single_paintings/animals/20190217_104932.jpg')
    args = parser.parse_args()
    image_to_locate = cv2.imread(args.input)

    #images = []
    zalen = []
    im_folders = os.listdir("images/zalen")
    for folder in im_folders:
        ims_in_folder = os.listdir("images/zalen/" + folder)
        print(folder)
        for im in ims_in_folder:
            image = cv2.imread("images/zalen/" + folder + "/" + im, 1)
            #images.append(image)
            zalen.append(folder)
