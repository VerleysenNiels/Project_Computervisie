import cv2
import platform
import numpy as np


def main():
    print('python=={}'.format(platform.python_version()))
    print('opencv-contrib-python=={}'.format(cv2.__version__))
    print('numpy=={}'.format(np.__version__))


if __name__ == "__main__":
    main()
