import cv2
import numpy as np

def extract_colors(img, block_size):
    """ Extract average color for each block with given size
        Return a list for each color with the average values for that color
        Input must be a BGR image and a valid size for the blocks
    """

    red = []
    green = []
    blue = []

    for y in range(0, int(img.shape[0]/block_size)):
        for x in range(0, int(img.shape[1] / block_size)):
            blue.append(np.average(img[y*block_size:y*block_size+block_size, x*block_size:x*block_size+block_size, 0]))
            green.append(np.average(img[y*block_size:y*block_size+block_size, x*block_size:x*block_size+block_size, 1]))
            red.append(np.average(img[y*block_size:y*block_size+block_size, x*block_size:x*block_size+block_size, 2]))

    return [red, green, blue]
