import cv2
import os
import logging
import numpy as np
import viz_utils


def imread(path, resize=True, bw=False):
    """Read an image and resize to 1080p.
    """
    logging.info('Reading ' + path)
    img = cv2.imread(path)
    if resize:
        img = cv2.resize(
            img, (0, 0),
            fx=1080 / img.shape[0],
            fy=1080 / img.shape[0],
            interpolation=cv2.INTER_AREA)
    if bw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def imread_folder(folder, resize=True, bw=False):
    """Read all images from a directory into a vector
    Returns:
        An iterator of (path, image) tuples
    """
    for root, directories, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(('.png', '.jpeg', '.jpg')):
                path = os.path.join(root, filename)
                img = imread(path, resize=resize, bw=bw)
                yield (path, img)


def imwrite(path, image):
    """Write image to path, creating directories if necessary
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cv2.imwrite(path, image)


def read_video(path, interval=15):
    video = cv2.VideoCapture(path)
    id = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if(id % interval == 0):
            yield frame
        id += 1

    video.release()
