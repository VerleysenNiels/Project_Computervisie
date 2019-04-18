import cv2
import os


def imread(path, resize=True, bw=False):
    """Read an image and resize to 1080p.
    """
    print(path)
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
        A vector of (path, image) tuples
    """
    res = []
    ls = os.listdir(folder)
    for path in ls:
        if path.endswith(('.png', '.jpeg', '.jpg')):
            img = imread(folder + '/' + path, resize=resize, bw=bw)
            yield (folder + '/' + path, img)
