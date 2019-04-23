import cv2
import numpy as np


def order_points(pts):
    """Orders points clockwise, starting w top left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(img, points):
    """Transform an image's perspective given 4 (unordered) points and crop the image around it
    """
    if len(points) < 4:
        return img

    points = order_points(points)

    # Calculate destination rectangle
    x_max, y_max = 496, 496  # points.max(axis=0)
    x_min, y_min = 0, 0  # points.min(axis=0)
    dst = np.float32([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ])
    M = cv2.getPerspectiveTransform(points, dst)
    img = cv2.warpPerspective(img, M, img.shape[1::-1])
    crop_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    return crop_img
