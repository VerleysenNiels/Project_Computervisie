import cv2
import numpy as np
import math_utils


def imshow(img, name='Image', norm=False, resize=False):
    """Display an image

    Arguments:
        img  -- The image to be shown. If the dtype is `float` or `signed`,
            the `norm` should be set to true


    Keyword Arguments:
        name {str} -- The name of the window (default: {'Image'})
        norm {bool} -- Normalize the image between 0 and 255 (default: {False})
        resize {bool} -- Resize the image to be 500px wide (default: {False})
    """

    if resize:
        img = cv2.resize(
            img, (0, 0),
            fx=500 / img.shape[0],
            fy=500 / img.shape[0],
            interpolation=cv2.INTER_NEAREST)
    if norm:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(name, img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def overlay_lines(img, lines):
    """Overlay lines in polar coordinates on an image
    """
    for i in lines:
        for rho, theta in i:
            x1, y1, x2, y2 = math_utils.to_carthesian(rho, theta)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return img


def overlay_lines_cartesian(image, lines):
    for line in lines:
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    return image


def overlay_points(img, points):
    out = img
    for i, point in enumerate(points):
        out = cv2.circle(out,
                         tuple(point),  # center
                         img.shape[0]//200,                 # radius
                         ((47 * i) % 255, (67 * i) %
                          255, (97 * i) % 255),  # random color
                         img.shape[0]//200,
                         lineType=cv2.FILLED
                         )
    return out
