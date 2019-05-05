import cv2
import numpy as np
import math_utils


def imshow(img, name='Image', norm=False, resize=False):
    """Display an image
    """
    if resize:
        img = cv2.resize(
            img, (0, 0),
            fx=720 / img.shape[0],
            fy=720 / img.shape[0],
            interpolation=cv2.INTER_NEAREST)
    if norm:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(name, img.astype('uint8'))
    cv2.waitKey(500)
    # cv2.destroyAllWindows()


def overlay_lines(img, lines):
    """Overlay lines in polar coordinates on an image
    """
    out = np.copy(img)
    for i in lines:
        for rho, theta in i:
            x1, y1, x2, y2 = math_utils.to_carthesian(rho, theta)
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return out


def overlay_lines_cartesian(img, lines):
    """Overlay lines in Cartesian coordinates on an image
    lines: array containing (x1, y1, x2, y2) quadruplets
    """
    out = np.copy(img)
    for line in lines:
        cv2.line(out, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    return out


def overlay_points(img, points):
    """Overlay points on an image
    """
    out = np.copy(img)
    for i, point in enumerate(points):
        out = cv2.circle(out,
                         tuple(point),       # center
                         img.shape[0]//200,  # radius
                         ((47 * i) % 255, (67 * i) %
                          255, (97 * i) % 255),  # random color
                         img.shape[0]//200,
                         lineType=cv2.FILLED
                         )
    return out
