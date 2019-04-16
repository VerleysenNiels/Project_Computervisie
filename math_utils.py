import numpy as np
from itertools import combinations
import random


def to_cartesian(line):
    """Convert lines in polar system to lines in Cartesian system."""
    rho, theta = line[0], line[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    return x1, y1, x2, y2


def eliminate_duplicates(lines, a_threshold=.1, b_threshold=50):
    """Remove similar lines, based on a threshold

    Arguments:
        lines {array} -- Numpy array of Carthesian coordinates: x1, y1, x2, y2

    Keyword Arguments:
        a_threshold {float} -- Slope threshold (default: {.1})
        b_threshold {int} -- Vertical bias threshold (default: {50})

    Returns:
        array -- Filtered array of lines
    """

    eliminated = np.zeros(len(lines), dtype=bool)

    for i, j in combinations(range(len(lines)), 2):
        if eliminated[i] or eliminated[j]:
            continue

        a, b = lines[i], lines[j]
        x1, y1, x2,  y2 = a[0], a[1], a[2], a[3]
        x3, y3, x4, y4 = b[0], b[1], b[2], b[3]
        a1 = (y2-y1) / (x2-x1)
        a2 = (y4-y3) / (x4-x3)

        b1 = -a1 * x1 + y1
        b2 = -a2 * x3 + y3

        if abs(a2-a1) < a_threshold and abs(b2-b1) < b_threshold:
            eliminated[i] = True

    return lines[eliminated == False]


def intersections(a, b):
    """Calculate intersection between two lines in cartesian form
    """
    x1, y1,  x2,  y2 = a[0], a[1], a[2], a[3]
    x3, y3, x4, y4 = b[0], b[1], b[2], b[3]
    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    if d:
        d = float(d)
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) -
             (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) -
             (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return int(x), int(y)
    else:
        return -1, -1


def bounding_rect(lines):
    # TODO: Find best 4 lines to perform
    # perspective transform.
    return random.choices(lines, k=4)
