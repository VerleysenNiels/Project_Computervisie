import numpy as np
from itertools import combinations
import random
import logging


def to_polar(line):
    """Convert lines in Cartesian system to lines in polar system.
    """
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]

    dx = x2-x1
    dy = y2-y1

    rho = np.abs(x2*y1 - y2*x1) / np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx) + np.pi / 2

    return rho, theta


def to_cartesian(line):
    """Convert lines in polar system to lines in Cartesian system.
    """
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


def eliminate_duplicates(lines, rho_threshold=10, theta_threshold=.1):
    """Remove similar lines, based on a threshold
    """
    eliminated = np.zeros(len(lines), dtype=bool)

    for i, j in combinations(range(len(lines)), 2):
        if eliminated[i] or eliminated[j]:
            continue

        r1, theta1 = to_polar(lines[i])
        r2, theta2 = to_polar(lines[j])

        if abs(r1 - r2) < rho_threshold and min(abs(theta1 - theta2), abs(theta2 - theta1)) < theta_threshold:
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
        return x, y
    else:
        return -1, -1


def bounding_rect(lines, corners, theta_threshold=.1):
    """Pick 4 lines which are most likely to be the edges of the painting,
    used for perspective correction
    """

    if len(lines) < 4:
        logging.warning(
            'Perspective transform: Not enough lines found (%d).', len(lines))
        return []

    straight = np.pi / 2
    parallel = []
    perpendicular = []
    best = 0
    # Foreach line, find +/- parallel and +/- perpendicular lines
    for i in range(len(lines)):
        parallel.append([])
        perpendicular.append([])

        r1, theta1 = to_polar(lines[i])

        for j in range(len(lines)):
            r2, theta2 = to_polar(lines[j])

            if i != j:
                angle_diff = min(abs(theta1 - theta2), abs(theta2 - theta1))
                if angle_diff < theta_threshold:
                    parallel[i].append((j, abs(r1-r2)))
                elif straight - theta_threshold < angle_diff < straight + theta_threshold:
                    perpendicular[i].append((j, r2))

        if len(perpendicular[i]) + len(parallel[i]) > len(perpendicular[best]) + len(parallel[best]) \
                and len(perpendicular[i]) >= 2 \
                and len(parallel[i]) >= 1:
            best = i

    parallel[best].sort(key=lambda x: x[1], reverse=True)
    perpendicular[best].sort(key=lambda x: x[1], reverse=True)

    if len(parallel[best]) < 1 or len(perpendicular[best]) < 2:
        logging.warning(
            'Perspective transform: not enough parallel/perpendicular lines found.')
        return []

    # TODO: integrate `corners` in voting system somehow
    l1 = lines[best]
    l2 = lines[parallel[best][0][0]]
    l3 = lines[perpendicular[best][0][0]]
    l4 = lines[perpendicular[best][-1][0]]
    return np.array([l1, l2, l3, l4])
