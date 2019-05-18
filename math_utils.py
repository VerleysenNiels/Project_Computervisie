import logging
import os
import random
import itertools
import math
import perspective
import cv2
import numpy as np
from sklearn.metrics import f1_score


def rolling_avg(paths):
    if paths and len(paths) != 0: 
        samples = list(map(lambda x: os.path.basename(os.path.dirname(x)), paths))
        return max(set(samples), key=samples.count)
    return ''


def mean_difference(points, img):
    mask_1 = create_mask(points, img, False)
    mask_2 = create_mask(points, img, True)
    mean_1 = cv2.mean(img, mask_1)
    mean_2 = cv2.mean(img, mask_2)
    diff = (mean_1[0] - mean_2[0])**2
    + (mean_1[1] - mean_2[1])**2
    + (mean_1[2] - mean_2[2])**2
    return diff

def precision(points, img):
    mask = create_mask(points, img, False)
    return f1_score(mask.flatten()//255, mask.flatten()//255)  

def to_polar(line):
    '''Convert lines in Cartesian system to lines in polar system.
    '''
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]

    dx = x2-x1
    dy = y2-y1

    rho = np.abs(x2*y1 - y2*x1) / np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx) + np.pi / 2

    return rho, theta


def to_cartesian(line):
    '''Convert lines in polar system to lines in Cartesian system.
    '''
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


def euclid_dist(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def approx_dist(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def eliminate_duplicates(lines, rho_threshold=10, theta_threshold=.1):
    '''Remove similar lines, based on a threshold
    '''
    eliminated = np.zeros(len(lines), dtype=bool)

    for i, j in itertools.combinations(range(len(lines)), 2):
        if eliminated[i] or eliminated[j]:
            continue

        r1, theta1 = to_polar(lines[i])
        r2, theta2 = to_polar(lines[j])

        if abs(r1 - r2) < rho_threshold and min(abs(theta1 - theta2), abs(theta2 - theta1)) < theta_threshold:
            eliminated[i] = True

    return lines[eliminated == False]


def intersections_polar(a, b):
    '''Calculate intersection (cartesian) between two lines in polar form
    '''
    return intersections(to_cartesian(a), to_cartesian(b))


def intersections(a, b):
    '''Calculate intersection between two lines in cartesian form
    '''
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


def out_of_ratio(width, height, ratio):
    return width > 0 \
        and height > 0  \
        and (width/height < ratio or height/width < ratio)


def bounding_rect(lines, hparams, theta_threshold=.1):
    '''Pick 4 lines which are most likely to be the edges of the painting,
    used for perspective correction
    '''
    ratio = hparams['ratio']
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

    # Sort the lines by rho-difference
    parallel[best].sort(key=lambda x: x[1], reverse=True)
    perpendicular[best].sort(key=lambda x: x[1], reverse=True)

    if len(parallel[best]) < 1 or len(perpendicular[best]) < 2:
        logging.warning(
            'Perspective transform: Not enough parallel/perpendicular lines found.')
        return []

    # Initialize indices of the bounding rectangle
    par = 0     # highest Δrho
    perp1 = 0   # highest Δrho
    perp2 = -1  # lowest Δrho

    # While the ratio of the bounding rectangle is not realistic for a
    # painting, try to decrease te size and get a better ratio
    good_ratio = False
    while not good_ratio:
        # Get 3 corners of the rectangle
        p1 = intersections(
            lines[best], lines[perpendicular[best][perp1][0]])
        p2 = intersections(
            lines[best], lines[perpendicular[best][perp2][0]])
        p3 = intersections(
            lines[parallel[best][par][0]], lines[perpendicular[best][perp1][0]])

        # Not really width and height, it can be the other way around as well
        width = euclid_dist(p1, p2)
        height = euclid_dist(p1, p3)

        if out_of_ratio(width, height, ratio):
            # Bad ratio
            if width > height:
                # Look for maximum distance to remove (perp1 or perp2)
                # Calculate new intersections of best with perp1 + 1 and with perp2 -1 and the corresponding distances

                p = intersections(
                    lines[best], lines[perpendicular[best][perp1+1][0]])
                dist_perp1 = euclid_dist(p2, p)

                p = intersections(
                    lines[best], lines[perpendicular[best][perp2-1][0]])
                dist_perp2 = euclid_dist(p1, p)

                # change index with biggest distance from previous line
                if dist_perp1 > dist_perp2 and perp1 + 1 < len(perpendicular[best]) + perp2:
                    perp1 += 1
                elif len(perpendicular[best]) - 1 + perp2 > perp1 + 1:
                    perp2 -= 1
                else:
                    # It is not possible to make the bounding rectangle smaller
                    good_ratio = True
            elif par + 1 < len(parallel[best]) - 1:
                par += 1
            else:
                # It is impossible to make the bounding rectangle smaller
                good_ratio = True
        else:
            good_ratio = True

    # Calculate final intersections
    p1 = intersections(
        lines[best], lines[perpendicular[best][perp1][0]])
    p2 = intersections(
        lines[best], lines[perpendicular[best][perp2][0]])
    p3 = intersections(
        lines[parallel[best][par][0]], lines[perpendicular[best][perp1][0]])

    width = euclid_dist(p1, p2)
    height = euclid_dist(p1, p3)

    # Check if the ratio is now good
    if out_of_ratio(width, height, ratio):
        # Ratio is still bad, so there is no good bounding rectangle
        logging.warning('Perspective transform: Bad aspect ratio (%f) ',
                        min(width / height, height/width))
        return []
    else:
        # The ratio is good, return the bounding rectangle
        l1 = lines[best]
        l2 = lines[parallel[best][par][0]]
        l3 = lines[perpendicular[best][perp1][0]]
        l4 = lines[perpendicular[best][perp2][0]]
        return np.array([l1, l2, l3, l4])


def bounding_rect_2(lines, estimate, theta_threshold=.1):
    if len(lines) < 4:
        logging.warning(
            'Perspective transform: Not enough lines found (%d).', len(lines))
        return []

    max_diff = 0
    max = np.array([0, 1, 2, 3])
    HALF_PI = np.pi / 2

    parallel = [[] for _ in range(len(lines))]
    perpendicular = [[] for _ in range(len(lines))]
    logging.debug('Calculating parallel/perpendiculars...')
    for i, j in itertools.combinations(range(len(lines)), 2):
        r1, theta1 = to_polar(lines[i])
        r2, theta2 = to_polar(lines[j])

        angle_diff = min(abs(theta1 - theta2), abs(theta2 - theta1))
        if angle_diff < theta_threshold:
            parallel[i].append(j)
        elif HALF_PI - theta_threshold < angle_diff < HALF_PI + theta_threshold:
            perpendicular[i].append(j)


    logging.debug('Calculating best score...')
    for i in range(len(lines)):
        for k in range(len(parallel[i])):
            for j, l in itertools.combinations(range(len(perpendicular[i])), 2):
                l1 = lines[i]
                l2 = lines[perpendicular[i][j]]
                l3 = lines[parallel[i][k]]
                l4 = lines[perpendicular[i][l]]

                p1 = intersections(l1, l2)
                p2 = intersections(l2, l3)
                p3 = intersections(l3, l4)
                p4 = intersections(l4, l1)
                points = np.array([p1, p2, p3, p4])

                mean_diff = mean_difference(points, estimate)
                # mean_diff = precision(points, estimate)
                if mean_diff > max_diff:
                    max_diff = mean_diff
                    max = np.array([l1, l3, l2, l4])
    return max


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


def create_mask(points, img, invert=True):
    '''Create a binary mask 
    '''
    points = order_points(points).astype('int32')
    pts = points.reshape((-1, 1, 2))
    if not invert:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
    else:
        mask = np.full((img.shape[0], img.shape[1]), 255, dtype='uint8')
        mask = cv2.fillPoly(mask, [pts], (0, 0, 0))
    return mask
