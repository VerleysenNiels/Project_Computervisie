import itertools
import logging
import math
import os
import random

import cv2
import numpy as np
from sklearn.metrics import f1_score

import src.utils.perspective_transform as perspective


def rolling_avg(paths):
    if paths and len(paths) != 0:
        samples = list(
            map(lambda x: os.path.basename(os.path.dirname(x)), paths))
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


def angle_betw_lines(line1, line2):
    # equation as "y = ax + b"
    # a = (y2 - y1)/(x2 - x1)
    dx1 = (line1[1, 0] - line1[0, 0])
    dx2 = (line2[1, 0] - line2[0, 0])
    dy1 = (line1[1, 1] - line1[0, 1])
    dy2 = (line2[1, 1] - line2[0, 1])
    theta1 = math.pi/2
    theta2 = math.pi/2
    if dx1 != 0:
        theta1 = math.atan(dy1 / dx1)
    if dx2 != 0:
        theta2 = math.atan(dy2 / dx2)
    phi = theta1 - theta2
    while phi < 0:
        phi += math.pi
    return phi


def pnt_above_line(pt, ln):
    """a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0"""
    if ((ln[0][1] - ln[1][1]) * pt[1] + (ln[0][0] - ln[1][0]) * pt[0] +
            ((ln[0][0] * ln[1][1]) - (ln[1][0] * ln[0][1]))) > 0:
        print("above")
        return True
    print("below")
    return False


def pnt_below_line(pt, ln):
    """a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0"""
    if ((ln[0][1] - ln[1][1]) * pt[1] + (ln[0][0] - ln[1][0]) * pt[0] +
            ((ln[0][0] * ln[1][1]) - (ln[1][0] * ln[0][1]))) < 0:
        print("below")
        return True
    print("above")
    return False


def pnt_left_line(pt, ln):
    # a point (xp, yp) lies left of a line if ((x2-x1)*(yp-y1))-((y2-y1)*(xp-x1)) > 0
    if (((ln[1][0] - ln[0][0]) * (pt[1] - ln[0][1])) - ((ln[1][1] - ln[0][1]) * (pt[0] - ln[0][0]))) > 0:
        print("left")
        return True
    print("right")
    return False


def pnt_right_line(pt, ln):
    # a point (xp, yp) lies left of a line if ((x2-x1)*(yp-y1))-((y2-y1)*(xp-x1)) > 0
    if (((ln[1][0] - ln[0][0]) * (pt[1] - ln[0][1])) - ((ln[1][1] - ln[0][1]) * (pt[0] - ln[0][0]))) < 0:
        print("right")
        return True
    print("left")
    return False


def point_inside_quad(pt, quad):
    max_x1 = max(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
    min_x1 = min(quad[0][0], quad[1][0], quad[2][0], quad[3][0])
    max_y1 = max(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
    min_y1 = min(quad[0][1], quad[1][1], quad[2][1], quad[3][1])
    if pt[0] < max_x1 and pt[0] > min_x1 and pt[1] < max_y1 and pt[1] > min_y1:
        return True
    return False


def no_intersection(quad1, quad2):
    # still need to implement this
    #quad1 = perspective.order_points(quad1)
    #quad2 = perspective.order_points(quad2)
    max_x1 = max(quad1[0][0], quad1[1][0], quad1[2][0], quad1[3][0])
    min_x1 = min(quad1[0][0], quad1[1][0], quad1[2][0], quad1[3][0])
    max_y1 = max(quad1[0][1], quad1[1][1], quad1[2][1], quad1[3][1])
    min_y1 = min(quad1[0][1], quad1[1][1], quad1[2][1], quad1[3][1])
    max_x2 = max(quad2[0][0], quad2[1][0], quad2[2][0], quad2[3][0])
    min_x2 = min(quad2[0][0], quad2[1][0], quad2[2][0], quad2[3][0])
    max_y2 = max(quad2[0][1], quad2[1][1], quad2[2][1], quad2[3][1])
    min_y2 = min(quad2[0][1], quad2[1][1], quad2[2][1], quad2[3][1])
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return True
    return False


def draw_quad(pts, image, col):
    cv2.line(image, tuple(pts[0]), tuple(pts[1]), col)
    cv2.line(image, tuple(pts[1]), tuple(pts[2]), col)
    cv2.line(image, tuple(pts[2]), tuple(pts[3]), col)
    cv2.line(image, tuple(pts[3]), tuple(pts[0]), col)



def calculate_area(pts):
    pts = perspective.order_points(pts)
    a = euclid_dist(pts[2], pts[3])
    b = euclid_dist(pts[3], pts[0])
    c = euclid_dist(pts[0], pts[1])
    d = euclid_dist(pts[1], pts[2])
    t = 0.5 * (a + b + c + d)
    angle1 = angle_betw_lines(
        np.array([pts[2], pts[3]]), np.array([pts[3], pts[0]]))
    angle2 = angle_betw_lines(
        np.array([pts[0], pts[1]]), np.array([pts[1], pts[2]]))
    area = math.sqrt(((t - a) * (t - b) * (t - c) * (t - d))
                     - (a * b * c * d * ((math.cos((angle1 + angle2)/2)) ** 2)))
    return area


def get_intersection_pts(pts1, pts2):
    pts1 = perspective.order_points(pts1)
    pts2 = perspective.order_points(pts2)
    pts3 = pts1
    # first check if either corner lies completely inside of the other (that's the two first ifs
    # if not, check which lies most inside to get the inside intersection
    if not no_intersection(pts1, pts2):
        if point_inside_quad(pts1[0], pts2):
            pts3[0] = pts1[0]
        elif point_inside_quad(pts2[0], pts1):
            pts3[0] = pts2[0]
        else:
            pta = intersections(np.array([pts1[0][0], pts1[0][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]]))
            ptb = intersections(np.array([pts1[0][0], pts1[0][1], pts1[1][0], pts1[1][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[3][0], pts2[3][1]]))
            if pta[0] < 0 or pta[0] > 10000 or pta[1] < 0 or pta[1] > 10000:
                pts3[0] = ptb
            elif ptb[0] < 0 or ptb[0] > 10000 or ptb[1] < 0 or ptb[1] > 10000:
                pts3[0] = pta
            elif ptb[0] > pta[0]:
                pts3[0] = ptb
            else:
                pts3[0] = pta

        if point_inside_quad(pts1[1], pts2):
            pts3[1] = pts1[1]
        elif point_inside_quad(pts2[1], pts1):
            pts3[1] = pts2[1]
        else:
            pta = intersections(np.array([pts1[1][0], pts1[1][1], pts1[2][0], pts1[2][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]]))
            ptb = intersections(np.array([pts1[0][0], pts1[0][1], pts1[1][0], pts1[1][1]]),
                                           np.array([pts2[1][0], pts2[1][1], pts2[2][0], pts2[2][1]]))
            if pta[0] < 0 or pta[0] > 10000 or pta[1] < 0 or pta[1] > 10000:
                pts3[1] = ptb
            elif ptb[0] < 0 or ptb[0] > 10000 or ptb[1] < 0 or ptb[1] > 10000:
                pts3[1] = pta
            elif ptb[0] < pta[0]:
                pts3[1] = ptb
            else:
                pts3[1] = pta

        if point_inside_quad(pts1[2], pts2):
            pts3[2] = pts1[2]
        elif point_inside_quad(pts2[2], pts1):
            pts3[2] = pts2[2]
        else:
            pta = intersections(np.array([pts1[1][0], pts1[1][1], pts1[2][0], pts1[2][1]]),
                                           np.array([pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]]))
            ptb = intersections(np.array([pts1[2][0], pts1[2][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[1][0], pts2[1][1], pts2[2][0], pts2[2][1]]))
            if pta[0] < 0 or pta[0] > 10000 or pta[1] < 0 or pta[1] > 10000:
                pts3[2] = ptb
            elif ptb[0] < 0 or ptb[0] > 10000 or ptb[1] < 0 or ptb[1] > 10000:
                pts3[2] = pta
            elif ptb[0] < pta[0]:
                pts3[2] = ptb
            else:
                pts3[2] = pta

        if point_inside_quad(pts1[3], pts2):
            pts3[3] = pts1[3]
        elif point_inside_quad(pts2[3], pts1):
            pts3[3] = pts2[3]
        else:
            pta = intersections(np.array([pts1[3][0], pts1[3][1], pts1[0][0], pts1[0][1]]),
                                           np.array([pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]]))
            ptb = intersections(np.array([pts1[2][0], pts1[2][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[3][0], pts2[3][1]]))
            if pta[0] < 0 or pta[0] > 10000 or pta[1] < 0 or pta[1] > 10000:
                pts3[3] = ptb
            elif ptb[0] < 0 or ptb[0] > 10000 or ptb[1] < 0 or ptb[1] > 10000:
                pts3[3] = pta
            elif ptb[0] > pta[0]:
                pts3[3] = ptb
            else:
                pts3[3] = pta
        return pts3
    return np.array([(-1, -1), (-1, -1), (-1, -1), (-1, -1)])


def calculate_intersection(pts1, pts2):
    if not no_intersection(pts1, pts2):
        pts3 = get_intersection_pts(pts1, pts2)
        pts3 = perspective.order_points(pts3)
        intersection = calculate_area(pts3)
        return intersection
    return 0


def calculate_union(pts1, pts2):
    union = calculate_area(pts1) + calculate_area(pts2) - \
        calculate_intersection(pts1, pts2)
    return union
