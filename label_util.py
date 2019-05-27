import math
import perspective
import cv2
import numpy as np
import math_utils

def angle_betw_lines (line1, line2):
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
    while phi<0:
        phi += math.pi
    return phi


def pnt_above_line(pt, ln):
    # a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0
    if ((ln[0][1] - ln[1][1]) * pt[1] + (ln[0][0] - ln[1][0]) * pt[0] +
       ((ln[0][0] * ln[1][1]) - (ln[1][0] * ln[0][1]))) > 0:
        print("above")
        return True
    print("below")
    return False


def pnt_below_line(pt, ln):
    # a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0
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
    if pt[0]<max_x1 and pt[0]>min_x1 and pt[1]<max_y1 and pt[1]>min_y1:
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
    if max_x1<min_x2 or max_x2<min_x1 or max_y1<min_y2 or max_y2<min_y1:
        return True
    return False


def draw_quad(pts, image, col):
    cv2.line(image, tuple(pts[0]), tuple(pts[1]), col)
    cv2.line(image, tuple(pts[1]), tuple(pts[2]), col)
    cv2.line(image, tuple(pts[2]), tuple(pts[3]), col)
    cv2.line(image, tuple(pts[3]), tuple(pts[0]), col)


def euclidian_dist(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_area (pts):
    pts = perspective.order_points(pts)
    a = euclidian_dist(pts[2], pts[3])
    b = euclidian_dist(pts[3], pts[0])
    c = euclidian_dist(pts[0], pts[1])
    d = euclidian_dist(pts[1], pts[2])
    t = 0.5 * (a + b + c + d)
    angle1 = angle_betw_lines(np.array([pts[2], pts[3]]), np.array([pts[3], pts[0]]))
    angle2 = angle_betw_lines(np.array([pts[0], pts[1]]), np.array([pts[1], pts[2]]))
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
            pta = math_utils.intersections(np.array([pts1[0][0], pts1[0][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]]))
            ptb = math_utils.intersections(np.array([pts1[0][0], pts1[0][1], pts1[1][0], pts1[1][1]]),
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
            pta = math_utils.intersections(np.array([pts1[1][0], pts1[1][1], pts1[2][0], pts1[2][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]]))
            ptb = math_utils.intersections(np.array([pts1[0][0], pts1[0][1], pts1[1][0], pts1[1][1]]),
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
            pta = math_utils.intersections(np.array([pts1[1][0], pts1[1][1], pts1[2][0], pts1[2][1]]),
                                           np.array([pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]]))
            ptb = math_utils.intersections(np.array([pts1[2][0], pts1[2][1], pts1[3][0], pts1[3][1]]),
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
            pta = math_utils.intersections(np.array([pts1[3][0], pts1[3][1], pts1[0][0], pts1[0][1]]),
                                           np.array([pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]]))
            ptb = math_utils.intersections(np.array([pts1[2][0], pts1[2][1], pts1[3][0], pts1[3][1]]),
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


def calculate_union (pts1, pts2):
    union = calculate_area(pts1) + calculate_area(pts2) - calculate_intersection(pts1, pts2)
    return union