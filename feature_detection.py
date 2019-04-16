import cv2
import numpy as np
import viz_utils
import io_utils
import math_utils
from itertools import combinations


def parametricIntersect(r1, t1, r2, t2):
    """Calculate intersection between two lines in rho theta form
    """
    ct1 = np.cos(t1)     # matrix element a
    st1 = np.sin(t1)     # b
    ct2 = np.cos(t2)     # c
    st2 = np.sin(t2)     # d
    d = ct1*st2-st1*ct2        # determinative (rearranged matrix for inverse)
    if d != 0.0:
        x = int((st2*r1-st1*r2)/d)
        y = int((-ct2*r1+ct1*r2)/d)
        return((x, y))
    else:  # lines are parallel and will NEVER intersect!
        return(None)


def detect_lines_old(img):
    # img = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))  NO SECOND RESIZE
    viz_utils.imshow(img, resize=True)
    canny = cv2.Canny(img, 200, 255)
    viz_utils.imshow(canny, resize=True)
    # STILL NOT REALLY WORKING
    lines = cv2.HoughLines(canny, 10, (np.pi / 2), 15)

    intersections = []
    points = []  # TEST
    if len(lines) < 2:
        print("Less then two lines found")
    else:
        for line1 in range(0, len(lines)-1):
            for line2 in range(line1+1, len(lines)):
                point = parametricIntersect(
                    lines[line1][0][0], lines[line1][0][1], lines[line2][0][0], lines[line2][0][1])
                if point is not None:
                    intersections.append([line1, line2, point[0], point[1]])
                    points.append([point])

    testLines = viz_utils.drawLines(img, lines)
    testPoints = viz_utils.overlay_points(testLines, points)

    viz_utils.imshow(testPoints, resize=True)

    return lines


def detect_lines(img):
    """ Detect lines using HoughLinesP
    """
    canny = cv2.Canny(img, 200, 255)
    viz_utils.imshow(canny)
    lines = cv2.HoughLinesP(canny, 1, (np.pi / 180), 128,
                            minLineLength=10, maxLineGap=50)
    return np.reshape(lines, (-1, 4))


def detect_contours(img):
    # img = cv2.resize(img, (int(img.shape[1] / 10), int(img.shape[0] / 10)))
    viz_utils.imshow(img, resize=False)
    canny = cv2.Canny(img, 200, 255)
    viz_utils.imshow(canny, resize=False)
    contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
    viz_utils.imshow(img, resize=False)
    return img


def detect_corners(img, maxCorners=0):
    return cv2.goodFeaturesToTrack(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        maxCorners, .3, 15, useHarrisDetector=True)


if __name__ == "__main__":
    # img = cv2.imread('images/single_paintings/still_life/20190217_101231.jpg')
    # corners = detect_corners(img)
    # testCorners = viz_utils.overlay_points(img, corners)
    # viz_utils.imshow(testCorners, resize=True)

    for path, img in io_utils.imread_folder('images/query_paintings_20'):
        viz_utils.imshow(img)
        # contour_img = detect_contours(img)
        lines = detect_lines(img)
        lines = math_utils.eliminate_duplicates(lines, 5)
        print(lines)
        lines = math_utils.bounding_rect(lines)
        viz_utils.overlay_lines_cartesian(img, lines)

        points = []
        for line1, line2 in combinations(lines, 2):
            points.append(math_utils.intersections(line1, line2))

        viz_utils.overlay_points(img, points)
        viz_utils.imshow(img, name=path)
