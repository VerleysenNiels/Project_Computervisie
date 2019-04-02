import cv2
import numpy as np
import utils

# Calculate intersection between two lines
def parametricIntersect(r1, t1, r2, t2):
    ct1 = np.cos(t1)     # matrix element a
    st1 = np.sin(t1)     # b
    ct2 = np.cos(t2)     # c
    st2 = np.sin(t2)     # d
    d = ct1*st2-st1*ct2        # determinative (rearranged matrix for inverse)
    if d != 0.0:
        x = int((st2*r1-st1*r2)/d)
        y = int((-ct2*r1+ct1*r2)/d)
        return((x, y))
    else: # lines are parallel and will NEVER intersect!
        return(None)

# Detect lines in grayscale image
def detect_lines(img):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 200)

    intersections = []

    for line1 in range(0, len(lines)-1):
        for line2 in range(line1+1, len(lines)):
            point = parametricIntersect(lines[line1][0], lines[line1][1], lines[line2][0], lines[line2][1])
            if point is not None:
                intersections.append([line1, line2, point[0], point[1]])

def detect_corners(img, maxCorners=0):
    return cv2.goodFeaturesToTrack(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        maxCorners, .3, 15, useHarrisDetector=True)


if __name__ == "__main__":
    img = cv2.imread('images/single_paintings/still_life/20190217_101231.jpg')
    corners = detect_corners(img)
    img = utils.overlay_points(img, corners)
    utils.imshow(img, resize=True)

