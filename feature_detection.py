import cv2
import numpy as np
import viz_utils
import io_utils
import math_utils
from perspective import perspective_transform
from itertools import combinations
import logging


def detect_lines(img, threshold=128, canny_treshold=150):
    """ Detect lines using HoughLinesP
    """
    canny = cv2.Canny(img, 2 * canny_treshold // 3, canny_treshold)
    lines = cv2.HoughLinesP(
        canny, 1, (np.pi / 180), threshold, minLineLength=50, maxLineGap=150
    )
    try:
        return np.reshape(lines, (-1, 4))
    except:
        return np.array([])


def detect_contours(img):
    """
    NOT USED
    Detect contours using findContours
    """
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, cnt, -1, (0, 0, 0), 5)
    viz_utils.imshow(img, resize=False)
    return img


def detect_corners(img):
    source = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    dst = cv2.cornerHarris(
        np.float32(source), 10, 3, 0.04
    )
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(
        source,
        np.float32(centroids),
        (5, 5),
        (-1, -1),
        criteria
    )
    if logging.root.level == logging.DEBUG:
        img_harris = viz_utils.overlay_points(np.copy(img), corners)
        viz_utils.imshow(img_harris, resize=True)
    return corners


def detect_perspective(img):
    """Automatically detect perspective points, used in perspective
    transformation.

    Returns:
        An array of 4 points. If the perspective cannot be detected, the array
        is empty
    """

    all_lines = detect_lines(cv2.medianBlur(img, 15))  # Detect all lines
    # all_corners = detect_corners(img)
    lines = math_utils.bounding_rect(
        all_lines, None)  # Pick best 4 lines

    if len(lines) < 4:
        return []

    points = np.int32(
        [
            math_utils.intersections(lines[0], lines[2]),
            math_utils.intersections(lines[1], lines[2]),
            math_utils.intersections(lines[1], lines[3]),
            math_utils.intersections(lines[0], lines[3]),
        ]
    )  # Calculate intersection points

    if logging.root.level == logging.DEBUG:
        img_lines = viz_utils.overlay_lines_cartesian(img, all_lines)
        pts = points.reshape((-1, 1, 2))
        img_lines = cv2.polylines(img_lines, [pts], True, (255, 0, 0), 2)
        img_lines = viz_utils.overlay_points(img_lines, points)
        viz_utils.imshow(img_lines, resize=True)

    return points


def equalize_histogram(img):
    """Equalize the histogram of an image. This often gives better results
    since the pixel data is more spread out.
    """

    # Convert BGR image to YUV (luminance - color - color)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # Equalize luminance channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # Convert back to BGR
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
