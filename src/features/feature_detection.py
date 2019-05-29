import logging
from itertools import combinations

import cv2
import numpy as np
from skimage import feature

import src.utils.io as io
import src.utils.math as math
import src.utils.viz as viz
from src.utils.perspective_transform import perspective_transform


def create_dog(dim, angle=0):
    center = (int(dim / 2), int(dim / 2))
    # Create 1D Gaussian kernel
    kernel = cv2.getGaussianKernel(dim, dim / 3)  # sigma relative to dim
    # Copy it to the middle of a square matrix
    kernel = np.pad(kernel, ((0, 0), center),
                    'constant', constant_values=(0, 0))
    # Create 1D row Gaussian kernel
    kernel2 = np.transpose(cv2.getGaussianKernel(dim, dim / 25))
    # Filter square with row kernel
    kernel = cv2.filter2D(kernel, -1, kernel2)
    # Derive kernel with Sobel
    kernel = cv2.Sobel(kernel, cv2.CV_64F, 1, 0)
    # Obtain rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    # Rotate kernel with matrix
    kernel = cv2.warpAffine(kernel, rot_mat, (dim, dim))
    return kernel


def filter_dog(img, hparams):
    kernel1 = create_dog(hparams['dog_dimension'], 0)
    kernel2 = create_dog(hparams['dog_dimension'], 90)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.abs(cv2.filter2D(bw, cv2.CV_32F, kernel1))
    out += np.abs(cv2.filter2D(bw, cv2.CV_32F, kernel2))
    if logging.root.level <= logging.DEBUG:
        viz.imshow(out, norm=True)
    _, out = cv2.threshold(
        out, hparams['dog_threshold'], 255, cv2.THRESH_BINARY)
    if hparams['dog_erode']:
        out = cv2.erode(
            out,
            cv2.getStructuringElement(cv2.MORPH_RECT, (hparams['dog_erode'], hparams['dog_erode'])))

    return out.astype('uint8')


def detect_lines(img, hparams):
    """ Detect lines using HoughLinesP
    """
    if hparams['remove_hblur']:
        img = cv2.erode(
            img,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
            iterations=2)

    # contours = local_binary_pattern(img)
    if hparams['edge_detection'] == 'dog':
        canny = filter_dog(img, hparams)
    else:
        canny = cv2.Canny(
            img,
            2 * hparams['canny_threshold'] // 3,
            hparams['canny_threshold'])

    if logging.root.level == logging.DEBUG:
        viz.imshow(canny, 'Edge detection', resize=True, norm=True)
        # viz.imshow(contours, norm=True)

    if hparams['remove_hblur']:
        canny = cv2.dilate(
            canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

    lines = cv2.HoughLinesP(
        canny, hparams['hough_resolution_rho'], hparams['hough_resolution_theta'] * (
            np.pi / 180),
        hparams['hough_threshold'],
        minLineLength=hparams['min_line_length'],
        maxLineGap=hparams['max_line_gap']
    )
    try:
        return np.reshape(lines, (-1, 4))
    except:
        return np.array([])


def detect_contours(img):
    """
    Detect contours using findContours
    """
    contours, hierarchy = cv2.findContours(
        cv2.blur(img, (11, 11)),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            filtered.append(cnt)
    filtered = np.array(filtered)
    out = np.zeros_like(img)
    out = cv2.drawContours(out, filtered, -1, (255, 255, 255), cv2.FILLED)
    return out


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
        img_harris = viz.overlay_points(np.copy(img), corners)
        viz.imshow(img_harris, resize=True)
    return corners


def detect_perspective(img, hparams):
    """Automatically detect perspective points, used in perspective
    transformation.

    Returns:
        An array of 4 points. If the perspective cannot be detected, the array
        is empty
    """
    blurred = cv2.medianBlur(img, hparams['pre_median'])
    all_lines = detect_lines(
        blurred,
        hparams)  # Detect all lines

    if logging.root.level <= logging.INFO:
        img = viz.overlay_lines_cartesian(img, all_lines)

    # points = math.bounding_rect_2(all_lines, hparams, img.shape)
    points = math.bounding_rect(all_lines, hparams)

    if len(points) < 4:
        return np.int32([]), img

    points = np.int32(points)  # Calculate intersection points

    if logging.root.level <= logging.INFO:
        img = viz.overlay_polygon(img, points, color=(255, 0, 0))

    return points, img


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


def dilate(img):
    img2 = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return img2


def local_binary_pattern(img):
    grayscale = equalize_histogram(img)
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)
    radius = 5
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(
        grayscale, n_points, radius, method='uniform')

    max = np.max(lbp)
    lbp = ((lbp <= .6 * max) * (lbp >= .4 * max)) * 255
    lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    lbp = cv2.medianBlur(lbp, 3)
    lbp = np.pad(lbp[10:img.shape[0] - 10, 10:img.shape[1] - 10],
                 (10, 10), 'constant', constant_values=(0, 0))  # Remove edge behaviour
    return lbp
