import logging
from itertools import combinations

import cv2
import numpy as np
from skimage import feature

import src.utils.io as io
import src.utils.math as math
import src.utils.viz as viz
from src.utils.perspective_transform import perspective_transform


def detect_perspective(img, hparams):
    """ Automatically detect perspective points, used in perspective
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


def create_dog(dim, angle=0):
    """ Create a Differential of Gaussian filter 

    Arguments:
        dim {int} -- Filter size
        angle {int} -- Filter angle (default: {0} = vertical)

    Returns:
        DoG filter
    """
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
    """ Detect edges in an image by filtering with a horizontal and vertical 
        DoG kernel, thresholding and eroding

    Arguments:
        img -- The input image. This image remains unchanged
        hparams {dict} -- The video or image hyperparameters subset

    Returns:
        A binary image containing edges
    """
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
    """ Detect lines using HoughLinesP """
    if hparams['remove_hblur']:
        img = cv2.erode(
            img,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
            iterations=2)

    if hparams['edge_detection'] == 'dog':
        canny = filter_dog(img, hparams)
    else:
        canny = cv2.Canny(
            img,
            2 * hparams['canny_threshold'] // 3,
            hparams['canny_threshold'])

    if logging.root.level == logging.DEBUG:
        viz.imshow(canny, 'Edge detection', resize=True, norm=True)

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


def dilate(img, size=3):
    """ Dilate an image """
    out = cv2.dilate(img, cv2.getStructuringElement(
        cv2.MORPH_RECT, (size, size)))
    return out
