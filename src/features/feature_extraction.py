import logging
import math as pymath

import cv2
import numpy as np

import src.utils.io as io
import src.utils.math as math
import src.utils.viz as viz


class FeatureExtraction(object):

    def extract_keypoints(self, img, hparams):
        """ Extract descriptors of an image using ORB, SURF or SIFT (depending
            on the hparams)

        Returns:
            Array of descriptors
        """
        params = hparams['feature_matching']
        if params['type'] == 'ORB':
            detector = cv2.ORB_create(nfeatures=params['keypoint_thresh'])
        elif params['type'] == 'SURF':
            detector = cv2.xfeatures2d_SURF.create(
                hessianThreshold=params['keypoint_thresh'])
        elif params['type'] == 'SIFT':
            detector = cv2.xfeatures2d_SIFT.create(
                nfeatures=params['keypoint_thresh'])
        else:
            logging.critical(
                'Unknown feature_matching.type: %s', params['type'])
            exit(1)
        keypoints, descriptors = detector.detectAndCompute(img, None)
        return descriptors

    def match_keypoints(self, descriptors_1, descriptors_2, hparams):
        """ Calculate the match score between two sets of descriptors using BF 
            or FLANN.

        Returns:
            A match score between 0 (no match) and 100,000 (perfect match)
         """
        params = hparams['feature_matching']
        if params['type'] == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif params['type'] in ['SIFT', 'SURF']:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)   # or pass empty dictionary
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            logging.critical(
                'Unknown feature_matching.type: %s', params['type'])
            exit(1)

        matches = matcher.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        n = params['matches_amount']
        good_matches = matches[n:]

        if len(good_matches) == 0:
            return 0
        score = 0
        for m in good_matches:
            score += m.distance
        return len(good_matches) / (score + .00001)

    def extract_hist(self, img):
        """ Extract a normalized histogram of `img` . """
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    def compare_hist(self, hist1, hist2):
        """ Compare two normalized histograms.
        Returns:
            A similarity score between 0 (no similarity) and 1 (perfect similarity)
         """
        CV_COMP_INTERSECT = 2
        score = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CORREL)
        return (1+score)/2
