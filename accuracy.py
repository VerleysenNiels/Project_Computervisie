import csv
import logging
import os

import cv2
import numpy as np
from shapely.geometry import Polygon

import feature_detection
import io_utils
import label_util
import viz_utils


class IoU():
    '''Intersection-over-Union accuracy calculator
    '''

    def __init__(self, images_dir, hparams):
        """        
        Arguments:
            images_dir {str} -- Path to original images
        """
        self.hparams = hparams
        self.images_dir = images_dir

    def compute_all(self, csv_path, log_file_path=None):
        """Compute all and avg IoU (Intersection-over-Union) given a ground
           truth

        Arguments:
            csv_path {str} -- Path to ground truth csv
            log_file_path {str} -- Path to write results to (default: {None})

        Returns:
            float -- average IoU
        """
        sum_iou = 0
        amount = 0

        with open(csv_path) as ground_truth_f:
            if log_file_path:
                output_log_file = open(log_file_path, 'w+')
            ground_truth = csv.reader(ground_truth_f, delimiter=';')
            for line in ground_truth:
                corners_exp = np.int32([
                    (line[1], line[2]),
                    (line[3], line[4]),
                    (line[5], line[6]),
                    (line[7], line[8])])
                path = os.path.join(self.images_dir, line[0])
                img = io_utils.imread(path)
                corners_pred, img = feature_detection.detect_perspective(
                    img, self.hparams)

                if logging.root.level == logging.DEBUG:
                    img = viz_utils.overlay_polygon(
                        img, corners_exp, color=(0, 255, 255))
                    img = viz_utils.overlay_polygon(
                        img, corners_pred)
                    viz_utils.imshow(img)

                iou = self.compute_2(corners_pred, corners_exp)
                logging.info('%s: %.2f%%', line[0], 100 * iou)
                if log_file_path:
                    output_log_file.write('%s;%.4f\n' % (line[0], iou))
                sum_iou += iou
                amount += 1

        avg_iou = sum_iou / amount
        logging.info('Average IoU: %.2f%%', 100 * avg_iou)
        if log_file_path:
            output_log_file.write('Average;%.4f\n' % avg_iou)
        return avg_iou

    def compute(self, corners_pred, corners_exp):
        '''Compute the IoU given an array of predicted corners and an array
           of expected corners (ground truth)
        '''
        if len(corners_pred) == 4:
            inters = label_util.calculate_intersection(
                corners_exp, corners_pred)
            union = label_util.calculate_union(corners_exp, corners_pred)
            if union != 0:
                return min(inters / union, 1.00)
        return 0.00

    def compute_2(self, corners_pred, corners_exp):
        '''Compute the IoU given an array of predicted corners and an array
           of expected corners (ground truth).
           This function uses the shapely package
        '''
        if len(corners_pred) == 4:
            a = Polygon(corners_pred)
            b = Polygon(corners_exp)
            intersections = a.intersection(b)
            union = a.union(b)
            union_area = a.union(b).area
            if union_area != 0:
                return intersections.area / union_area

        return 0.00
