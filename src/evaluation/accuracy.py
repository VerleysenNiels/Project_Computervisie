import csv
import logging
import os

import cv2
import numpy as np

import src.features.feature_detection as feature_detection
import src.utils.io as io
import src.utils.math as math
import src.utils.viz as viz


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
                img = io.imread(path)
                corners_pred, img = feature_detection.detect_perspective(
                    img, self.hparams)
                if logging.root.level <= logging.INFO:
                    img = viz.overlay_polygon(
                        img, corners_exp, color=(0, 0, 255))
                    img = viz.overlay_polygon(
                        img, corners_pred)
                    viz.imshow(img)

                iou = self.compute(corners_pred, corners_exp)
                logging.info('%s: %.2f%%', line[0], 100 * iou)
                if log_file_path:
                    output_log_file.write('%s;%.4f\n' % (line[0], iou))
                    output_log_file.flush()
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
            inters = math.calculate_intersection(
                corners_exp, corners_pred)
            union = math.calculate_union(corners_exp, corners_pred)
            if union != 0:
                return min(inters / union, 1.00)
        return 0.00
