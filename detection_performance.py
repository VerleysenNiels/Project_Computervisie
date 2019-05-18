import csv
import logging
import os

import cv2
import numpy as np

import feature_detection
import io_utils
import label_util


class IoU():
    '''Intersection-over-Union accuracy calculator
    '''

    def __init__(self, images_dir):
        """        
        Arguments:
            images_dir {str} -- Path to original images
        """
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
                corners_exp = np.array([
                    (int(line[1]), int(line[2])),
                    (int(line[3]), int(line[4])),
                    (int(line[5]), int(line[6])),
                    (int(line[7]), int(line[8]))])
                path = os.path.join(self.images_dir, line[0])
                img = io_utils.imread(path)
                corners_pred, _ = feature_detection.detect_perspective(img)

                iou = self.compute(corners_pred, corners_exp)
                logging.info('%s: %.2f%%', line[0], 100 * iou)
                if log_file_path:
                    output_log_file.write('%s;%.4f\n' % (line[0], iou))
                sum_iou += iou
                amount += 1

        avg_iou = sum_iou / amount
        logging.info('Average IoU: %.2f%%', 100 * avg_iou)
        return avg_iou

    def compute(self, corners_pred, corners_exp):
        '''Compute the IoU given an array of predicted corners and an array
           of expected corners (ground truth)
        '''
        if len(corners_pred) >= 4:
            inters = label_util.calculate_intersection(
                corners_exp, corners_pred)
            union = label_util.calculate_union(corners_exp, corners_pred)
            if union != 0:
                return inters / union
        return 0
