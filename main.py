import argparse
import json
import logging
import math as pymath
import os
import pickle
import platform
import shutil
import sys
import time

import cv2
import numpy as np

import src.features.feature_detection as feature_detection
import src.utils.io as io
import src.utils.math as math
import src.utils.perspective_transform as perspective
import src.utils.viz as viz
from src.evaluation.accuracy import IoU
from src.evaluation.video_ground_truth import VideoGroundTruth
from src.features.feature_extraction import FeatureExtraction
from src.inference.infer import infer, infer_frame
from src.inference.room_graph import RoomGraph


class PaintingClassifier(object):
    def __init__(self):
        description = R'''
        This tool helps you detect the hall you are in at the MSK in Ghent.
        The tool requires an image folder with paintings. The name of the 
        parent directory is the hall where the painting is located. 
        '''
        self.check_versions()
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(
            'command',
            choices=['build', 'eval_corners', 'eval_hall', 'infer'],
            help='Subcommand to run. See main.py SUBCOMMAND -h for more info.'
        )

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def check_versions(self):
        '''Check python package versions
        '''
        assert cv2.__version__.startswith('3.')

    def build(self):
        description = R'''
        Description:
            Build painting database from raw images directory.
        Example:
            python main.py build .\images\zalen\ -v -v
        '''
        parser = self._build_parser(description)
        parser.add_argument('directory')
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)
        self.hparams = json.load(open(args.config))
        for file in ['descriptors.pickle', 'histograms.pickle']:
            if os.path.isfile(file):
                logging.info('Removing old %s', file)
                os.remove(file)

        for path, img in io.imread_folder(args.directory):
            # img = feature_detection.equalize_histogram(img)
            # img = feature_detection.dilate(img)
            points, img_out = feature_detection.detect_perspective(
                img, self.hparams['image'])
            if logging.root.level == logging.DEBUG:
                viz.imshow(img_out, resize=True)
            img = perspective.perspective_transform(img, points)

            # Write to DB folder
            label = os.path.basename(os.path.dirname(path))
            out_path = os.path.join(
                './db/images', label.lower(), os.path.basename(path))
            logging.info('Writing to ' + out_path)
            io.imwrite(out_path, img)

    def eval_corners(self):
        description = R'''
        Description: 
            Evaluate the classifier on prelabeled data
        Example:
            python main.py eval_corners .\images\zalen\ .\csv_corners\all.csv -o .\csv_detection_perf\perf_all.csv -v -v
        '''
        parser = self._build_parser(description)
        parser.add_argument(
            'image_dir',
            help='Path to original images dir')
        parser.add_argument(
            'ground_truth',
            help='Path to csv with corner labels')
        parser.add_argument(
            '-o', '--output_dir',
            default='results/latest',
            help='Path to output directory')
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)
        self.hparams = json.load(open(args.config))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        shutil.copy2(args.config, os.path.join(
            args.output_dir, 'hparams.json'))
        iou = IoU(args.image_dir, self.hparams['image'])
        avg_iou = iou.compute_all(args.ground_truth, os.path.join(
            args.output_dir, 'accuracy-corners.csv'))

    def eval_hall(self):
        description = R'''
        Description: 
            Evaluate the classifier on prelabeled data
        Example:
            python main.py eval_hall .\video-frames -v -v
        '''
        parser = self._build_parser(description)
        parser.add_argument(
            'ground_truth_dir',
            help='Path to original images dir')
        parser.add_argument(
            '-o', '--output_dir',
            default='results/latest',
            help='Path to output directory')
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)
        self.hparams = json.load(open(args.config))
        # Copy hparams to output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        shutil.copy2(args.config, os.path.join(
            args.output_dir, 'hparams.json'))

        correct = 0
        total = 0
        total_time = 0
        descriptors, histograms = self._get_descriptors_histograms()
        file = open(os.path.join(
            args.output_dir, 'accuracy-hall.csv'), 'w+')
        file.write('Path;Expected;Predicted\n')
        extraction = FeatureExtraction()
        hparams = self.hparams
        for path, img in io.imread_folder(args.ground_truth_dir, resize=False):
            start_time = time.time()
            best = None
            best_score = 0
            descriptor = extraction.extract_keypoints(img, hparams)

            logits_descriptor = []
            labels = []
            for pathd in descriptors:
                if descriptors[pathd] is not None:
                    score_key = extraction.match_keypoints(
                        descriptor, descriptors[pathd], hparams)
                    logits_descriptor.append(score_key)
                    labels.append(pathd)

            scores = logits_descriptor
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best = labels[best_idx]
            end_time = time.time()
            total_time += (end_time - start_time)
            total += 1
            logging.info('Average matching speed: %.2fs per frame',
                         total_time/total)

            if best:
                predicted = os.path.basename(os.path.dirname(best))
            else:
                predicted = 'None'
            expected = os.path.basename(os.path.dirname(path))
            logging.debug('Expected:  %s', expected)
            logging.debug('Predicted: %s', predicted)
            file.write(path + ';' + expected + ';' + predicted + '\n')
            if expected == predicted:
                correct += 1
            logging.info('%d/%d (%.2f%%) correct', correct,
                         total, 100 * correct / total)
            viz.imshow(img, name='Frame')
            cv2.waitKey(1000)

    def infer(self):
        description = R'''
        Example:
            python main.py infer .\videos\MSK_01.mp4 -v -v

        Possible parameters for finetuning:
            border for too blurry images
            amount of previous labels used to determine the average label
            amount of transitions the algorithm wants to change room, but is not allowed (prevent being stuck in wrong room)
        '''
        parser = self._build_parser(description)
        parser.add_argument(
            'file',
            help='Video file to infer the hall ID from.',
            type=argparse.FileType('r'))
        parser.add_argument(
            '-m', '--measure', dest='ground_truth',
            help='Passes a file with the ground truth for the video to measure the accuracy')
        parser.add_argument(
            '-r', '--rooms', dest='room_file', default='./ground_truth/floor_plan/msk.csv',
            help='Passes a file with the rooms of the museum and how they are connected')
        parser.add_argument(
            '--gopro', dest='gopro_mode', action='count', default=0,
            help="If passed as an argument, then the infer function will treat the video file as a gopro video"
        )
        parser.add_argument(
            '-s', '--silent',
            default=False,
            action='store_true',
            help='Do not show any images, only output to commandline')
        parser.add_argument(
            '-p', '--map', dest='map',
            default='./ground_truth/floor_plan/msk.jpg',
            help='Passes an image that represents the map of the building'
        )
        parser.add_argument(
            '-t', '--coords', dest='coords',
            default='./ground_truth/floor_plan/room_coords.csv',
            help='Passes a csv file that represents the coordinates of all the rooms present on the map of the building'
        )
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)
        self.hparams = json.load(open(args.config))
        descriptors, histograms = self._get_descriptors_histograms()
        infer(args, self.hparams, descriptors, histograms)

    def _build_logger(self, level):
        logging.basicConfig(
            format='[%(levelname)s]\t%(asctime)s - %(message)s', level=max(4 - level, 0) * 10
        )

    def _build_parser(self, description):
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument('-v', '--verbose', dest='verbose_count',
                            action='count', default=0,
                            help='increases log verbosity for each occurence.')
        parser.add_argument(
            '-c', '--config',
            default='./config/hparams.json',
            help='Path to hparams file')
        return parser

    def _get_descriptors_histograms(self):
        if os.path.isfile('./db/features/descriptors.pickle'):
            logging.info('Reading descriptors from descriptors.pickle...')
            logging.warning(
                'When testing different descriptors, make sure to delete the previous pickles')
            with open('./db/features/descriptors.pickle', 'rb') as file:
                descriptors = pickle.load(file)
            with open('./db/features/histograms.pickle', 'rb') as file:
                histograms = pickle.load(file)
        else:
            if not os.path.exists('./db/features'):
                os.makedirs('./db/features')
            logging.info('Computing descriptors from db...')
            descriptors = dict()
            histograms = dict()
            extr = FeatureExtraction()
            for path, img in io.imread_folder('./db/images', resize=False):
                if img.shape == (512, 512, 3):
                    descriptors[path] = extr.extract_keypoints(
                        img, self.hparams)
                    histograms[path] = extr.extract_hist(img)

            logging.info('Writing descriptors to descriptors.pickle...')
            with open('./db/features/descriptors.pickle', 'wb+') as file:
                pickle.dump(descriptors, file,  protocol=-1)

            logging.info('Writing histograms to histograms.pickle...')
            with open('./db/features/histograms.pickle', 'wb+') as file:
                pickle.dump(histograms, file, protocol=-1)

        return descriptors, histograms


if __name__ == '__main__':
    PaintingClassifier()
