import argparse
import json
import logging
import math as pymath
import os
import pickle
import platform
import shutil
import sys

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
        descriptors, histograms = self._get_descriptors_histograms()
        file = open(os.path.join(
            args.output_dir, 'accuracy-corners.csv'), 'w+')
        file.write('Path;Expected;Predicted\n')
        for path, img in io.imread_folder(args.ground_truth_dir, resize=False):
            best, best_score, img = infer_frame(
                img, FeatureExtraction(), descriptors, histograms, self.hparams)
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
            total += 1
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

        groundTruth = None
        frames_correct = 0
        frames = 1
        if measurementMode:
            groundTruth = VideoGroundTruth()
            groundTruth.read_file(args.ground_truth)

        logging.warning('Press Q to quit')
        labels = []

        painting = np.zeros((10, 10, 3), np.uint8)

        room_graph = RoomGraph(args.room_file)

        floor_plan = cv2.imread('.\msk_grondplan.jpg')
        blank_image = None
        hall = None  # Keep track of current room (internally)
        likely_hall = None
        stuck = 0  # Counter to detect being stuck in a room (bug when using graph)

        modes = ['ERROR_MODE', 'WARNING_MODE', 'INFO_MODE', 'DEBUG_MODE']

        for frame in io_utils.read_video(args.file.name, interval=self.hparams['frame_sampling']):
            #frame = viz_utils.process_gopro_video(frame, 6, 10)
            frame = cv2.resize(
                frame, (0, 0),
                fx=720 / frame.shape[0],
                fy=720 / frame.shape[0],
                interpolation=cv2.INTER_AREA)

            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            blurry = cv2.Laplacian(frame, cv2.CV_64F).var()

            # Change this border for blurry
            if blurry < self.hparams['blurry_threshold']:
                points, frame = feature_detection.detect_perspective(
                    frame, self.hparams['video'])

                if len(points) == 4:

                    best_score = math.inf
                    best = '?'
                    current = best
                    points = perspective.order_points(points)
                    img = perspective.perspective_transform(frame, points)

                    descriptor = extr.extract_keypoints(img)
                    histogram_frame = extr.extract_hist(img)
                    for path in descriptors:
                        if descriptors[path] is not None and histograms[path] is not None:
                            score_key = extr.match_keypoints(
                                descriptor, descriptors[path])
                            score_hist = extr.compare_hist(
                                histogram_frame, histograms[path])
                            score = self.hparams['keypoints_weight'] * score_key + \
                                self.hparams['histogram_weight'] * score_hist
                            if score < best_score:
                                best = path
                                best_score = score
                    logging.info(best)
                    if best != '?':
                        if best != current:
                            painting = cv2.imread(best)
                        labels.append(best)
                        window = self.hparams['rolling_avg_window']
                        labels = labels[-window:]
                    next_hall = math_utils.rolling_avg(labels)
                    if hall is None or hall == next_hall:
                        hall = next_hall
                        stuck = 0
                    elif room_graph.transition_possible(hall, next_hall):
                        viz_utils.draw_path_line(
                            floor_plan, str(next_hall), str(hall))
                        hall = next_hall
                        stuck = 0
                    else:
                        stuck += 1

                    # Alllow transition if algorithm is stuck in a room
                    if stuck > self.hparams['stuck_threshold']:
                        hall = next_hall
                        stuck = 0

                    # Calculate most likely path
                    changed, highest_likely_path = room_graph.highest_likely_path(hall)
                    if changed:
                        # REDRAW PATH
                        floor_plan = cv2.imread('.\msk_grondplan.jpg')
                        for r in range(0, len(highest_likely_path) - 2):
                            viz_utils.draw_path_line(floor_plan, str(highest_likely_path[r]), str(highest_likely_path[r+1]))

                        # CHANGE MOST LIKELY HALL
                        likely_hall = highest_likely_path[-1]


                # Write amount of blurriness
                frame = cv2.putText(frame, 'Not blurry: %.0f' % blurry, (20, 40), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, 'Too blurry: %.0f' % blurry, (20, 40),
                                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

            h, w = frame.shape[:2]
            try:
                h1, w1 = painting.shape[:2]
                blank_image = np.zeros((h, int(0.5 * w), 3), np.uint8)
                blank_image[0:h1, 0:w1] = painting
            except AttributeError:
                logging.info('Not an image')
            frame = np.concatenate((frame, blank_image), axis=1)

            # Write predicted room and display image
            frame = cv2.putText(
                frame, likely_hall, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)
            frame = cv2.putText(frame, modes[args.verbose_count], (20, 20), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.imshow(args.file.name + ' (press Q to quit)', frame)
            cv2.namedWindow('Floor Plan', cv2.WINDOW_NORMAL)
            cv2.imshow('Floor Plan', floor_plan)

            if measurementMode and groundTruth.room_in_frame(frames) == likely_hall:
                    frames_correct += 1

            frames += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        if measurementMode:
            print('Accuracy: ' + str(frames_correct / (frames - 1)))

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
        return descriptors, histograms


if __name__ == '__main__':
    PaintingClassifier()
