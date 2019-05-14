import argparse
import logging
import math
import os
import pickle
import platform
import sys

import cv2
import numpy as np

import feature_detection
import io_utils
import math_utils
import perspective
import viz_utils
import Room_graph
from classifiers import RandomForestClassifier
from feature_extraction import FeatureExtraction


class PaintingClassifier(object):
    def __init__(self):
        self.check_versions()
        parser = argparse.ArgumentParser(
            description="Locate a painting in the MSK")
        parser.add_argument(
            "command", choices=["build", "train", "infer"], help="Subcommand to run"
        )

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def check_versions(self):
        assert cv2.__version__.startswith("4.")

    def build(self):
        parser = argparse.ArgumentParser(
            description="Build painting database from raw images directory"
        )
        parser.add_argument("directory")
        parser.add_argument("-v", "--verbose", dest="verbose_count",
                            action="count", default=0,
                            help="increases log verbosity for each occurence.")
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)

        for path, img in io_utils.imread_folder(args.directory):
            # img = feature_detection.equalize_histogram(img)
            # img = feature_detection.dilate(img)
            points, img = feature_detection.detect_perspective(img)
            if logging.root.level == logging.DEBUG:
                viz_utils.imshow(img, resize=True)
            img = perspective.perspective_transform(img, points)

            # Write to DB folder
            label = os.path.basename(os.path.dirname(path))
            out_path = os.path.join(
                'db', label.lower(), os.path.basename(path))
            logging.info('Writing to ' + out_path)
            io_utils.imwrite(out_path, img)

    def train(self):
        """ Train classifier built from db folder.
            Example command:

        """
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-v", "--verbose", dest="verbose_count",
                            action="count", default=0,
                            help="increases log verbosity for each occurence.")
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)

        logging.info('Build features from ./db')
        feature_extraction = FeatureExtraction()
        X, y = [], []
        for path, img in io_utils.imread_folder('./db', resize=False):
            if img.shape != (512, 512, 3):
                logging.error(
                    'Skipping %s because of incorrect shape %s', path, img.shape)
                continue
            label = os.path.basename(os.path.dirname(path))
            y.append(label)

            features = feature_extraction.extract_features(img)
            X.append(features)

        X = np.array(X)
        y = np.array(y)

        logging.info('Train classifier on %d samples...', X.shape[0])
        logging.debug('X.shape = %s', X.shape)
        logging.debug('y.shape = %s', y.shape)
        classifier = RandomForestClassifier()
        classifier.train(X, y)
        logging.info('Evaluate classifier on training data...')
        accuracy = classifier.eval(X, y)
        logging.info('Accuracy: %f', accuracy)
        with open(classifier, 'wb+') as file:
            pickle.dump(classifier, file)

    def infer(self):
        '''
        Example:
            python main.py infer .\videos\MSK_01.mp4 -v -v

        Possible parameters for finetuning:
            border for too blurry images
            amount of previous labels used to determine the average label
            amount of transitions the algorithm wants to change room, but is not allowed (prevent being stuck in wrong room)
        '''
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "file", help="Video file to infer the hall ID from.", type=argparse.FileType('r'))
        parser.add_argument("-v", "--verbose", dest="verbose_count",
                            action="count", default=0,
                            help="increases log verbosity for each occurence.")
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)

        extr = FeatureExtraction()
        if os.path.isfile('descriptors.pickle'):
            logging.info('Reading descriptors from descriptors.pickle...')
            with open('descriptors.pickle', 'rb') as file:
                descriptors = pickle.load(file)
            with open('histograms.pickle', 'rb') as file:
                histograms = pickle.load(file)

        else:
            logging.info('Computing descriptors from db...')
            descriptors = dict()
            histograms = dict()
            for path, img in io_utils.imread_folder('./db', resize=False):
                if img.shape == (512, 512, 3):
                    descriptors[path] = extr.extract_keypoints(img)
                    histograms[path] = extr.extract_hist(img)


            logging.info('Writing descriptors to descriptors.pickle...')
            with open('descriptors.pickle', 'wb+') as file:
                # protocol 0 is printable ASCII
                pickle.dump(descriptors, file,  protocol=-1)

            with open('histograms.pickle', 'wb+') as file:
                pickle.dump(histograms, file, protocol=-1)

        logging.warning('Press Q to quit')
        labels = []

        painting = np.zeros((10, 10, 3), np.uint8)

        grondplan = cv2.imread(".\msk_grondplan.jpg")
        hall = None  # Keep track of current room
        stuck = 0  # Counter to detect being stuck in a room (bug when using graph)
        modes = ["ERROR_MODE", "WARNING_MODE", "INFO_MODE", "DEBUG_MODE"]

        for frame in io_utils.read_video(args.file.name, interval=5):
            frame = cv2.resize(
                frame, (0, 0),
                fx=720 / frame.shape[0],
                fy=720 / frame.shape[0],
                interpolation=cv2.INTER_AREA)

            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            blurry = cv2.Laplacian(frame, cv2.CV_64F).var()


            # Change this border for blurry
            if blurry > 65:
                points, frame = feature_detection.detect_perspective(
                    frame, remove_hblur=True, minLineLength=70, maxLineGap=5)

                if len(points) == 4:

                    best_score = math.inf
                    best = '?'
                    current = best
                    points = perspective.order_points(points)
                    img = perspective.perspective_transform(frame, points)

                    descriptor = extr.extract_keypoints(img)
                    histogram_frame = extr.extract_hist(img)
                    for path in descriptors:
                        if descriptors[path] is not None:
                            score_key = extr.match_keypoints(
                                descriptor, descriptors[path])
                            score_hist = extr.compare_hist(histogram_frame, histograms[path])
                            score = 0.5*score_key + 0.5*score_hist
                            if score < best_score:
                                best = path
                                best_score = score
                    logging.info(best)
                    if best != '?':
                        if best != current:
                            current = best
                            painting = cv2.imread(best)
                        labels.append(best)
                        labels = labels[-15:]
                    next_hall = math_utils.rolling_avg(labels)
                    if hall is None or hall == next_hall:
                        hall = next_hall
                        stuck = 0
                    elif Room_graph.transition_possible(hall, next_hall):
                        viz_utils.draw_path_line(grondplan, str(next_hall), str(hall))
                        hall = next_hall
                        stuck = 0
                    else:
                        stuck += 1

                    # ToDo: Needs finetuning
                    # Alllow transition if algorithm is stuck in a room
                    if stuck > 17:
                        hall = next_hall
                        stuck = 0


            # Write amount of blurriness

                frame = cv2.putText(frame, "Not blurry: " + str(round(blurry)), (20, 40), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Too blurry: " + str(round(blurry)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

            h, w = frame.shape[:2]
            h1, w1 = painting.shape[:2]
            blank_image = np.zeros((h, int(0.5*w), 3), np.uint8)
            blank_image[0:h1, 0:w1] = painting
            frame = np.concatenate((frame, blank_image), axis=1)

            # Write predicted room and display image
            frame = cv2.putText(frame, hall, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)
            frame = cv2.putText(frame, modes[args.verbose_count], (20, 20), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.imshow(args.file.name + ' (press Q to quit)', frame)
            cv2.imshow('Grondplan', grondplan)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _build_logger(self, level):
        logging.basicConfig(
            format="[%(levelname)s]\t%(asctime)s - %(message)s", level=max(4 - level, 0) * 10
        )


if __name__ == "__main__":
    PaintingClassifier()
