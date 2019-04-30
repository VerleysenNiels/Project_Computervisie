import cv2
import platform
import numpy as np
import os
import io_utils
import viz_utils
import argparse
import sys
import feature_detection
from feature_extraction import FeatureExtraction
from classifiers import RandomForestClassifier, NeuralNetClassifier
import perspective
import logging


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
            img = feature_detection.equalize_histogram(img)
            img = feature_detection.dilate(img)
            points, img = feature_detection.detect_perspective(img)
            img = perspective.perspective_transform(img, points)

            if logging.root.level == logging.DEBUG:
                viz_utils.imshow(img, resize=True)

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
        #classifier = RandomForestClassifier()
        classifier = NeuralNetClassifier()
        classifier.train(X, y)
        logging.info('Evaluate classifier on training data...')
        accuracy = classifier.eval(X, y)
        logging.info('Accuracy: %f', accuracy)

    def infer(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "file", help="Video file to infer the hall ID from.", type=argparse.FileType('r'))
        parser.add_argument("-v", "--verbose", dest="verbose_count",
                            action="count", default=0,
                            help="increases log verbosity for each occurence.")
        args = parser.parse_args(sys.argv[2:])
        self._build_logger(args.verbose_count)
        logging.warning('Press Q to quit')
        for frame in io_utils.read_video(args.file.name, interval=1):
            frame = cv2.resize(
                frame, (0, 0),
                fx=720 / frame.shape[0],
                fy=720 / frame.shape[0],
                interpolation=cv2.INTER_AREA)

            points, frame = feature_detection.detect_perspective(
                frame, remove_hblur=True, minLineLength=70, maxLineGap=5)
            cv2.imshow(args.file.name + ' (press Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _build_logger(self, level):
        logging.basicConfig(
            format="[%(levelname)s]\t%(asctime)s - %(message)s", level=max(3 - level, 0) * 10
        )


if __name__ == "__main__":
    PaintingClassifier()
