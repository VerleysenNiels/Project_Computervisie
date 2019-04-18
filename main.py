import cv2
import platform
import numpy as np
import os
import io_utils
import viz_utils
import argparse
import sys
import feature_detection
import perspective
import logging

level = logging.DEBUG


class PaintingClassifier(object):

    def __init__(self):
        self.check_versions()
        logging.basicConfig(
            format='[%(levelname)s] %(asctime)s - %(message)s', level=level)
        parser = argparse.ArgumentParser(
            description='Locate a painting in the MSK')
        parser.add_argument(
            'command',
            choices=['build', 'train', 'eval'],
            help='Subcommand to run')

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def check_versions(self):
        assert platform.python_version().startswith('3.6')
        assert cv2.__version__.startswith('4.')
        assert np.__version__.startswith('1.16')

    def build(self):
        parser = argparse.ArgumentParser(
            description='Build painting database from raw images directory')
        parser.add_argument('directory')
        args = parser.parse_args(sys.argv[2:])

        for path, img in io_utils.imread_folder(args.directory):
            if logging.root.level == logging.DEBUG:
                viz_utils.imshow(img, resize=True)

            img = feature_detection.equalize_histogram(img)

            points = feature_detection.detect_perspective(img)
            img = perspective.perspective_transform(img, points)

            if logging.root.level == logging.DEBUG:
                viz_utils.imshow(img, resize=True)

            # Write to DB folder
            # io_utils.imwrite(os.path.join('db', os.path.basename(path)), img)

    def train(self):
        parser = argparse.ArgumentParser(description='')
        args = parser.parse_args(sys.argv[2:])

    def eval(self):
        parser = argparse.ArgumentParser(description='')
        args = parser.parse_args(sys.argv[2:])


if __name__ == '__main__':
    PaintingClassifier()
