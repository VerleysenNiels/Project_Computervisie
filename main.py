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

level = logging.INFO


class PaintingClassifier(object):
    def __init__(self):
        self.check_versions()
        logging.basicConfig(
            format="[%(levelname)s] %(asctime)s - %(message)s", level=level
        )
        parser = argparse.ArgumentParser(
            description="Locate a painting in the MSK")
        parser.add_argument(
            "command", choices=["build", "train", "eval"], help="Subcommand to run"
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
        args = parser.parse_args(sys.argv[2:])

        for path, img in io_utils.imread_folder(args.directory):
            img = feature_detection.equalize_histogram(img)
            img = feature_detection.dilate(img)
            points = feature_detection.detect_perspective(img)
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
        parser = argparse.ArgumentParser(description="")
        args = parser.parse_args(sys.argv[2:])

    def eval(self):
        parser = argparse.ArgumentParser(description="")
        args = parser.parse_args(sys.argv[2:])


if __name__ == "__main__":
    PaintingClassifier()
