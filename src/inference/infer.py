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
from src.features.feature_extraction import FeatureExtraction
from src.inference.room_graph import RoomGraph
from src.evaluation.video_ground_truth import VideoGroundTruth


def infer(args, hparams, descriptors, histograms):

    measurementMode = args.ground_truth is not None and os.path.isfile(
        args.ground_truth)

    extr = FeatureExtraction()
    groundTruth = None
    if measurementMode:
        frames_correct = 0
        frames = 0
        groundTruth = VideoGroundTruth()
        groundTruth.read_file(args.ground_truth)

    labels = []
    painting = np.zeros((10, 10, 3), np.uint8)
    room_graph = RoomGraph(args.room_file)
    floor_plan = cv2.imread('./ground_truth/floor_plan/msk.jpg')
    room_coords = viz.read_room_coords(
        './ground_truth/floor_plan/room_coords.csv')
    blank_image = None
    current_room = None  # Keep track of current room
    # Counter to detect being stuck in a room (bug when using graph)
    stuck = 0
    metadata = dict()
    modes = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
    metadata['Mode'] = modes[args.verbose_count]
    for frame in io.read_video(args.file.name, interval=hparams['frame_sampling']):
        # frame = viz.process_gopro_video(frame, 6, 10)
        frame = cv2.resize(
            frame, (0, 0),
            fx=720 / frame.shape[0],
            fy=720 / frame.shape[0],
            interpolation=cv2.INTER_AREA)

        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        blurry = cv2.Laplacian(frame, cv2.CV_64F).var()

        # Change this border for blurry
        if blurry < hparams['blurry_threshold']:
            points, frame = feature_detection.detect_perspective(
                frame, hparams['video'])

            if len(points) == 4:

                points = perspective.order_points(points)
                img = perspective.perspective_transform(frame, points)
                descriptor = extr.extract_keypoints(img, hparams)
                histogram_frame = extr.extract_hist(img)

                best_score = 0
                best = None
                for path in descriptors:
                    if descriptors[path] is not None and histograms[path] is not None:
                        score_key = extr.match_keypoints(
                            descriptor, descriptors[path], hparams)
                        score_hist = extr.compare_hist(
                            histogram_frame, histograms[path])
                        score = hparams['keypoints_weight'] * score_key + \
                            hparams['histogram_weight'] * \
                            score_hist
                        if score > best_score:
                            best = path
                            best_score = score
                logging.info(best)
                if best:
                    if best != current_room:
                        painting = cv2.imread(best)
                        painting = cv2.resize(
                            painting, (0, 0),
                            fx=360 / img.shape[0],
                            fy=360 / img.shape[0],
                            interpolation=cv2.INTER_AREA)
                    labels.append(best)
                    window = hparams['rolling_avg_window']
                    labels = labels[-window:]
                next_hall = math.rolling_avg(labels)
                if not current_room or current_room == next_hall:
                    current_room = next_hall
                    stuck = 0
                elif room_graph.transition_possible(current_room, next_hall):
                    viz.draw_path_line(
                        floor_plan, str(next_hall), str(current_room), room_coords)
                    current_room = next_hall
                    stuck = 0
                else:
                    stuck += 1

                # Alllow transition if algorithm is stuck in a room
                if stuck > hparams['stuck_threshold']:
                    current_room = next_hall
                    stuck = 0
            metadata['Blurriness'] = 'Not blurry (%.0f)' % blurry
        else:
            metadata['Blurriness'] = 'Too blurry (%.0f)' % blurry

        if measurementMode:
            frames += 1
            if groundTruth.room_in_frame(frames) == current_room:
                frames_correct += 1
            metadata['Cumulative acc.'] = '%.1f%%' % (
                100 * frames_correct / frames)
            logging.info('Cumulative accuracy: %.1f%%',
                         100*frames_correct / frames)

        if not args.silent:
            h, w = frame.shape[:2]
            try:
                h1, w1 = painting.shape[:2]
                blank_image = np.zeros((h, 510, 3), np.uint8)
                blank_image[0:h1, 0:h1] = painting
            except AttributeError:
                logging.info('Not an image')
            frame = np.concatenate((frame, blank_image), axis=1)
            frame[h-360:h, w:w+510] = floor_plan

            metadata['Prediction'] = current_room if current_room else '?'
            for i, key in enumerate(metadata):
                text = key + ': ' + metadata[key]
                frame = cv2.putText(
                    frame, text, (20, 20*(i+1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            # Write predicted room and display image

            cv2.imshow(args.file.name + ' (press Q to quit)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    if measurementMode:
        logging.info('Global accuracy: %.1f%%', 100*frames_correct / frames)

    return frames_correct / frame
