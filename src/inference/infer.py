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
from src.inference.room_graph import RoomGraph


def infer_frame(frame, extraction, descriptors, histograms, hparams):
    points, frame = feature_detection.detect_perspective(
        frame, hparams['video'])

    best = None
    best_score = 0
    if len(points) == 4:
        points = perspective.order_points(points)
        painting = perspective.perspective_transform(frame, points)
        descriptor = extraction.extract_keypoints(painting, hparams)
        histogram = extraction.extract_hist(painting)

        logits_descriptor = []
        logits_histogram = []
        labels = []
        for path in descriptors:
            if descriptors[path] is not None and histograms[path] is not None:
                score_key = extraction.match_keypoints(
                    descriptor, descriptors[path], hparams)
                score_hist = extraction.compare_hist(
                    histogram, histograms[path])
                logits_descriptor.append(score_key)
                logits_histogram.append(score_hist)
                labels.append(path)

        scores_descriptor = math.softmax(logits_descriptor)
        scores_histogram = math.softmax(logits_histogram)
        scores = hparams['keypoints_weight'] * scores_descriptor + \
            hparams['histogram_weight'] * scores_histogram
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best = labels[best_idx]
    return best, best_score, frame


def infer(args, hparams, descriptors, histograms):

    measurementMode = args.ground_truth is not None and os.path.isfile(
        args.ground_truth)

    goproMode = False
    if args.gopro_mode > 0:
        goproMode = True
        logging.info('Enabled gopro mode')

    extr = FeatureExtraction()
    groundTruth = None
    if measurementMode:
        frames_correct = 0
        matches_correct = 0
        frames = 0
        groundTruth = VideoGroundTruth()
        groundTruth.read_file(args.ground_truth)

    labels = []
    painting = np.zeros((10, 10, 3), np.uint8)
    room_graph = RoomGraph(args.room_file)
    floor_plan = cv2.imread(args.map)
    room_coords = viz.read_room_coords(args.coords)
    highest_likely_path = []
    blank_image = None
    current_room = None  # Keep track of current room (internally)
    metadata = dict()
    modes = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
    metadata['Mode'] = modes[args.verbose_count]
    total_time, total_frames = 0, 0
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
        if blurry > hparams['blurry_threshold']:

            start_time = time.time()
            best, best_score, frame = infer_frame(
                frame, extr, descriptors, histograms, hparams)
            end_time = time.time()
            logging.info('%s (confidence: %.2f%%)', best, 100*best_score)
            total_time += (end_time - start_time)
            total_frames += 1
            logging.info('Average matching speed: %.2fs per frame',
                         total_time/total_frames)
            if best:

                if best != current_room:
                    painting = cv2.imread(best)
                    painting = cv2.resize(
                        painting, (0, 0),
                        fx=360 / painting.shape[0],
                        fy=360 / painting.shape[0],
                        interpolation=cv2.INTER_AREA)

                next_hall = os.path.basename(os.path.dirname(best))
                score = best_score
                logging.info('%s (%.2f%% sure)', next_hall, 100 * score)

                # Calculate most likely path
                changed, highest_likely_path = room_graph.highest_likely_path(
                    next_hall, score)

                if changed:
                    # REDRAW PATH
                    floor_plan = cv2.imread(args.map)
                    for r in range(0, len(highest_likely_path) - 1):
                        viz.draw_path_line(floor_plan, str(highest_likely_path[r]), str(
                            highest_likely_path[r + 1]), room_coords)

            metadata['Blurriness'] = 'Not blurry (%.0f)' % blurry
        else:
            metadata['Blurriness'] = 'Too blurry (%.0f)' % blurry

        if measurementMode:
            frames += 1
            if next_hall and groundTruth.room_in_frame(frames * hparams['frame_sampling']) == next_hall:
                matches_correct += 1
            if len(highest_likely_path) > 0 and groundTruth.room_in_frame(frames * hparams['frame_sampling']) == highest_likely_path[-1]:
                frames_correct += 1

            metadata['Cumulative acc.'] = '%.1f%%' % (
                100 * frames_correct / frames)
            logging.info('Cumulative accuracy: %.1f%%',
                         100*frames_correct / frames)
            logging.info('Cumulative matching accuracy: %.1f%%',
                         100*matches_correct / frames)

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

            metadata['Prediction'] = highest_likely_path[-1] if len(
                highest_likely_path) > 0 else '?'
            for i, key in enumerate(metadata):
                text = key + ': ' + metadata[key]
                frame = cv2.putText(
                    frame, text, (20, 20*(i+1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            # Write predicted room and display image

            cv2.imshow(args.file.name + ' (press Q to quit)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    if measurementMode:
        logging.info('Global accuracy: %.1f%%', 100*frames_correct / frames)

    return frames_correct / frames
