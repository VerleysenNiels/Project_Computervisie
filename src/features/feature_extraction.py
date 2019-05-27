import logging

import math as pymath
import cv2
import numpy as np

import src.utils.io as io
import src.utils.math as math
import src.utils.viz as viz


class FeatureExtraction(object):

    def __init__(self):
        logging.debug('FeatureExtraction.__init__')
        self.filters = []
        for scale in [9, 19]:
            for angle in range(6):
                dog = self.create_dog(scale, angle * 30)
                self.filters.append(dog)

    def extract_features(self, image):
        bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = []
        for _filter in self.filters:
            filtered = cv2.filter2D(bw, cv2.CV_32F, _filter)
            filtered = self.max_pool(np.array(filtered))
            features.append(filtered)
            # viz.imshow(filtered, norm=True, resize=True)

        features = np.array(features)
        features = features.reshape((12, -1))
        colors = self.extract_colors(image)
        logging.debug('colors.shape = %s', colors.shape)
        logging.debug('features.shape = %s', features.shape)
        features = np.append(features, colors, axis=0).flatten()
        logging.debug('features.shape = %s', features.shape)
        return features

    def create_dog(self, dim, angle=0):
        center = (dim // 2, dim // 2)
        # Create 1D Gaussian kernel
        kernel = cv2.getGaussianKernel(dim, dim / 5)
        # Copy it to the middle of a square matrix
        kernel = np.pad(kernel, ((0, 0), center),
                        'constant', constant_values=(0, 0))
        # Create 1D row Gaussian kernel
        kernel2 = np.transpose(cv2.getGaussianKernel(dim, dim / 12))
        # Filter square with row kernel
        kernel = cv2.filter2D(kernel, -1, kernel2)
        # Derive kernel with Sobel
        kernel = cv2.Sobel(kernel, cv2.CV_64F, 1, 0)
        # Obtain rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        # Rotate kernel with matrix
        kernel = cv2.warpAffine(kernel, rot_mat, (dim, dim))
        return kernel

    def max_pool(self, image, block_size=16):
        """Assumes dimensions are n*size"""
        # Init array of shape (12, x/16, y/16)
        res = np.zeros((image.shape[0] // block_size, image.shape[1] // block_size),
                       dtype=np.float)

        for row in range(res.shape[0]):
            for col in range(res.shape[1]):
                arr_slice = image[row * block_size:row *
                                  block_size + block_size, col * block_size:col * block_size + block_size]
                res[row, col] = np.max(np.abs(arr_slice))
        return res

    def extract_colors(self, img, block_size=16):
        """ Extract average color for each block with given size
            Return a list for each color with the average values for that colorf
            Input must be a BGR image and a valid size for the blocks
        """
        red = []
        green = []
        blue = []

        for y in range(0, img.shape[0]//block_size):
            for x in range(0, img.shape[1] // block_size):
                blue.append(np.average(
                    img[y*block_size:y*block_size+block_size, x*block_size:x*block_size+block_size, 0]))
                green.append(np.average(
                    img[y*block_size:y*block_size+block_size, x*block_size:x*block_size+block_size, 1]))
                red.append(np.average(
                    img[y*block_size:y*block_size+block_size, x*block_size:x*block_size+block_size, 2]))

        return np.array([red, green, blue])

    def extract_keypoints(self, img, hparams):
        params = hparams['feature_matching']
        if params['type'] == 'ORB':
            detector = cv2.ORB_create(nfeatures=params['keypoint_thresh'])
        elif params['type'] == 'SURF':
            detector = cv2.xfeatures2d_SURF.create(
                hessianThreshold=params['keypoint_thresh'])
        elif params['type'] == 'SIFT':
            detector = cv2.xfeatures2d_SIFT.create()
        else:
            logging.critical(
                'Unknown feature_matching.type: %s', params['type'])
            exit(1)
        keypoints, descriptors = detector.detectAndCompute(img, None)
        return descriptors

    def match_keypoints(self, descriptors_im, descriptors_db, hparams):
        params = hparams['feature_matching']
        if params['type'] == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif params['type'] in ['SIFT', 'SURF']:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)   # or pass empty dictionary
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            logging.critical(
                'Unknown feature_matching.type: %s', params['type'])
            exit(1)

        if params['match_knn'] == False:
            matches = matcher.match(descriptors_im, descriptors_db)
            matches = sorted(matches, key=lambda x: x.distance)
            n = params['matches_amount']
            good_matches = matches[n:]
        else:
            matches = matcher.knnMatch(descriptors_db, descriptors_im, k=2)
            # store all the good matches as per Lowe's ratio test.
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) == 0:
            return 0
        score = 0
        for m in good_matches:
            score += m.distance
        # high score indicates good match
        return len(good_matches) / (score + .00001)

    def gabor_filtering(self, img):
        g_kernel = cv2.getGaborKernel(
            (21, 21), 8.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.filter2D(grayscale, cv2.CV_8UC3, g_kernel)
        return filtered_img, g_kernel

    def extract_hist(self, img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    def compare_hist(self, hist1, hist2):
        # -1 = no similarity
        # +1 = perfect similarity
        score = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CORREL)
        return (1+score)/2


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(message)s", level=logging.DEBUG
    )
    extr = FeatureExtraction()
    for _, img in io.imread_folder('images/'):
        # img = cv2.medianBlur(img, 9)
        viz.imshow(img)
        img = extr.texture_extraction(img)
        viz.imshow(img)
