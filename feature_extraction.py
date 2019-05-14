import cv2
import numpy as np
import logging
import viz_utils
import io_utils
from skimage import feature
import math


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
            # viz_utils.imshow(filtered, norm=True, resize=True)

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

    def extract_keypoints(self, img_gray):
        orb = cv2.ORB_create(nfeatures=300)
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)
        return descriptors

    def match_keypoints(self, descriptors_im, descriptors_db):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_im, descriptors_db)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) == 0:
            return math.inf

        score = 0
        for m in matches[:20]:
            score += m.distance
        # low score indicates good match
        return score / min(len(matches), 20)

    def texture_extraction(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(grayscale, n_points, radius)
        return lbp

    def gabor_filtering(self, img):
        g_kernel = cv2.getGaborKernel(
            (21, 21), 8.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.filter2D(grayscale, cv2.CV_8UC3, g_kernel)
        return filtered_img, g_kernel


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(message)s", level=logging.DEBUG
    )
    extr = FeatureExtraction()
    for _, img in io_utils.imread_folder('images/'):
        # img = cv2.medianBlur(img, 9)
        img = extr.texture_extraction(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = ((img > 254) + (img < 2)) * img
        viz_utils.imshow(img)
