import cv2
import numpy as np
import logging
import viz_utils
import io_utils


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
            viz_utils.imshow(filtered, norm=True, resize=True)

        features = np.array(features)

        print(features.shape)

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

    def max_pool(self, image, size=16):
        """Assumes dimensions are n*size"""
        # Init array of shape (12, x/16, y/16)
        res = np.zeros((image.shape[0] // size, image.shape[1] // size),
                       dtype=np.float)

        for row in range(res.shape[0]):
            for col in range(res.shape[1]):
                arr_slice = image[row * size:row *
                                  size + size, col * size:col * size + size]
                res[row, col] = np.max(np.abs(arr_slice))
        return res


if __name__ == "__main__":
    extr = FeatureExtraction()
    for path, img in io_utils.imread_folder('./db'):
        extr.extract_features(img)
