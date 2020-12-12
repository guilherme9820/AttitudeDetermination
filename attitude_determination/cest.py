# -*- coding: utf-8 -*-
"""
CEST
=====

CEST - Centroid Extractor for Star Trackers adapted from https://github.com/mgm8/cest.

"""


import pandas as pd
import numpy as np
import cv2


class StarPixel:

    def __init__(self, pos_row, pos_col, pixel_value):
        self.pos_row = pos_row
        self.pos_col = pos_col
        self.pixel_value = pixel_value


class Centroid:

    def __init__(self, pos_x=0, pos_y=0, value=0):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.value = value
        self.pixels = 0


def get_star_pixels(image: np.ndarray, threshold: int) -> list:

    if image.ndim > 2:
        image = image[:, :, 1]  # Green channel

    height, width = image.shape

    star_pixels = []

    for row in range(height):
        for col in range(width):

            pixel_value = image[row, col]

            if pixel_value > threshold:
                star_pixel = StarPixel(int(row), int(col), int(pixel_value))

                star_pixels.append(star_pixel)

    return star_pixels


class CDPU:

    def __init__(self):
        # Pixel distance threshold value (Euclidean distance).
        self._DISTANCE_THRESHOLD_ = 5
        self._DISTANCE_THRESHOLD_MAN_ = 2 * \
            np.sqrt(self._DISTANCE_THRESHOLD_**2) / np.sqrt(2)

        self.centroid = Centroid()
        self.pixels = 0
        self.weight = 1

    def update(self, new_x, new_y, new_value, weight):

        if self.distance_from(new_x, new_y) < self._DISTANCE_THRESHOLD_MAN_:
            self.weight *= weight
            self.centroid.pos_x = self.weight * \
                self.centroid.pos_x + (1 - self.weight) * new_x
            self.centroid.pos_y = self.weight * \
                self.centroid.pos_y + (1 - self.weight) * new_y

            self.centroid.value = int(self.weight * self.centroid.value
                                      + (1 - self.weight) * new_value)
            self.centroid.pixels += 1

            self.pixels += 1

    def set_centroid(self, new_x, new_y, new_value):
        self.centroid.pos_x = new_x
        self.centroid.pos_y = new_y
        self.centroid.value += new_value
        self.centroid.pixels = 1

    def distance_from(self, target_x, target_y):
        return np.abs(self.centroid.pos_x - target_x) + np.abs(self.centroid.pos_y - target_y)

    def get_centroid(self):
        return self.centroid

    def get_pixels(self):
        return self.pixels


class Centroider:

    def __init__(self, max_cdpus=None, distance_threshold=None):

        self.max_cdpus = max_cdpus or 20
        self.distance_threshold = distance_threshold or 10
        self.reset()

    def compute(self, star_pixel, weight):

        star_x = star_pixel.pos_col
        star_y = star_pixel.pos_row

        if len(self.cdpus) < self.max_cdpus:

            pix_capt = False

            for cdpu in self.cdpus:
                if cdpu.distance_from(star_x, star_y) < self.distance_threshold:
                    pix_capt = True
                    break

            if not pix_capt:
                self.cdpus.append(CDPU())
                self.cdpus[-1].set_centroid(star_x,
                                            star_y,
                                            star_pixel.pixel_value)

        for cdpu in self.cdpus:
            cdpu.update(star_x, star_y, star_pixel.pixel_value, weight)

    def compute_from_list(self, star_pixels, weight):
        self.reset()

        for star_pixel in star_pixels:
            self.compute(star_pixel, weight)

        return self.get_centroids()

    def get_centroids(self):

        centroids = []

        for cdpu in self.cdpus:
            centroids.append(cdpu.get_centroid())

        return centroids

    def sort_centroids(self, centroids):
        def policy(element):
            return element.value * element.pixels

        return sorted(centroids, key=policy, reverse=True)

    def reset(self):
        self.cdpus = []

    def print_centroids(self, image, centroids, print_id):

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for centroid in centroids:
            # Center coordinates
            center_coordinates = (int(centroid.pos_x), int(centroid.pos_y))

            img_res = cv2.circle(image, center_coordinates, 5, (0, 255, 0), 1)

        if print_id:
            centroids = self.sort_centroids(centroids)

            for idx, centroid in enumerate(centroids):
                img_res = cv2.putText(img_res,
                                      str(idx+1),
                                      (int(centroid.pos_x)+10,
                                       int(centroid.pos_y)+10),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      (0, 0, 255))

        return img_res

    def save_centroids(self, filename):

        columns = ['pixels', 'value', 'x', 'y']

        centroids = np.array([]).reshape(0, len(columns))

        for centroid in self.get_centroids():

            temp = [centroid.pixels, centroid.value,
                    centroid.pos_x, centroid.pos_y]

            centroids = np.vstack([centroids, temp])

        dataframe = pd.DataFrame(data=centroids, columns=columns)

        dataframe.to_csv(filename, index=False)


if __name__ == "__main__":

    STAR_THRESHOLD_VALUE = 150
    GAIN_WEIGHT = 0.8
    MAX_NUMBER_OF_CENTROIDS = 60

    # star_image = cv2.imread("stars-image.png", 0)
    star_image = cv2.imread("/home/gsantos/Documents/TensorflowProjects/DeepLearning/datasets/star_dataset/groundtruth/fov_8/hip_26311.png", 0)

    centroids1 = get_k_top_centroids(star_image, k_top=8)

    # star_pixels1 = get_star_pixels(star_image, STAR_THRESHOLD_VALUE)

    centroider1 = Centroider(max_cdpus=MAX_NUMBER_OF_CENTROIDS)

    # centroids1 = centroider1.compute_from_list(star_pixels1, GAIN_WEIGHT)

    result = centroider1.print_centroids(star_image, centroids1, True)

    # centroider1.save_centroids("data_t.csv")

    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
