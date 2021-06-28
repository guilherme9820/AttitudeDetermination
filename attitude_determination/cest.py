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
    """ Holds an image coordinates along with its pixel intensity.

        In the following example the star_pixel object will hold
        the pixel value at the coordinate (row=50, column=35).

            star_pixel = StarPixel(50, 35, 150)

        Args:
            pos_row: Row coordinate.
            pos_col: Column coordinate.
            pixel_value: Pixel intensity [0, 255].
    """

    def __init__(self, pos_row, pos_col, pixel_value):
        self.pos_row = pos_row
        self.pos_col = pos_col
        self.pixel_value = pixel_value


class Centroid:
    """ Holds the centroid coordinates, its pixel intensity, and the
        number of pixels that compose this centroid.

        Args:
            pos_x: Row coordinate.
            pos_y: Column coordinate.
            value: Pixel intensity [0, 255].
            pixel: Number of pixels used to compose this centroid.
    """

    def __init__(self, pos_x=0, pos_y=0, value=0, pixels=0):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.value = value
        self.pixels = pixels


def get_star_pixels(image: np.ndarray, threshold: int) -> list:
    """ Retrieves a collection of star pixels given a 2-D array
        containing the star image. The pixel will only be added
        if its value is greater or equal than the threshold value.

        Args:
            image: A 2-D numpy array containing the image.
            threshold: Pixel threshold value.
    """

    if image.ndim > 2:
        image = image[:, :, 1]  # Green channel

    height, width = image.shape

    star_pixels = []

    for row in range(height):
        for col in range(width):

            pixel_value = image[row, col]

            if pixel_value >= threshold:
                star_pixel = StarPixel(int(row), int(col), int(pixel_value))

                star_pixels.append(star_pixel)

    return star_pixels


class CDPU:

    def __init__(self):
        # Pixel distance threshold value (Euclidean distance).
        self._DISTANCE_THRESHOLD_ = 5
        self._DISTANCE_THRESHOLD_MAN_ = self._DISTANCE_THRESHOLD_ * np.sqrt(2)

        self.centroid = Centroid()
        self.weight = 1

    def update(self, new_x, new_y, new_value, weight):
        """ Updates the centroid position and value based on a new star pixel.
            If this new star pixel lies within a radius of _DISTANCE_THRESHOLD_MAN_
            from the current centroid, it will update the centroid attributes.

        Args:
            new_x: The row coordinate of the new star pixel.
            new_y: The column coordinate of the new star pixel.
            new_value: The pixel intensity of the new star pixel.
            weight: A weight ranging between 0 and 1 which dictates the relevance
                    of the new star pixel for the calculation.
        """

        if self.distance_from(new_x, new_y) <= self._DISTANCE_THRESHOLD_MAN_:
            self.weight *= weight
            self.centroid.pos_x = self.weight * \
                self.centroid.pos_x + (1 - self.weight) * new_x
            self.centroid.pos_y = self.weight * \
                self.centroid.pos_y + (1 - self.weight) * new_y

            self.centroid.value = self.weight * self.centroid.value \
                + (1 - self.weight) * new_value
            self.centroid.pixels += 1

    def set_centroid(self, new_x, new_y, new_value, pixels=1):
        """ Sets the centroid attributes to the CDPU.

        Args:
            new_x: The row coordinate of the new star pixel.
            new_y: The column coordinate of the new star pixel.
            new_value: The pixel intensity of the new star pixel.
            pixels: The number of pixels that composes the centroid.
        """
        self.centroid.pos_x = new_x
        self.centroid.pos_y = new_y
        self.centroid.value = new_value
        self.centroid.pixels = pixels

    def distance_from(self, target_x, target_y):
        """ Evaluates the Manhattan distance from the current centroid
            to the given coordinate.

        Args:
            target_x: A row coordinate.
            target_y: A column coordinate.
        """
        return np.abs(self.centroid.pos_x - target_x) + np.abs(self.centroid.pos_y - target_y)

    def get_centroid(self):
        return self.centroid

    def get_pixels(self):
        return self.centroid.pixels


class Centroider:
    """ This class is responsible for taking a collection of StarPixels and
        evaluate all possible centroids.

        Args:
            max_cdpus: Maximum number of centroids that will be evaluated.
            distance_threshold: A soft threshold distance between a given
                                StarPixel and a centroid.
    """

    def __init__(self, max_cdpus=None, distance_threshold=None):

        self.max_cdpus = max_cdpus or 20
        self.distance_threshold = distance_threshold or 10
        self.cdpus = None

    def compute(self, star_pixel, weight):
        """ Updates the CDPUs given a new StarPixel object and a weight.

        Args:
            star_pixel: A StarPixel object.
            weight: A weight ranging between 0 and 1 which dictates the relevance
                    of the new star pixel for the calculation.
        """

        star_x = star_pixel.pos_col
        star_y = star_pixel.pos_row

        if len(self.cdpus) < self.max_cdpus:

            pix_capt = False

            for cdpu in self.cdpus:
                if cdpu.distance_from(star_x, star_y) <= self.distance_threshold:
                    pix_capt = True
                    break

            if not pix_capt:
                self.add_cdpu(star_x,
                              star_y,
                              star_pixel.pixel_value,
                              1)

        for cdpu in self.cdpus:
            cdpu.update(star_x, star_y, star_pixel.pixel_value, weight)

    def compute_from_list(self, star_pixels, weight=0.8):
        """ Updates the CDPUs given a set of StarPixel objects and a weight.

        Args:
            star_pixels: A set of StarPixel objects.
            weight: A weight ranging between 0 and 1 which dictates the relevance
                    of the new star pixel for the calculation.
        """
        self.reset()

        for star_pixel in star_pixels:
            self.compute(star_pixel, weight)

        return self.get_centroids()

    def add_cdpu(self, pos_x, pos_y, value, pixels):
        """ Append a new CDPU to the list of CDPUs.

        Args:
            pos_x: The row coordinate of the new star pixel.
            pos_y: The column coordinate of the new star pixel.
            value: The pixel intensity of the new star pixel.
            pixels: The number of pixels that composes the centroid.
        """
        cdpu = CDPU()
        cdpu.set_centroid(pos_x,
                          pos_y,
                          value,
                          pixels)
        self.cdpus.append(cdpu)

    def get_centroids(self):
        """ Retrieves all centroids.
        """

        centroids = []

        for cdpu in self.cdpus:
            centroids.append(cdpu.get_centroid())

        return centroids

    def sort_centroids(self, centroids):
        """ Sorts the centroids in descending order. The sorting policy takes
            into consideration the centroid pixel intensity and the number of
            pixels that compose it.

        Args:
            centroids: List of centroids.
        """

        def policy(element):
            return element.value * element.pixels

        return sorted(centroids, key=policy, reverse=True)

    def reset(self):
        """ Cleans up the list of CDPUs.
        """
        self.cdpus = []

    def print_centroids(self, image, centroids, print_id=True):
        """ Annotates the centroids in the image.

        Args:
            image: A numpy 2-D array containing the image.
            centroids: A list of centroids.
            print_id (optional): It will display the centroid index. Defaults to True.

        Returns:
            Returns the annotated image.
        """
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

        return cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)

    def save_centroids(self, filename):
        """ Saves the centroids in a CSV file. The resulting dataframe
            will have 4 columns: ['pixels', 'value', 'x', 'y'].

        Args:
            filename: The CSV filename.
        """
        columns = ['pixels', 'value', 'x', 'y']

        centroids = np.array([]).reshape(0, len(columns))

        for centroid in self.get_centroids():

            temp = [centroid.pixels, centroid.value,
                    centroid.pos_x, centroid.pos_y]

            centroids = np.vstack([centroids, temp])

        dataframe = pd.DataFrame(data=centroids, columns=columns)

        dataframe.to_csv(filename, index=False)

    def load_centroids(self, csv_file):
        """ Loads a CSV file containing the centroids.

        Args:
            centroids: List of lists containing the centroids to be added.
        """
        centroids = pd.read_csv(csv_file)

        self.reset()

        for centroid in centroids.iterrows():
            pixels = centroid[1]['pixels']
            value = centroid[1]['value']
            pos_x = centroid[1]['x']
            pos_y = centroid[1]['y']

            self.add_cdpu(pos_x, pos_y, value, pixels)
