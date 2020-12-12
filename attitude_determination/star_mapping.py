from scipy.spatial.transform import Rotation
from scipy.special import erf
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
from .utils import get_k_top_centroids
from .utils import get_rot_quaternion
from .utils import perform_rotation
from .utils import perform_projection
from .utils import equatorial_to_cartesian


class StarMapping:

    def __init__(self,
                 fov=None,
                 resolution=None,
                 pixel_size=None,
                 focal_length=None,
                 mag_threshold=6,
                 roi=10,
                 C=25000,
                 B=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.fov = fov
        self.resolution = np.array(resolution)
        self.optical_center = self.resolution / 2
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.mag_threshold = mag_threshold
        self.C = C
        self.B = B
        self.roi = roi

        self.build_projection_matrix()

    def build_projection_matrix(self):

        res_x, res_y = self.resolution

        fov = np.radians(self.fov)

        if self.pixel_size and self.focal_length:
            self.focal_distance = self.focal_length / self.pixel_size
        else:
            # Estimates the focal distance based on the image
            # resolution and the specified field of view
            self.focal_distance = (res_y / (2 * np.tan(fov * 0.5)))

        self.projection_matrix = np.asarray([[self.focal_distance, 0, res_x * 0.5],
                                             [0, self.focal_distance, res_y * 0.5],
                                             [0, 0, 1]])

    def stars_within_radius(self, catalog, ref_star, pattern_radius, buffer_radius=0):
        """Retrieves the stars within a circular radius around a reference star.

        Args:
            catalog (pandas.DataFrame): A dataframe containing the stars informations [HIP, magnitude, right ascension, declination].
            ref_star (Union[list, numpy.ndarray]): The coordinates of the main star [right ascension, declination].
            pattern_radius (float): Maximum radius distance from main star (in radians).
            buffer_radius (float, optional): Minimum radius distance from main star (in radians). Defaults to 0.

        Returns:
            pandas.DataFrame: A reduced dataframe containing only the stars within a radius from the main star.
                              The returned dataframe has the same structure of the catalog argument.
        """

        ra_coords = catalog['right_ascension'].values

        de_coords = catalog['declination'].values

        # Converts right ascension back to great-circle
        delta_ra = ra_coords * np.cos(de_coords) - ref_star[0] * np.cos(ref_star[1])

        delta_de = de_coords - ref_star[1]

        # Circle equation is: (x - x0)^2 + (y - y0)^2 <= radius^2. So we must get all coordinates s.t.
        # their squared sum lies within [buffer_radius**2, pattern_radius**2].
        circle_radius = delta_ra**2 + delta_de**2

        upper_cond = circle_radius <= pattern_radius**2

        lower_cond = buffer_radius**2 <= circle_radius

        valid_indices = np.where(np.logical_and(upper_cond, lower_cond))

        return catalog.iloc[valid_indices]

    # def stars_within_radius(self, star_array, radius, ref_coord):

    #     magnitudes = star_array[:, 0]
    #     ra_coords = star_array[:, 1]
    #     de_coords = star_array[:, 2]

    #     # Elegible stars must be within range [ra - radius / cos(de), ra + radius / cos(de)],
    #     # where ra stands for right ascension
    #     ra_limits = [ref_coord[0] - (radius / np.cos(ref_coord[1])),
    #                  ref_coord[0] + (radius / np.cos(ref_coord[1]))]

    #     # Elegible stars must be within range [de - radius, de + radius],
    #     # where de stands for declination
    #     de_limits = [ref_coord[1] - radius, ref_coord[1] + radius]

    #     ra_indices_mask = np.logical_and(ra_limits[0] < ra_coords, ra_coords < ra_limits[1])
    #     de_indices_mask = np.logical_and(de_limits[0] < de_coords, de_coords < de_limits[1])

    #     indices = np.logical_and(ra_indices_mask, de_indices_mask)

    #     # eligible_ra = catalog['right_ascension'].between(ra_limits[0], ra_limits[1], inclusive=False)

    #     # eligible_de = catalog['declination'].between(de_limits[0], de_limits[1], inclusive=False)

    #     # Indices for those catalog entries that are within both right ascension and declination limits
    #     # indices = np.logical_and(eligible_ra, eligible_de)

    #     # mags = magnitudes[indices]
    #     # ra = ra_coords[indices]
    #     # de = de_coords[indices]

    #     # return (catalog.loc[indices]).reset_index(drop=True)
    #     # return np.vstack((eligible_mags, eligible_ra, eligible_de)).T
    #     return magnitudes[indices], ra_coords[indices], de_coords[indices]

    def filter_visible_stars(self, catalog):

        visible_stars_indices = np.where(catalog['magnitude'] <= self.mag_threshold)

        visible_stars = catalog.loc[visible_stars_indices]

        return visible_stars.reset_index(drop=True)

    def region_around_centroid(self, centroid, limit_value):

        # Defines the lower and upper bounds in the x-axis around the target star
        x_limits = np.arange(np.ceil(centroid[0] - limit_value), np.floor(centroid[0] + limit_value + 1))
        x_limits = x_limits[(x_limits >= 0) & (x_limits < self.resolution[0])].astype(np.int)

        # Defines the lower and upper bounds in the y-axis around the target star
        y_limits = np.arange(np.ceil(centroid[1] - limit_value), np.floor(centroid[1] + limit_value + 1))
        y_limits = y_limits[(y_limits >= 0) & (y_limits < self.resolution[1])].astype(np.int)

        return x_limits, y_limits

    def generate_false_stars(self, num_false_stars):
        random_stars = (np.random.rand(num_false_stars, 1) * self.resolution[0],
                        np.random.rand(num_false_stars, 1) * self.resolution[1])
        false_stars_positions = np.concatenate(random_stars, axis=1)
        false_stars_magnitudes = np.random.randint(1, self.mag_threshold, num_false_stars)

        return false_stars_positions, false_stars_magnitudes

    def awgn(self, mean_stddev, shape):
        # Additive White Gaussian Noise (AWGN)
        return np.random.normal(*mean_stddev, shape)

    def build_image(self, star_centroids, star_magnitudes, filename=None, save_image=True, **kwargs):

        false_star = kwargs.get('false_star', False)
        num_false_stars = kwargs.get('num_false_stars', np.random.randint(1, self.mag_threshold))
        pos_stddev = kwargs.get('pos_stddev', 0.)
        pos_mean = kwargs.get('pos_mean', 0.)
        mag_stddev = kwargs.get('mag_stddev', 0.)
        mag_mean = kwargs.get('mag_mean', 0.)
        background_stddev = kwargs.get('background_stddev', 0.)
        background_mean = kwargs.get('background_mean', 0.)

        if false_star:
            false_stars_positions, false_stars_magnitudes = self.generate_false_stars(num_false_stars)

            star_centroids = np.concatenate((star_centroids, false_stars_positions))
            star_magnitudes = np.concatenate((star_magnitudes, false_stars_magnitudes))

        # Add noise to star positions, magnitudes and to background
        star_centroids += self.awgn((pos_mean, pos_stddev), star_centroids.shape)
        star_magnitudes += self.awgn((mag_mean, mag_stddev), star_magnitudes.shape)
        background_intensity = self.B + self.awgn((background_mean, background_stddev), 1)

        # Pre calculates the coefficients
        coefficients = (self.C * np.pi * 0.5) / (2.512**star_magnitudes)

        constant = 0.5 * np.sqrt(2)

        # Creates a black screen with a given resolution size
        image = np.zeros(self.resolution) + background_intensity

        for centroid, coeff in zip(star_centroids, coefficients):

            x_limits, y_limits = self.region_around_centroid(centroid, self.roi)

            for x_coord in x_limits:
                for y_coord in y_limits:

                    # Calculates distance between current pixel and the star position with added noise
                    distance_x = x_coord - centroid[0]
                    distance_y = y_coord - centroid[1]

                    pixel_value = coeff * ((erf(distance_x * constant) - erf((distance_x + 1) * constant)) *
                                           (erf(distance_y * constant) - erf((distance_y + 1) * constant)))

                    # Set pixel value to corresponding position
                    image[x_coord, y_coord] += pixel_value

        # Clips pixel values to range [0, 255]
        # image = np.maximum(np.minimum(image, 255), 0).astype(np.uint8)
        image = np.clip(image, 0, 255).astype(np.uint8)

        if save_image:
            saving_dir = filename or f"./{datetime.now():%Y%m%d%H%M%S}.png"

            cv2.imwrite(saving_dir, image)

        return image

    def generate_image(self, catalog, position, min_number_stars, filename, **kwargs):

        star_array = catalog[['magnitude', 'right_ascension', 'declination']].values

        radius = np.radians(self.fov * 0.5)

        valid_stars = self.stars_within_radius(star_array, radius, position)

        unit_ref_coords = equatorial_to_cartesian(*position)

        quaternion = get_rot_quaternion(unit_ref_coords, (0, 0, 1))

        stars_coords = equatorial_to_cartesian(valid_stars[:, 1], valid_stars[:, 2])

        rotated_stars = perform_rotation(stars_coords, quaternion)

        projections, indices = perform_projection(rotated_stars, self.projection_matrix, self.resolution)

        if len(projections) >= min_number_stars:

            self.build_image(projections, valid_stars[indices, 0], filename=filename, **kwargs)

    def __str__(self):

        return f"Current Configuration:\n\n\
                 \r-> Field of View (degrees): {self.fov}\n\
                 \r-> Image resolution (px): {self.resolution}\n\
                 \r-> Optical center (px): {self.optical_center}\n\
                 \r-> Pixel size (mm): {self.pixel_size}\n\
                 \r-> Focal length (mm): {self.focal_length}\n\
                 \r-> Magnitude threshold: {self.mag_threshold}\n\
                 \r-> C: {self.C}\n\
                 \r-> B: {self.B}\n\
                 \r-> Maximum star radius (px): {self.roi}"