from scipy.spatial.transform import Rotation
from scipy.special import erf
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
from .utils import get_k_top_centroids
from .utils import get_rot_quaternion
from .utils import rotate_vectors
from .utils import project_onto_image_plane
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

        # Circle equation is: (x - x0)^2 + (y - y0)^2 <= radius^2. So we must get all coordinates
        # such that their squared sum lies within [buffer_radius**2, pattern_radius**2].
        circle_radius = delta_ra**2 + delta_de**2

        upper_cond = circle_radius <= pattern_radius**2

        lower_cond = buffer_radius**2 <= circle_radius

        valid_indices = np.where(np.logical_and(upper_cond, lower_cond))

        return catalog.iloc[valid_indices].reset_index(drop=True)

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
        image = np.clip(image, 0, 255).astype(np.uint8)

        if save_image:
            saving_dir = filename or f"./{datetime.now():%Y%m%d%H%M%S}.png"

            cv2.imwrite(saving_dir, image)

        return image

    def project_stars(self, stars, ref_star, radius):
        """ This function projects the stars around the reference star onto an image plane.
            It only projects the stars within a given radius.

        Args:
            stars: A dataframe containing the stars informations. Two columns are mandatory:
                   'right_ascension' and 'declination'.
            ref_star: The coordinates of the main star [right ascension, declination].
            radius: The radius around the reference star.

        Returns:
            A dataframe containing row and column positions for each star in the input dataframe.
        """

        valid_stars = self.stars_within_radius(stars, ref_star, radius)

        ref_unit_vectors = equatorial_to_cartesian(*ref_star)

        stars_unit_vectors = equatorial_to_cartesian(valid_stars['right_ascension'].values,
                                                     valid_stars['declination'].values)

        quaternion = get_rot_quaternion(ref_unit_vectors, (0, 0, 1))

        rotated_stars = rotate_vectors(stars_unit_vectors, quaternion)

        valid_projections, valid_indices = project_onto_image_plane(rotated_stars, self.projection_matrix, self.resolution)

        positions = pd.DataFrame(data=valid_projections, columns=['row', 'column'])

        valid_stars = valid_stars.iloc[valid_indices].reset_index(drop=True)

        return pd.concat([valid_stars, positions], axis=1, sort=False)

    def generate_image(self,
                       catalog,
                       reference,
                       min_number_stars,
                       filename=None,
                       save_image=False,
                       **kwargs):

        radius = np.radians(self.fov * 0.5)

        projections = self.project_stars(catalog, reference, radius)

        if len(projections) >= min_number_stars:
            star_centroids = projections[['row', 'column']].values
            magnitudes = projections['magnitude'].values
            return self.build_image(star_centroids,
                                    magnitudes,
                                    filename=filename,
                                    save_image=save_image,
                                    **kwargs)

        return None

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
