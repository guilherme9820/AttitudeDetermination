from scipy.spatial.transform import Rotation
from scipy.special import erf
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import os

def spherical_to_cartesian(right_ascension, declination):

    unit_vector = np.asarray([np.cos(right_ascension)*np.cos(declination),
                              np.sin(right_ascension)*np.cos(declination),
                              np.sin(declination)])

    return unit_vector.T

class StarMapping:

    def __init__(self,
                 fov=None,
                 resolution=None,
                 mag_threshold=6,
                 roi=10, 
                 C=25000, 
                 B=0):

        self.fov = fov
        self.resolution = resolution
        self.mag_threshold = mag_threshold
        self.C = C
        self.B = B
        self.roi = roi

        self.build_projection_matrix()

    def build_projection_matrix(self):

        res_x, res_y = self.resolution

        fov = np.radians(self.fov)

        f = (res_y / (2*np.tan(fov * 0.5)))

        self.projection_matrix = np.asarray([[f, 0, res_x*0.5],
                                             [0, -f, res_y*0.5],
                                             [0,  0,         1]])

    
    def stars_within_radius(self, star_array, radius, ref_coord):

        magnitudes = star_array[:, 0]
        ra_coords = star_array[:, 1]
        de_coords = star_array[:, 2]

        # Elegible stars must be within range [ra - radius / cos(de), ra + radius / cos(de)],
        # where ra stands for right ascension
        ra_limits = [ref_coord[0] - (radius / np.cos(ref_coord[1])),
                     ref_coord[0] + (radius / np.cos(ref_coord[1]))]

        # Elegible stars must be within range [de - radius, de + radius],
        # where de stands for declination
        de_limits = [ref_coord[1] - radius, ref_coord[1] + radius]

        ra_indices_mask = np.logical_and(ra_limits[0] < ra_coords, ra_coords < ra_limits[1])
        de_indices_mask = np.logical_and(de_limits[0] < de_coords, de_coords < de_limits[1])

        indices = np.logical_and(ra_indices_mask, de_indices_mask)

        # eligible_ra = catalog['right_ascension'].between(ra_limits[0], ra_limits[1], inclusive=False)

        # eligible_de = catalog['declination'].between(de_limits[0], de_limits[1], inclusive=False)

        # Indices for those catalog entries that are within both right ascension and declination limits
        # indices = np.logical_and(eligible_ra, eligible_de)

        # mags = magnitudes[indices]
        # ra = ra_coords[indices]
        # de = de_coords[indices]

        # return (catalog.loc[indices]).reset_index(drop=True)
        # return np.vstack((eligible_mags, eligible_ra, eligible_de)).T
        return magnitudes[indices], ra_coords[indices], de_coords[indices]

    def inscribed_cube_partitioning(self, catalog):

        def _stars_within_bounds(ra_limits=[], de_limits=[]):

            right_ascension = catalog['right_ascension']
            declination = catalog['declination']

            ra_lower_bound = right_ascension > ra_limits[0]
            ra_upper_bound = right_ascension <= ra_limits[1]

            if ra_limits[0] > ra_limits[1]:
                eligible_ra = np.logical_or(ra_lower_bound, ra_upper_bound)
            else:
                eligible_ra = np.logical_and(ra_lower_bound, ra_upper_bound)

            eligible_de = declination.between(de_limits[0], de_limits[1])
                
            indices = np.logical_and(eligible_ra, eligible_de)

            return (catalog.loc[indices]).reset_index(drop=True)

        ANGLE = {-90: np.radians(-90),
                 -45: np.radians(-45),
                 0: 0,
                 45: np.radians(45),
                 90: np.radians(90),
                 135: np.radians(135),
                 225: np.radians(255),
                 315: np.radians(315),
                 360: np.radians(360)}

        S1 = _stars_within_bounds([ANGLE[315], ANGLE[45]], [ANGLE[-45], ANGLE[45]])
        S2 = _stars_within_bounds([ANGLE[45], ANGLE[135]], [ANGLE[-45], ANGLE[45]])
        S3 = _stars_within_bounds([ANGLE[135], ANGLE[225]], [ANGLE[-45], ANGLE[45]])
        S4 = _stars_within_bounds([ANGLE[225], ANGLE[315]], [ANGLE[-45], ANGLE[45]])
        S5 = _stars_within_bounds([ANGLE[0], ANGLE[360]], [ANGLE[45], ANGLE[90]])
        S6 = _stars_within_bounds([ANGLE[0], ANGLE[360]], [ANGLE[-90], ANGLE[-45]])

        partitions = (S1, S2, S3, S4, S5, S6)

        return partitions
  
    def filter_visible_stars(self, catalog):

        visible_stars_indices = np.where(catalog['magnitude'] <= self.mag_threshold)

        visible_stars = catalog.loc[visible_stars_indices]

        return visible_stars.reset_index(drop=True)

    def get_rot_quaternion(self, vector1, vector2):

        orthogonal_vector = np.cross(vector1, vector2)

        theta = np.arctan2(np.linalg.norm(orthogonal_vector), np.dot(vector1, vector2))

        orthonormal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

        angle = np.cos(theta * 0.5)

        orthonormal_vector *= np.sin(theta * 0.5)

        return np.array([orthonormal_vector[0], orthonormal_vector[1], orthonormal_vector[2], angle])

    
    def perform_rotation(self, unit_vectors, rotation_tensor):

        rotation = Rotation.from_quat(rotation_tensor)

        return rotation.apply(unit_vectors)

    
    def perform_projection(self, vectors):

        z_signs = np.sign(vectors[:, -1])

        vectors = np.asarray([vector / vector[-1] for vector in vectors])

        projections = (self.projection_matrix @ vectors.T).T

        projections[:, -1] = z_signs

        return projections

    
    def bounds_checking(self, projections):

        condition = np.logical_and.reduce((projections[:, 0] >= 0, # x >= 0
                                           projections[:, 1] >= 0, # y >= 0
                                           projections[:, 2] >= 0, # z >= 0
                                           projections[:, 0] < self.resolution[0], # x < res_x
                                           projections[:, 1] < self.resolution[1])) # y < res_y

        indices = np.where(condition)

        return projections[indices][:, :-1], indices

    def build_image(self, star_centroids, star_magnitudes, filename=None, **kwargs):

        false_star = kwargs.get('false_star', False)
        num_false_stars = kwargs.get('num_false_stars', np.random.randint(1, self.mag_threshold))
        pos_stddev = kwargs.get('pos_stddev', 0.)
        pos_mean = kwargs.get('pos_mean', 0.)
        mag_stddev = kwargs.get('mag_stddev', 0.)
        mag_mean = kwargs.get('mag_mean', 0.)
        background_stddev = kwargs.get('background_stddev', 0.)
        background_mean = kwargs.get('background_mean', 0.)

        def region_around_centroid(centroid, limit_value):

            # Defines the lower and upper bounds in the x-axis around the target star
            x_limits = np.arange(np.ceil(centroid[0] - limit_value), np.floor(centroid[0] + limit_value + 1))
            x_limits = x_limits[(x_limits >= 0) & (x_limits < self.resolution[0])].astype(np.int)

            # Defines the lower and upper bounds in the y-axis around the target star
            y_limits = np.arange(np.ceil(centroid[1] - limit_value), np.floor(centroid[1] + limit_value + 1))
            y_limits = y_limits[(y_limits >= 0) & (y_limits < self.resolution[1])].astype(np.int)

            return x_limits, y_limits

        # Additive White Gaussian Noise (AWGN)
        awgn = lambda mean_stddev, shape: np.random.normal(*mean_stddev, shape)

        if false_star:
            random_stars = (np.random.rand(num_false_stars, 1) * self.resolution[0],
                            np.random.rand(num_false_stars, 1) * self.resolution[1])
            false_stars_positions = np.concatenate(random_stars, axis=1)
            false_stars_magnitudes = np.random.randint(1, self.mag_threshold, num_false_stars)

            star_centroids = np.concatenate((star_centroids, false_stars_positions))
            star_magnitudes = np.concatenate((star_magnitudes, false_stars_magnitudes))

        # Add noise to star positions, magnitudes and to background 
        star_centroids += awgn((pos_mean, pos_stddev), star_centroids.shape)
        star_magnitudes += awgn((mag_mean, mag_stddev), star_magnitudes.shape)
        background_intensity = self.B + awgn((background_mean, background_stddev), 1)

        # Pre calculates the coefficients
        coefficients = (self.C * np.pi * 0.5) / (2.512**star_magnitudes)

        constant = 0.5*np.sqrt(2)

        # Creates a black screen with a given resolution size
        image = np.zeros(self.resolution) + background_intensity 

        for centroid, coeff in zip(star_centroids, coefficients):

            x_limits, y_limits = region_around_centroid(centroid, self.roi)

            for x in x_limits:
                for y in y_limits:

                    # Calculates distance between current pixel and the star position with added noise
                    distance_x = x - centroid[0]
                    distance_y = y - centroid[1]

                    pixel_value = coeff * ((erf(distance_x*constant) - erf((distance_x+1)*constant)) *
                                           (erf(distance_y*constant) - erf((distance_y+1)*constant)))                   

                    # Set pixel value to corresponding position
                    image[x, y] += pixel_value
        
        # Clips pixel values to range [0, 255] 
        image = np.maximum(np.minimum(image, 255), 0).astype(np.uint8)

        saving_dir = filename or f"./{datetime.now():%Y%m%d%H%M%S}.png"

        cv2.imwrite(saving_dir, image)

    def generate_image(self, catalog, position, min_number_stars, filename, **kwargs):

        star_array = catalog[['magnitude', 'right_ascension', 'declination']].values
        
        radius = np.sqrt(2*np.deg2rad(self.fov)**2) * 0.5

        stars_mags, stars_ra, stars_de = self.stars_within_radius(star_array, radius, position)

        unit_ref_coords = spherical_to_cartesian(*position)  

        quaternion = self.get_rot_quaternion(unit_ref_coords, (0, 0, 1))

        stars_coords = spherical_to_cartesian(stars_ra, stars_de)

        rotated_stars = self.perform_rotation(stars_coords, quaternion)

        projections = self.perform_projection(rotated_stars)

        projections, indices = self.bounds_checking(projections)

        if len(projections) >= min_number_stars:

            # stars_magnitudes = stars_in_fov['magnitude'].loc[indices].values

            self.build_image(projections, stars_mags[indices], filename=filename, **kwargs)

    def __str__(self):

        return f"Current Configuration:\n\n\
                 \r-> Field of View (degrees): {self.fov}\n\
                 \r-> Image resolution (px): {self.resolution}\n\
                 \r-> Magnitude threshold: {self.mag_threshold}\n\
                 \r-> C: {self.C}\n\
                 \r-> B: {self.B}\n\
                 \r-> Maximum star radius (px): {self.roi}"

class StarSubnet(StarMapping):

    def __init__(self,
                 fov=None,
                 resolution=None,
                 mag_threshold=6,
                 roi=10, 
                 C=25000, 
                 B=0):
        super().__init__(fov=fov,
                         resolution=resolution,
                         mag_threshold=mag_threshold,
                         roi=roi, 
                         C=C, 
                         B=B)

    def build_triplet_features(self, catalog, global_fov, triplet_fov):

        global_fov = np.radians(global_fov * 0.5)
        triplet_fov = np.radians(triplet_fov)

        visible_stars = self.filter_visible_stars(catalog)

        star_array = visible_stars[['magnitude', 'right_ascension', 'declination']].values

        for ref_coord in star_array[:, 1:]:

            stars_mags, stars_ra, stars_de = self.stars_within_radius(star_array, global_fov, ref_coord)

            ref_unit_vector = spherical_to_cartesian(*ref_coord)  

            stars_unit_vectors = spherical_to_cartesian(stars_ra, stars_de)

            quaternion = self.get_rot_quaternion(ref_unit_vector, (0, 0, 1))

            rotated_stars = self.perform_rotation(stars_unit_vectors, quaternion)

            projections = self.perform_projection(rotated_stars)

            projections, indices = self.bounds_checking(projections)

            fov_array = np.vstack((stars_mags, stars_ra, stars_de)).T

            stars_mags, stars_ra, stars_de = self.stars_within_radius(fov_array, triplet_fov, ref_coord)

            neighbours, adj_distances = self.get_closest_neighbours(ref_unit_vector, stars_unit_vectors, k_neighbours=2)

            opp_distance = np.linalg.norm(neighbours[0] - neighbours[1])

            # if len(projections) >= min_number_stars:

            #     # stars_magnitudes = stars_in_fov['magnitude'].loc[indices].values

            #     self.build_image(projections, stars_mags[indices], filename=filename, **kwargs)

    def get_closest_neighbours(self, main_star, neighbours, k_neighbours=2):

        # Measures the Euclidean distance between each star and the main star
        distances = np.linalg.norm(main_star - neighbours, axis=1)

        # Gets the indices that sort the array
        sorting_indices = distances.argsort()

        sorted_neighbours = neighbours[sorting_indices]

        sorted_distances = distances[sorting_indices]

        # Gets the k closest neighbours and their distances. The first entry is
        # ignored because it corresponds to the main star
        if k_neighbours == -1:
            # Gets all neighbours
            closest_neighbours = sorted_neighbours[1:]
            closest_distances = sorted_distances[1:]
        else:
            closest_neighbours = sorted_neighbours[1:1 + k_neighbours]
            closest_distances = sorted_distances[1:1 + k_neighbours]

        return closest_neighbours, closest_distances

    def test_body_vectors(self, catalog, fov, ref_coord):

        star_array = catalog[['magnitude', 'right_ascension', 'declination']].values
        
        radius = np.deg2rad(fov * 0.5)

        stars_mags, stars_ra, stars_de = self.stars_within_radius(star_array, radius, ref_coord)

        ref_unit_vector = spherical_to_cartesian(*ref_coord)  

        stars_unit_vector = spherical_to_cartesian(stars_ra, stars_de)

        quaternion = self.get_rot_quaternion(ref_unit_vector, (0, 0, 1))

        rotated_stars = self.perform_rotation(stars_unit_vector, quaternion)

        projections = self.perform_projection(rotated_stars)

        projections, indices = self.bounds_checking(projections)

        optical_center = [self.resolution[0] / 2, self.resolution[1] / 2]

        focal_length = self.resolution[0] / (2 * np.tan(fov * 0.5))

        body_vectors = self.body_vectors_from_centroids(projections, optical_center, focal_length)

        print(rotated_stars[indices], '\n',  body_vectors)

    def body_vectors_from_centroids(self, centroids, optical_center, focal_length):
        """ Transforms the star centroids to unit vectors referenced by the
            star sensor coordinate frame. 

            Implementation based on equation (1) present in the paper
            "Accuracy performance of star trackers - a tutorial (https://ieeexplore.ieee.org/document/1008988)"

            Args:
                centroids: An array containing the star centroids (pixel coordinate).
                optical_center: Intersection of the focal plane and the optical axis.
                focal_length: Star sensor focal distance.
            
            Returns:
                Unit vectors in the star sensor coordinate frame corresponding to each
                centroid given as argument.
        """

        diffs = centroids - optical_center

        atan = np.arctan(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) / focal_length)
        atan2 = np.arctan2(diffs[:, 1], diffs[:, 0])

        body_vectors = np.asarray([np.cos(atan2) * np.cos(0.5 * np.pi - atan),
                                   np.sin(atan2) * np.cos(0.5 * np.pi - atan),
                                   np.sin(0.5 * np.pi - atan)])

        return body_vectors.T

if __name__ == "__main__":

    catalog = pd.read_csv("curr_time_hipparcos.csv")

    star_map = StarSubnet(8, [256, 256], 6)

    visible_stars = star_map.filter_visible_stars(catalog)
    visible_stars = visible_stars.set_index('HIP')

    ra, de = visible_stars[['right_ascension', 'declination']].loc[18744]

    star_map.test_body_vectors(visible_stars, 8, [ra, de])

    star_map.generate_image(visible_stars, [ra, de], 10, 'test_image.png')