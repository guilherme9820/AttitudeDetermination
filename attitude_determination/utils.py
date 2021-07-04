from typing import Union
from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd
from .cest import get_star_pixels
from .cest import Centroider


def equatorial_to_cartesian(right_ascension: np.ndarray, declination: np.ndarray) -> np.ndarray:
    """ Converts from equatorial coordinate system to cartesian coordinate system.
        This function considers that the vector in equatorial coordinate is a unit vector.

        Args:
            right_ascension: Array of shape (N,) where N is the number of samples and each sample has
                             the azimuth angles to be converted.
            declination: Array of shape (N,) where N is the number of samples and each sample has
                         the elevation angles to be converted.

        Returns:
            An array of shape (N, 3), where each sample has three components: x, y and z.
    """

    unit_vector = np.asarray([np.cos(right_ascension) * np.cos(declination),
                              np.sin(right_ascension) * np.cos(declination),
                              np.sin(declination)])

    return unit_vector.T


def cartesian_to_equatorial(x_coord: np.ndarray, y_coord: np.ndarray, z_coord: np.ndarray) -> np.ndarray:
    """ Converts from cartesian coordinate system to equatorial coordinate system.
        This function considers that the vector in cartesian coordinate is a unit vector.

        Args:
            x_coord: Array of shape (N,) where N is the number of samples.
            y_coord: Array of shape (N,) where N is the number of samples.
            z_coord: Array of shape (N,) where N is the number of samples.

        Returns:
            An array of shape (N, 2), where each sample has two components: right ascension and declination.
    """

    right_ascension = np.arctan(y_coord / x_coord)

    declination = np.arcsin(z_coord)

    x_sign = np.sign(x_coord)
    y_sign = np.sign(y_coord)

    right_ascension = np.where(x_sign == -1., np.pi + right_ascension, right_ascension)

    right_ascension = np.where(np.logical_and(x_sign == 1., y_sign == -1.), 2 * np.pi + right_ascension, right_ascension)

    unit_vector = np.asarray([right_ascension, declination])

    return unit_vector.T


def body_vectors_from_centroids(centroids: Union[list, np.ndarray],
                                optical_center: Union[list, np.ndarray],
                                focal_distance: float) -> np.ndarray:
    """ Transforms the star centroids to unit vectors referenced by the
        star sensor coordinate frame.

        Implementation based on equation (1) present in the paper
        "Accuracy performance of star trackers - a tutorial ( https://ieeexplore.ieee.org/document/1008988 )"

        Args:
            centroids: An array containing the star centroids (pixel coordinate).
            optical_center: Intersection of the focal plane and the optical axis (pixel coordinate).
            focal_distance: Star sensor focal distance.

        Returns:
            Unit vectors in the star sensor coordinate frame corresponding to each centroid.
    """

    centroids = np.array(centroids)
    optical_center = np.array(optical_center)

    diffs = centroids - optical_center

    atan = np.arctan(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) / focal_distance)
    atan2 = np.arctan2(diffs[:, 0], diffs[:, 1])

    body_vectors = np.asarray([np.cos(atan2) * np.cos(0.5 * np.pi - atan),
                               np.sin(atan2) * np.cos(0.5 * np.pi - atan),
                               np.sin(0.5 * np.pi - atan)])

    return body_vectors.T.astype('float32')


def ref_vectors_from_catalog(catalog: pd.DataFrame, identifiers: Union[np.ndarray, list]) -> np.ndarray:
    """ Returns a collection of unit vectors from a star catalog given their IDs.

        Args:
            catalog: A pandas DataFrame containing the stars.
            identifiers: The star IDs.

        Returns:
            Unit vectors in the reference coordinate frame.
    """

    dataframe = catalog.set_index('HIP')

    ref_stars = dataframe.loc[identifiers]

    right_ascension = ref_stars['right_ascension'].values
    declination = ref_stars['declination'].values

    return equatorial_to_cartesian(right_ascension, declination)


def get_k_top_centroids(star_image: np.ndarray, k_top: int = 4, location_only: bool = True) -> list:
    """ Evaluates the centroids in a star image.

        Args:
            star_image: A numpy 2-D array containing the pixel values.
            k_top: The number of centroids in descending order of relevance. Defaults to 4.
            location_only: Returns only the centroids positions if set to true.
                           Returns a list of Centroid objects otherwise. Defaults to true.
        Returns:
            A list containing the most relevant centroids.
    """

    star_pixels = get_star_pixels(star_image, threshold=150)

    centroider = Centroider(max_cdpus=60)

    centroids = centroider.compute_from_list(star_pixels, 0.8)

    centroids = sorted(centroids, key=lambda x: x.pixels, reverse=True)

    if k_top == -1:
        # Retrieves all centroids
        top_centroids = centroids
    else:
        top_centroids = centroids[:k_top]

    if location_only:
        return [[centroid.pos_y, centroid.pos_x] for centroid in top_centroids]

    return top_centroids


def get_rot_quaternion(vector1, vector2):

    orthogonal_vector = np.cross(vector1, vector2)

    theta = np.arctan2(np.linalg.norm(orthogonal_vector),
                       np.dot(vector1, vector2))

    orthonormal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    angle = np.cos(theta * 0.5)

    orthonormal_vector *= np.sin(theta * 0.5)

    return np.array([orthonormal_vector[0], orthonormal_vector[1], orthonormal_vector[2], angle])


def rotate_vectors(unit_vectors, rotation_tensor, representation='quaternion'):
    """ Applies a rotation operator over a collection of vectors.

    Args:
        unit_vectors: An array of shape (N, 3) where each sample in N is a unit vector.
        rotation_tensor: The rotation operation. It can be either an array of shape (3, 3) or an array
                         of shape (4).
        representation (optional): This argument defines which rotation parametrization will be used.
                                   It can be 'dcm' or 'quaternion', where the former stands for Direction
                                   Consine Matrix. Defaults to 'quaternion'.

    Returns:
        An array of shape (N, 3) containing the rotated unit vectors.
    """

    if representation == 'quaternion':
        rotation = Rotation.from_quat(rotation_tensor)
    elif representation == 'matrix':
        rotation = Rotation.from_matrix(rotation_tensor)

    return rotation.apply(unit_vectors)


def project_onto_image_plane(vectors, projection_matrix, image_resolution, return_indices=True):
    """ Projects a collection of vectors onto an image plane given its projection matrix.

    Args:
        vectors: An array of shape (N, 3).
        projection_matrix: An array of shape (3, 3).
        image_resolution: A 1-D array containing the image witdh and height, in that order.
        return_indices (optional): If True, it will return the indices of the vectors that were
                                   projected properly. Defaults to True.

    Returns:
        An array of shape (N, 2) containing the x and y coordiantes of the image plane. Returns
        an array containing the valid indices if 'return_indices' was set to True. 
    """
    projections = (projection_matrix @ vectors.T).T

    projections = projections / np.expand_dims(vectors[:, -1], axis=1)

    # After projecting the stars coordinates from the body frame of reference to an image plane
    # it's necessary to validate its boundaries, because some stars may fall out the image
    # resolution range.
    condition = np.logical_and.reduce((projections[:, 0] >= 0,  # x >= 0
                                       projections[:, 1] >= 0,  # y >= 0
                                       projections[:, 2] >= 0,  # z >= 0
                                       projections[:, 0] < image_resolution[0],  # x < res_x
                                       projections[:, 1] < image_resolution[1]))  # y < res_y

    valid_indices = np.where(condition)

    if return_indices:
        return projections[valid_indices][:, :-1], valid_indices

    return projections[valid_indices][:, :-1]


def inscribed_cube_partitioning(catalog):

    def _stars_within_bounds(ra_limits, de_limits):

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
