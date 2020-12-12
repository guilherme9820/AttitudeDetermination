from scipy.spatial.transform import Rotation
import numpy as np
from .cest import get_star_pixels
from .cest import Centroider


def equatorial_to_cartesian(right_ascension, declination):
    """ Converts from equatorial coordinate system to cartesian coordinate system.
        This function considers that the vector in equatorial coordinate is a unit vector.

        Args:
            right_ascension (numpy.ndarray): Array of shape (N,) where N is the number of samples and each sample has
                             the azimuth angles to be converted.
            declination  (numpy.ndarray): Array of shape (N,) where N is the number of samples and each sample has
                         the elevation angles to be converted.

        Returns:
            numpy.ndarray: An array of shape (N, 3), where each sample has three components: x, y and z.
    """

    unit_vector = np.asarray([np.cos(right_ascension) * np.cos(declination),
                              np.sin(right_ascension) * np.cos(declination),
                              np.sin(declination)])

    return unit_vector.T


def cartesian_to_equatorial(x_coord, y_coord, z_coord):
    """ Converts from cartesian coordinate system to equatorial coordinate system.
        This function considers that the vector in coordinate coordinate is a unit vector.

        Args:
            x_coord (numpy.ndarray): Array of shape (N,) where N is the number of samples.
            y_coord (numpy.ndarray): Array of shape (N,) where N is the number of samples.
            z_coord (numpy.ndarray): Array of shape (N,) where N is the number of samples.

        Returns:
            numpy.ndarray: An array of shape (N, 2), where each sample has two components: right ascension and declination.
    """

    right_ascension = np.arctan(y_coord / x_coord)

    declination = np.arcsin(z_coord)

    x_sign = np.sign(x_coord)
    y_sign = np.sign(y_coord)

    right_ascension = np.where(x_sign == -1., np.pi + right_ascension, right_ascension)

    right_ascension = np.where(np.logical_and(x_sign == 1., y_sign == -1.), 2 * np.pi + right_ascension, right_ascension)

    unit_vector = np.asarray([right_ascension, declination])

    return unit_vector.T


def body_vectors_from_centroids(centroids, optical_center, focal_distance):
    """ Transforms the star centroids to unit vectors referenced by the
        star sensor coordinate frame.

        Implementation based on equation (1) present in the paper
        "Accuracy performance of star trackers - a tutorial ( https://ieeexplore.ieee.org/document/1008988 )"

        Args:
            centroids (Union[list, numpy.ndarray]): An array containing the star centroids (pixel coordinate).
            optical_center (Union[list, numpy.ndarray]): Intersection of the focal plane and the optical axis (pixel coordinate).
            focal_distance (float): Star sensor focal distance.

        Returns:
            numpy.ndarray: Unit vectors in the star sensor coordinate frame
                           corresponding to each centroid given as argument.
    """

    centroids = np.array(centroids)
    optical_center = np.array(optical_center)

    if centroids.ndim < 3:
        centroids = centroids[np.newaxis, :, :]

    diffs = centroids - optical_center

    atan = np.arctan(np.sqrt(diffs[..., :-1]**2 + diffs[..., -1:]**2) / focal_distance)
    atan2 = np.arctan2(diffs[..., -1:], diffs[..., :-1])

    body_vectors = np.asarray([np.cos(atan2) * np.cos(0.5 * np.pi - atan),
                               np.sin(atan2) * np.cos(0.5 * np.pi - atan),
                               np.sin(0.5 * np.pi - atan)])

    body_vectors = np.transpose(body_vectors, [1, 2, 0, 3])

    return body_vectors.squeeze().astype('float32')


def ref_vectors_from_catalog(catalog, identifiers):

    dataframe = catalog.set_index('HIP')

    ref_stars = dataframe.loc[identifiers]

    right_ascension = ref_stars['right_ascension'].values
    declination = ref_stars['declination'].values

    return equatorial_to_cartesian(right_ascension, declination)


def get_k_top_centroids(star_image, k_top=4):

    star_pixels = get_star_pixels(star_image, 150)

    centroider = Centroider(max_cdpus=60)

    centroids = centroider.compute_from_list(star_pixels, 0.8)

    centroids = sorted(centroids, key=lambda x: x.pixels, reverse=True)

    if k_top == -1:
        # Retrieves all centroids
        top_centroids = centroids
    else:
        top_centroids = centroids[:k_top]

    return [[centroid.pos_y, centroid.pos_x] for centroid in top_centroids]


def get_rot_quaternion(vector1, vector2):

    orthogonal_vector = np.cross(vector1, vector2)

    theta = np.arctan2(np.linalg.norm(orthogonal_vector),
                       np.dot(vector1, vector2))

    orthonormal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    angle = np.cos(theta * 0.5)

    orthonormal_vector *= np.sin(theta * 0.5)

    return np.array([orthonormal_vector[0], orthonormal_vector[1], orthonormal_vector[2], angle])


def perform_rotation(unit_vectors, rotation_tensor, representation='quaternion'):

    if representation == 'quaternion':
        rotation = Rotation.from_quat(rotation_tensor)
    elif representation == 'matrix':
        rotation = Rotation.from_matrix(rotation_tensor)

    return rotation.apply(unit_vectors)


def perform_projection(vectors, projection_matrix, image_resolution, return_indices=True):

    projections = (projection_matrix @ vectors.T).T

    projections = projections / np.expand_dims(vectors[:, -1], axis=1)

    # After projecting the stars coordinates from body frame of reference to an image plane
    # its necessary to validate its boundaries, because some stars may fall out the image
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
