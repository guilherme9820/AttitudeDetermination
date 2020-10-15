import numpy as np


def equatorial_to_cartesian(right_ascension, declination):
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


def cartesian_to_equatorial(x, y, z):
    """ Converts from cartesian coordinate system to equatorial coordinate system.
        This function considers that the vector in coordinate coordinate is a unit vector.

        Args:
            x: Array of shape (N,) where N is the number of samples.
            y: Array of shape (N,) where N is the number of samples.
            z: Array of shape (N,) where N is the number of samples.

        Returns:
            An array of shape (N, 2), where each sample has two components: right ascension and declination.
    """

    right_ascension = np.arctan(y / x)

    declination = np.arcsin(z)

    x_sign = np.sign(x)
    y_sign = np.sign(y)

    right_ascension = np.where(x_sign == -1., np.pi + right_ascension, right_ascension)

    right_ascension = np.where(np.logical_and(x_sign == 1., y_sign == -1.), 2 * np.pi + right_ascension, right_ascension)

    unit_vector = np.asarray([right_ascension, declination])

    return unit_vector.T


def body_vectors_from_centroids(centroids, optical_center, focal_distance):
    """ Transforms the star centroids to unit vectors referenced by the
        star sensor coordinate frame.

        Implementation based on equation (1) present in the paper
        "Accuracy performance of star trackers - a tutorial (https://ieeexplore.ieee.org/document/1008988)"

        Args:
            centroids: An array containing the star centroids (pixel coordinate).
            optical_center: Intersection of the focal plane and the optical axis (pixel coordinate).
            focal_distance: Star sensor focal distance.

        Returns:
            Unit vectors in the star sensor coordinate frame corresponding to each
            centroid given as argument.
    """

    diffs = centroids - optical_center

    atan = np.arctan(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) / focal_distance)
    atan2 = np.arctan2(diffs[:, 1], diffs[:, 0])

    body_vectors = np.asarray([np.cos(atan2) * np.cos(0.5 * np.pi - atan),
                               np.sin(atan2) * np.cos(0.5 * np.pi - atan),
                               np.sin(0.5 * np.pi - atan)])

    return body_vectors.T
