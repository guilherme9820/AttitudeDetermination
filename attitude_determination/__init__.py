import os
from pkg_resources import get_distribution, DistributionNotFound


def get_version():
    """ Returns the version of the package defined in setup.py.

    Raises:
        DistributionNotFound: It is raised when the package is not installed yet.

    Returns:
        The package version.
    """
    package_name = os.path.basename(os.path.dirname(__file__))

    try:
        _dist = get_distribution(package_name)
        # Normalize case for Windows systems
        dist_loc = os.path.normcase(_dist.location)
        here = os.path.normcase(__file__)
        if not here.startswith(os.path.join(dist_loc, package_name)):
            # not installed, but there is another version that *is*
            raise DistributionNotFound
    except DistributionNotFound:
        return 'Please install this project with setup.py'
    else:
        return _dist.version


__version__ = get_version()
