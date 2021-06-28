# -*- coding: utf-8 -*-
"""
HipparcosCatalog
=====
Hipparcos catalog is available at https://www.cosmos.esa.int/web/hipparcos.
"""

import os
from typing import Union
from time import gmtime
import pandas as pd
import numpy as np
import wget


class HipparcosCatalog:
    """
    Generates a reduced version of Hipparcos Catalogue.
    Parameters
    ----------
    target_epoch : float, default: None
        A desired epoch for catalog. If no year (julian years) is specified then 
        the catalog entries are updated to the current year.
    Attributes
    ----------
    ref_epoch : float
        Original catalog epoch (J1991.25).
    target_epoch : float
        New catalog epoch.
    """

    def __init__(self,
                 target_epoch: float = None,  # Current date (julian years)
                 ):

        self.ref_epoch = 1991.25  # Catalogue year (julian years)
        self.target_epoch = target_epoch if target_epoch is not None else self.curr_time_to_epoch()

    def undo_great_circle(self,
                          proper_motion: Union[np.float32, np.ndarray],
                          declination: Union[np.float32, np.ndarray]
                          ) -> Union[np.float32, np.ndarray]:
        """Converts the proper motion in right ascension of a celestial
           body from great circle to its original format given its declination.

        Args:
            proper_motion: Proper motion in right ascension of the celestial body (in great circle).
            declination : Declination of the celestial body (in great circle).

        Returns:
            A float/numpy.array containing the converted values of proper motion. 
        """
        secant = 1 / np.cos(declination)

        return (secant * proper_motion).astype('float32')

    def mas_to_rad(self, variable: Union[np.float32, np.ndarray]) -> Union[np.float32, np.ndarray]:
        """Converts a variable from milliarcseconds to radians.

        Args:
            variable: Values in milliarcseconds.

        Returns:
            Values in radians.
        """
        return ((variable * np.pi) / 6.48e8).astype('float32')

    def coord_at_time(self,
                      coordinate: Union[np.float32, np.ndarray],
                      proper_motion: Union[np.float32, np.ndarray],
                      delta_time: float
                      ) -> Union[np.float32, np.ndarray]:
        """Based on given proper motion and a specified delta time, returns the new coordinates.

        Args:
            coordinate: Celestial body coordinate (right ascension/declination) at some epoch.
            proper_motion: Proper motion (right ascension/declination) of the celestial body.
            delta_time: Time difference between the old epoch and the new epoch in Julian years.

        Returns:
            Coordinate at the new epoch.
        """
        return (coordinate + proper_motion * delta_time).astype('float32')

    def curr_time_to_epoch(self):
        """ Converts current time (GMT) into julian years 

        Returns:
            The current GMT date to Julian years.
        """

        # Get current day in Greenwich Mean Time
        year, month, day, hour, minute, second, _, _, _ = gmtime()

        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour,
                                 minute=minute, second=second, tz='Etc/UCT')

        julian_date = timestamp.to_julian_date()

        return ((julian_date + 13)/365.25 - 4712).astype('float32')

    def retrieve_hipparcos(self):
        """ Downloads hipparcos-1 and hipparcos-2 catalogues 
            from https://www.cosmos.esa.int/web/hipparcos.
        """

        def maybe_download(file, url):
            if os.path.exists(f"./{file}"):
                print(f"A file called '{file}' already exists in this directory")

            else:
                print(f"Downloading '{file}.gz'...")
                # Download dataset as gzip
                wget.download(url, f"./{file}.gz", bar=None)

                print(f"Extracting '{file}.gz'...")
                os.system(f"gunzip ./{file}.gz")

        maybe_download("hipparcos_v1.dat", "http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/txt.gz?I/239/hip_main.dat")
        # "http://cdsarc.u-strasbg.fr/ftp/I/239/hip_main.dat"
        maybe_download("hipparcos_v2.dat", "https://cdsarc.unistra.fr/viz-bin/nph-Cat/txt.gz?I/311/hip2.dat.gz")
        # "https://cdsarc.unistra.fr/ftp/I/311/hip2.dat.gz"

    def reduce_catalogue(self, save_reduced_catalog: bool = False) -> pd.DataFrame:
        """Reduces the catalog downloaded in 'retrieve_hipparcos' method.

        Args:
            save_reduced_catalog: If True saves a .csv file containing the the reduced catalog.

        Returns:
            A Pandas dataframe containing the reduced Hipparcos catalog. 
            The dataframe contains the following entries: 
                'HIP': Unique star identifier (hipparcos number);
                'Vmag': Star magnitude;
                'RArad': Star right ascension (in radians);
                'DErad': Star declination (in radians);
                'pmRA': Star proper motion in right ascension (in great circle);
                'pmDE':  Star proper motion in declination (in great circle);
        """

        columns_hip1 = ['Catalog',
                        'HIP',
                        'Proxy',
                        'RAhms | DEdms',
                        'Vmag',
                        'VarFlag',
                        'r_Vmag',
                        'RAdeg | DEdeg',
                        'AstroRef',
                        'sPlx',
                        'pmRA',
                        'pmDE',
                        'e_RAdeg',
                        'e_DEdeg',
                        'e_Plx',
                        'e_pmRA',
                        'e_pmDE',
                        'DE:RA',
                        'Plx:RA',
                        'Plx:DE',
                        'pmRA:RA',
                        'pmRA:DE',
                        'pmRA:Plx',
                        'pmDE:RA',
                        'pmDE:DE',
                        'pmDE:Plx',
                        'pmDE:pmRA',
                        'F1',
                        'F2 | | BTmag',
                        'e_BTmag',
                        'VTmag',
                        'e_VTmag',
                        'm_BTmag',
                        'B-V',
                        'e_B-V',
                        'r_B-V',
                        'V-I',
                        'e_V-I',
                        'r_V-I',
                        'CombMag',
                        'Hpmag',
                        'e_Hpmag',
                        'Hpscat',
                        'o_Hpmag',
                        'm_Hpmag',
                        'Hpmax',
                        'HPmin',
                        'Period',
                        'HvarType',
                        'moreVar',
                        'morePhoto',
                        'CCDM',
                        'n_CCDM',
                        'Nsys',
                        'Ncomp',
                        'MultFlag',
                        'Source',
                        'Qual',
                        'm_HIP',
                        'theta',
                        'rho',
                        'e_rho',
                        'dHp',
                        'e_dHp',
                        'Survey',
                        'Chart',
                        'Notes',
                        'HD',
                        'BD',
                        'CoD',
                        'CPD',
                        '(V-I)red',
                        'SpType',
                        'r_SpType']

        columns_hip2 = ['HIP',
                        'Sn',
                        'So',
                        'Nc',
                        'RArad DErad',
                        'Plx',
                        'pmRA',
                        'pmDE',
                        'e_RArad',
                        'e_DErad',
                        'e_Plx',
                        'e_pmRA',
                        'e_pmDE',
                        'Ntr',
                        'F2',
                        'F1',
                        'var',
                        'ic',
                        'Hpmag',
                        'e_Hpmag',
                        'sHp',
                        'VA',
                        'B-V',
                        'e_B-V',
                        'V-I',
                        'UW']

        # Loads Hipparcos-1 data table
        hip1_data = pd.read_table('hipparcos_v1.dat', sep='|', skiprows=12, skipfooter=1, names=columns_hip1, engine='python')

        hip1_data.set_index('HIP', inplace=True)

        # Loads Hipparcos-2 data table
        hip2_data = pd.read_table("hipparcos_v2.dat", sep="|", skiprows=5, skipfooter=1, names=columns_hip2, engine='python')

        hip2_data.set_index('HIP', inplace=True)

        # Create an empty catalogue
        hipparcos_reduced = pd.DataFrame(columns=['HIP', 'Vmag', 'RArad', 'DErad', 'pmRA', 'pmDE'])

        RA_DE = hip2_data['RArad DErad'].str.split(expand=True)

        # Hipparcos indices
        hipparcos_reduced['HIP'] = hip2_data.index.to_series().astype('int32')

        # Right-ascension in radians
        hipparcos_reduced['RArad'] = RA_DE[0].astype('float32')

        # Declination in radians
        hipparcos_reduced['DErad'] = RA_DE[1].astype('float32')

        # Right-ascension proper motion
        hipparcos_reduced['pmRA'] = hip2_data['pmRA'].astype('float32')

        # Declination proper motion
        hipparcos_reduced['pmDE'] = hip2_data['pmDE'].astype('float32')

        # Uses magnitude values from Hipparcos-1 as magnitude values to the new catalogue
        hipparcos_reduced['Vmag'] = hip1_data['Vmag'].loc[hip2_data.index].astype('float32')

        if save_reduced_catalog:
            hipparcos_reduced.to_csv('hipparcos_reduced.csv.gz', sep=',', compression='gzip', index=False)

        hipparcos_reduced.reset_index(drop=True, inplace=True)

        return hipparcos_reduced

    def generate_catalogue(self, file_name="curr_time_hipparcos.csv", save_catalogue=False):
        """Generates the catalog at the present epoch.

        Args:
            file_name: Name of the .csv file.
            save_catalogue: If True saves a .csv file containing the the reduced catalog.

        Returns:
            A Pandas dataframe containing the reduced Hipparcos catalog at the present epoch. 
            The dataframe contains the following entries: 
                'HIP': Unique star identifier (hipparcos number);
                'magnitude': Star magnitude;
                'right_ascension': Star right ascension (in radians);
                'declination': Star declination (in radians);
        """

        self.retrieve_hipparcos()

        data = self.reduce_catalogue()

        delta_time = self.target_epoch - self.ref_epoch

        pmRA = self.mas_to_rad(data['pmRA'].values)
        pmDE = self.mas_to_rad(data['pmDE'].values)

        pmRA = self.undo_great_circle(pmRA, data['DErad'].values)

        RA = self.coord_at_time(data['RArad'].values, pmRA, delta_time)
        DE = self.coord_at_time(data['DErad'].values, pmDE, delta_time)

        data['magnitude'] = data['Vmag']
        data['right_ascension'] = RA
        data['declination'] = DE

        data = data.drop(columns=['RArad', 'DErad', 'pmRA', 'pmDE', 'Vmag'])

        if save_catalogue:
            data.to_csv(f"{file_name}.gz", sep=',', compression='gzip', index=False)

        return data
