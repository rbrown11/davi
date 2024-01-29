###################################################################
####################### NORMALIZE N TRACKS ########################
####################################################################


# imports

import math
import os
import pandas as pd
import numpy as np
import cv2
import numbers
from PIL import Image, ImageEnhance
from matplotlib import image
from matplotlib import pyplot as plt
import colorsys
from scipy.spatial import ConvexHull, Delaunay
import davi
from time import sleep
from tqdm import tqdm


from __main__ import *

print("FIXING N_TRACKS CHANGES")


nb_ants = input(
    "How many ants were detected, i.e. n_tracks in stitch tracklets? \n")

time.sleep(2)


path = original_detections_path


data = pd.read_hdf(path)


if nb_ants == "6":
    print("nothing to do here, seems like all 6 ladies are detected!")
    data.to_hdf(os.path.join(davi_output_path, 'all_6_ants.h5'),
                key='changed_names', format="fixed")

elif nb_ants == "5":
    ant6 = data.xs('ind1', level='individuals', axis=1, drop_level=False)

    ant6_copy = pd.DataFrame().reindex(columns=ant6.columns)

    for col in ant6_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind6")

            ant6_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    final = pd.concat([data, ant6_copy], axis=0)

    final.to_hdf(os.path.join(davi_output_path, 'all_6_ants.h5'),
                 key='changed_names', format="fixed")


elif nb_ants == "4":
    ant6 = data.xs('ind1', level='individuals', axis=1, drop_level=False)

    ant6_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant5_copy = pd.DataFrame().reindex(columns=ant6.columns)

    for col in ant6_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind6")

            ant6_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant5_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind5")

            ant5_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    final = pd.concat([data, ant5_copy, ant6_copy], axis=0)

    final.to_hdf(os.path.join(davi_output_path, 'all_6_ants.h5'),
                 key='changed_names', format="fixed")


elif nb_ants == "3":
    ant6 = data.xs('ind1', level='individuals', axis=1, drop_level=False)

    ant6_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant5_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant4_copy = pd.DataFrame().reindex(columns=ant6.columns)

    for col in ant6_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind6")

            ant6_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant5_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind5")

            ant5_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant4_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind4")

            ant4_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    final = pd.concat([data, ant4_copy, ant5_copy, ant6_copy], axis=0)

    final.to_hdf(os.path.join(davi_output_path, 'all_6_ants.h5'),
                 key='changed_names', format="fixed")


elif nb_ants == "2":
    ant6 = data.xs('ind1', level='individuals', axis=1, drop_level=False)

    ant6_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant5_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant4_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant3_copy = pd.DataFrame().reindex(columns=ant6.columns)

    for col in ant6_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind6")

            ant6_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant5_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind5")

            ant5_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant4_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind4")

            ant4_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant3_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind3")

            ant3_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    final = pd.concat([data, ant3_copy, ant4_copy,
                      ant5_copy, ant6_copy], axis=0)

    final.to_hdf(os.path.join(davi_output_path, 'all_6_ants.h5'),
                 key='changed_names', format="fixed")


elif nb_ants == "1":
    ant6 = data.xs('ind1', level='individuals', axis=1, drop_level=False)

    ant6_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant5_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant4_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant3_copy = pd.DataFrame().reindex(columns=ant6.columns)
    ant2_copy = pd.DataFrame().reindex(columns=ant6.columns)

    for col in ant6_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind6")

            ant6_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant5_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind5")

            ant5_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant4_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind4")

            ant4_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant3_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind3")

            ant3_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    for col in ant2_copy.columns:
        col = list(col)

        if col[1] == "ind1":
            new_name = col[1].replace("ind1", "ind2")

            ant2_copy.rename(columns={col[0]: col[0], col[1]: new_name, col[2]: col[2], col[3]: col[3]},
                             inplace=True)

    final = pd.concat([data, ant2_copy, ant3_copy,
                      ant4_copy, ant5_copy, ant6_copy], axis=0)

    final.to_hdf(os.path.join(davi_output_path, 'all_6_ants.h5'),
                 key='changed_names', format="fixed")

else:
    print("Something is wrong here, check again how many ants you have.... (1 - 6)")

time.sleep(2)


print("The ant number has been adjusted! Now you can begin davi :)")
