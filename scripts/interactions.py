
###########################################################################
######################### INTERACTION EXTRACTION ##########################
###########################################################################


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
import itertools
from itertools import groupby
from operator import itemgetter

from __main__ import *


def remove_dupl_list(x):

    return list(dict.fromkeys(x))


detections_path = input(
    "first of all, what is the path to the detections? \n")

vidname_path = input("aaand what is the path to the original video? \n")

vidname = os.path.basename(vidname_path.strip('.mp4'))

data = pd.read_hdf(detections_path)


nb_ants = input("aaaaand how many ants were tracked? Either 6 or 11 pls :) \n")

if nb_ants == "6":
    ants = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']


elif nb_ants == "11":
    ants = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5',
            'ind6', 'ind7', 'ind8', 'ind9', 'ind10', 'ind11']


for x, y in itertools.combinations(ants, 2):

    overlapping_frames = []

    for i in tqdm(range(0, len(data))):

        abdomen_x_x = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', x, 'abdomen', 'x')][i]
        abdomen_y_x = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', x, 'abdomen', 'y')][i]
        thorax_x_x = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', x, 'thorax', 'x')][i]
        thorax_y_x = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', x, 'thorax', 'y')][i]
        head_x_x = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', x, 'head', 'x')][i]
        head_y_x = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', x, 'head', 'y')][i]

        abdomen_x_y = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', y, 'abdomen', 'x')][i]
        abdomen_y_y = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', y, 'abdomen', 'y')][i]
        thorax_x_y = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', y, 'thorax', 'x')][i]
        thorax_y_y = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', y, 'thorax', 'y')][i]
        head_x_y = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', y, 'head', 'x')][i]
        head_y_y = data[(
            'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', y, 'head', 'y')][i]

        # first comparisons

        thorax_distance_apart = davi.find_distance(
            thorax_x_x, thorax_x_y, thorax_y_x, thorax_y_y)

        if thorax_distance_apart < 200:
            if math.isnan(head_x_x) or math.isnan(head_y_y):
                continue
            else:

                # x bounding boxes

                x_theta_thorax_head = davi.get_vector_angle(
                    thorax_x_x, thorax_y_x, head_x_x, head_y_x)
                x_theta_abdomen_thorax = davi.get_vector_angle(
                    abdomen_x_x, abdomen_y_x, thorax_x_x, thorax_y_x)

                width = 20
                height = 35
                backwards = 7
                body_backwards = 15

                H1 = davi.calculate_new_coords(
                    head_x_x, head_y_x, x_theta_thorax_head+(math.pi/2), width)
                H2 = davi.calculate_new_coords(
                    head_x_x, head_y_x, x_theta_thorax_head-(math.pi/2), width)

                H3 = davi.calculate_new_coords(
                    H1[0], H1[1], x_theta_thorax_head, height)
                H4 = davi.calculate_new_coords(
                    H2[0], H2[1], x_theta_thorax_head, height)

                H5 = davi.calculate_new_coords(
                    H2[0], H2[1], x_theta_thorax_head+math.pi, backwards)
                H6 = davi.calculate_new_coords(
                    H1[0], H1[1], x_theta_thorax_head+math.pi, backwards)

                x_head_bbox = [[H6[0], H6[1]], [H3[0], H3[1]],
                               [H4[0], H4[1]], [H5[0], H5[1]]]

                x_head_bbox.append(x_head_bbox[0])

                xs, ys = zip(*x_head_bbox)

                B1 = davi.calculate_new_coords(
                    abdomen_x_x, abdomen_y_x, x_theta_abdomen_thorax+(math.pi/2), width)
                B2 = davi.calculate_new_coords(
                    abdomen_x_x, abdomen_y_x, x_theta_abdomen_thorax-(math.pi/2), width)

                B3 = davi.calculate_new_coords(
                    thorax_x_x, thorax_y_x, x_theta_abdomen_thorax+(math.pi/2), width)
                B4 = davi.calculate_new_coords(
                    thorax_x_x, thorax_y_x, x_theta_abdomen_thorax-(math.pi/2), width)

                B5 = davi.calculate_new_coords(
                    B1[0], B1[1], x_theta_abdomen_thorax+math.pi, body_backwards)
                B6 = davi.calculate_new_coords(
                    B2[0], B2[1], x_theta_abdomen_thorax+math.pi, body_backwards)

                x_body_bbox = [[B6[0], B6[1]], [B4[0], B4[1]],
                               [B3[0], B3[1]], [B5[0], B5[1]]]

                x_body_bbox.append(x_body_bbox[0])

                bxs, bys = zip(*x_body_bbox)

                # y bounding boxes

                y_theta_thorax_head = davi.get_vector_angle(
                    thorax_x_y, thorax_y_y, head_x_y, head_y_y)
                y_theta_abdomen_thorax = davi.get_vector_angle(
                    abdomen_x_y, abdomen_y_y, thorax_x_y, thorax_y_y)

                width = 20
                height = 35
                backwards = 7
                body_backwards = 15

                RH1 = davi.calculate_new_coords(
                    head_x_y, head_y_y, y_theta_thorax_head+(math.pi/2), width)
                RH2 = davi.calculate_new_coords(
                    head_x_y, head_y_y, y_theta_thorax_head-(math.pi/2), width)

                RH3 = davi.calculate_new_coords(
                    RH1[0], RH1[1], y_theta_thorax_head, height)
                RH4 = davi.calculate_new_coords(
                    RH2[0], RH2[1], y_theta_thorax_head, height)

                RH5 = davi.calculate_new_coords(
                    RH2[0], RH2[1], y_theta_thorax_head+math.pi, backwards)
                RH6 = davi.calculate_new_coords(
                    RH1[0], RH1[1], y_theta_thorax_head+math.pi, backwards)

                y_head_bbox = [[RH6[0], RH6[1]], [RH3[0], RH3[1]], [
                    RH4[0], RH4[1]], [RH5[0], RH5[1]]]

                y_head_bbox.append(y_head_bbox[0])

                xys, yys = zip(*y_head_bbox)

                RB1 = davi.calculate_new_coords(
                    abdomen_x_y, abdomen_y_y, y_theta_abdomen_thorax+(math.pi/2), width)
                RB2 = davi.calculate_new_coords(
                    abdomen_x_y, abdomen_y_y, y_theta_abdomen_thorax-(math.pi/2), width)

                RB3 = davi.calculate_new_coords(
                    thorax_x_y, thorax_y_y, y_theta_abdomen_thorax+(math.pi/2), width)
                RB4 = davi.calculate_new_coords(
                    thorax_x_y, thorax_y_y, y_theta_abdomen_thorax-(math.pi/2), width)

                RB5 = davi.calculate_new_coords(
                    RB1[0], RB1[1], y_theta_abdomen_thorax+math.pi, body_backwards)
                RB6 = davi.calculate_new_coords(
                    RB2[0], RB2[1], y_theta_abdomen_thorax+math.pi, body_backwards)

                y_body_bbox = [[RB6[0], RB6[1]], [RB4[0], RB4[1]], [
                    RB3[0], RB3[1]], [RB5[0], RB5[1]]]

                y_body_bbox.append(y_body_bbox[0])

                bxys, byys = zip(*y_body_bbox)

                # intersections over unions

                if math.isnan(x_theta_thorax_head) or math.isnan(y_theta_thorax_head):
                    continue
                else:

                    if math.isnan(head_x_x) or math.isnan(thorax_x_x) or math.isnan(head_y_y) or math.isnan(thorax_y_y):

                        continue

                    else:

                        x_head = davi.find_box_space(x_head_bbox)
                        y_head = davi.find_box_space(y_head_bbox)

                        overlap_head = davi.calculate_iou(x_head, y_head)

                        if overlap_head > 0:
                            overlapping_frames.append(i)

                if math.isnan(x_theta_thorax_head) or math.isnan(y_theta_abdomen_thorax):
                    continue
                else:

                    if math.isnan(head_x_x) or math.isnan(thorax_x_x) or math.isnan(abdomen_y_y) or math.isnan(thorax_y_y):
                        #                     print(head_x_x, thorax_x_x)
                        #                     print(abdomen_y_y, thorax_y_y)
                        continue

                    else:

                        x_head = davi.find_box_space(x_head_bbox)
                        y_body = davi.find_box_space(y_body_bbox)

                        overlap_head_body = davi.calculate_iou(x_head, y_body)

                        if overlap_head_body > 0:
                            overlapping_frames.append(i)

                if math.isnan(x_theta_abdomen_thorax) or math.isnan(y_theta_thorax_head):
                    continue
                else:
                    if math.isnan(abdomen_x_x) or math.isnan(thorax_x_x) or math.isnan(head_y_y) or math.isnan(thorax_y_y):
                        #                     print(abdomen_x_x, thorax_x_x)
                        #                     print(head_y_y, thorax_y_y)
                        continue

                    else:

                        x_body = davi.find_box_space(x_body_bbox)
                        y_head = davi.find_box_space(y_head_bbox)

                        overlap_body_head = davi.calculate_iou(x_body, y_head)

                        if overlap_body_head > 0:
                            overlapping_frames.append(i)

                if math.isnan(x_theta_abdomen_thorax) or math.isnan(y_theta_abdomen_thorax):
                    continue
                else:

                    if math.isnan(abdomen_x_x) or math.isnan(thorax_x_x) or math.isnan(abdomen_y_y) or math.isnan(thorax_y_y):
                        #                     print(abdomen_x_x, thorax_x_x)
                        #                     print(abdomen_y_y, thorax_y_y)
                        continue
                    else:

                        x_body = davi.find_box_space(x_body_bbox)
                        y_body = davi.find_box_space(y_body_bbox)

                        overlap_body_body = davi.calculate_iou(x_body, y_body)

                        if overlap_body_body > 0:
                            overlapping_frames.append(i)

    overlapping_frames = remove_dupl_list(overlapping_frames)

    interactions = []

    start = []
    end = []

    for k, g in groupby(enumerate(overlapping_frames), lambda x: x[0]-x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        interactions.append((group[0], group[-1]))

    for index, m in enumerate(interactions):

        ra = m

        start.append(ra[0])

        end.append(ra[1])

    interactions_dict = {"ant1_id": x,
                         "ant2_id": y,
                         "frame_start": start,
                         "frame_end": end}

    interactions_df = pd.DataFrame(interactions_dict)

    locals()["contacts_"+str(x) + str(y)] = interactions_df


contact_network = pd.concat([contacts_ind1ind2, contacts_ind1ind3, contacts_ind1ind4, contacts_ind1ind5, contacts_ind1ind6,
                            contacts_ind2ind3, contacts_ind2ind4, contacts_ind2ind5, contacts_ind2ind6,
                            contacts_ind3ind4, contacts_ind3ind5, contacts_ind3ind6,
                            contacts_ind4ind5, contacts_ind4ind6,
                            contacts_ind5ind6], ignore_index=True)


contact_network['duration'] = abs(
    contact_network['frame_start'] - contact_network['frame_end'])


contact_network = contact_network[contact_network.duration != 0].reset_index().drop([
    'index'], axis=1)


filtered_contacts = pd.DataFrame().reindex(columns=contact_network.columns)


for i in range(0, len(contact_network)):

    row = contact_network.loc[i]

    if i == 0:
        row_to_append = pd.DataFrame([{"ant1_id": row["ant1_id"],
                                       "ant2_id": row['ant2_id'],
                                       "frame_start": row['frame_start'],
                                       "frame_end": row['frame_end']}])
        filtered_contacts = pd.concat(
            [filtered_contacts, row_to_append], ignore_index=True)

    if i != 0:

        pre_row = filtered_contacts.iloc[-1]

        diff_rows = abs(row['frame_start'] - pre_row['frame_end'])

        if diff_rows < 20:  # a second

            if row['ant1_id'] == pre_row['ant1_id'] and row['ant2_id'] == pre_row['ant2_id']:

                row_to_append = pd.DataFrame([{"ant1_id": row['ant1_id'],
                                               "ant2_id": row['ant2_id'],
                                               "frame_start": pre_row['frame_start'],
                                               "frame_end": row['frame_end']}])

                filtered_contacts = filtered_contacts.iloc[:-1, :]

                filtered_contacts = pd.concat(
                    [filtered_contacts, row_to_append], ignore_index=True)
            else:

                row_to_append = pd.DataFrame([{"ant1_id": row["ant1_id"],
                                               "ant2_id": row['ant2_id'],
                                               "frame_start": row['frame_start'],
                                               "frame_end": row['frame_end']}])
                filtered_contacts = pd.concat(
                    [filtered_contacts, row_to_append], ignore_index=True)

        else:
            row_to_append = pd.DataFrame([{"ant1_id": row["ant1_id"],
                                           "ant2_id": row['ant2_id'],
                                           "frame_start": row['frame_start'],
                                           "frame_end": row['frame_end']}])

            filtered_contacts = pd.concat(
                [filtered_contacts, row_to_append], ignore_index=True)


filtered_contacts['duration'] = filtered_contacts['frame_end'] - \
    filtered_contacts['frame_start']


filtered_contacts['vidname'] = vidname	

save_place = os.path.dirname(detections_path)

filtered_contacts.to_csv(os.path.join(
    save_place, vidname+"_interaction_network.csv"), index=False)


print("finished! Enjoy your new contact network :)")
