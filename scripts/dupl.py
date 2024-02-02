
###########################################################################
######################### DUPLICATE CLEANING STAGE ########################
###########################################################################


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


# read in data

path = davi_output_path

ant1_path = os.path.join(path, 'ant1_detections.h5')
ant2_path = os.path.join(path, 'ant2_detections.h5')
ant3_path = os.path.join(path, 'ant3_detections.h5')
ant4_path = os.path.join(path, 'ant4_detections.h5')
ant5_path = os.path.join(path, 'ant5_detections.h5')
ant6_path = os.path.join(path, 'ant6_detections.h5')

ant1 = pd.read_hdf(ant1_path)
ant2 = pd.read_hdf(ant2_path)
ant3 = pd.read_hdf(ant3_path)
ant4 = pd.read_hdf(ant4_path)
ant5 = pd.read_hdf(ant5_path)
ant6 = pd.read_hdf(ant6_path)


final = pd.concat([ant1, ant2, ant3, ant4, ant5, ant6], join='inner', axis=1)


original_path = original_detections_path

data = pd.read_hdf(original_path)


vidname = os.path.basename(video_path)


ant1 = data.xs('ind1', level='individuals', axis=1, drop_level=False)
ant2 = data.xs('ind2', level='individuals', axis=1, drop_level=False)
ant3 = data.xs('ind3', level='individuals', axis=1, drop_level=False)
ant4 = data.xs('ind4', level='individuals', axis=1, drop_level=False)
ant5 = data.xs('ind5', level='individuals', axis=1, drop_level=False)
ant6 = data.xs('ind6', level='individuals', axis=1, drop_level=False)


ant1_copy = final.xs('ind1', level='individuals', axis=1, drop_level=False)
ant2_copy = final.xs('ind2', level='individuals', axis=1, drop_level=False)
ant3_copy = final.xs('ind3', level='individuals', axis=1, drop_level=False)
ant4_copy = final.xs('ind4', level='individuals', axis=1, drop_level=False)
ant5_copy = final.xs('ind5', level='individuals', axis=1, drop_level=False)
ant6_copy = final.xs('ind6', level='individuals', axis=1, drop_level=False)

length = 14
width = 16
extra = 4


# ant1 - ant2

dupl_1_2 = []
ref_dupl_1_2 = []

for i in range(0, len(final)):

    ind1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i]
    ind1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i]

    ind2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
    ind2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]

    if math.isnan(ind1_abx):
        continue
    else:

        if ind1_abx == ind2_abx:
            if math.isnan(ind1_abx):
                continue
            else:
                dupl_1_2.append(i)


if len(dupl_1_2) > 0:
    ref_dupl_1_2.append(dupl_1_2[0])

for x, y in zip(dupl_1_2[::], dupl_1_2[1::]):
    if abs(x-y) > 1:
        ref_dupl_1_2.append(x)


if len(dupl_1_2) > 0:
    ref_dupl_1_2.append(dupl_1_2[-1])


if len(ref_dupl_1_2) != 0:
    ref_dupl_1_2 = list(set(ref_dupl_1_2))

ref_dupl_1_2 = sorted(ref_dupl_1_2)


for n in range(0, len(ant1)):
    nan_check = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][n]
    nan_check2 = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'head', 'x')][n]
    nan_check3 = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000',
                       'ind1', 'top_left_antenna', 'x')][n]
    nan_check4 = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000',
                       'ind1', 'base_hind_right_leg', 'x')][n]
    nan_check5 = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000',
                       'ind1', 'top_right_antenna', 'x')][n]

    if math.isnan(nan_check) and math.isnan(nan_check2) and math.isnan(nan_check3) and math.isnan(nan_check4) and math.isnan(nan_check5):
        empty_row = ant1.loc[n]
        break

weird_detections = []

for dupl in tqdm(dupl_1_2):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ## other detections ##

    ant1_abx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ## thetas ##

    ant1_current_theta = davi.get_vector_angle(
        ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)

    ant1_original_theta = davi.get_vector_angle(
        ant1_abx_o, ant1_aby_o, ant1_thx_o, ant1_thy_o)
    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)

    ## waffles ##

    if ant1_current_theta == 0 or math.isnan(ant1_current_theta):
        ant1_current_ant1_original = np.nan
        ant1_current_ant2_original = np.nan
        # maybe data = nan and loop continues

    elif ant1_current_theta != 0:

        ant1_current_waffle = davi.create_waffle(
            ant1_abx, ant1_aby, ant1_current_theta, pix)
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)

        if ant1_original_theta == 0 or math.isnan(ant1_original_theta):
            ant1_current_ant1_original = np.nan

        elif ant1_original_theta != 0:
            ant1_original_waffle = davi.create_waffle(
                ant1_abx_o, ant1_aby_o, ant1_original_theta, pix)
            ant1_current_ant1_original = davi.subtract_colours_2(
                ant1_current_waffle, ant1_original_waffle)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant1_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant1_current_ant2_original = davi.subtract_colours_2(
                ant1_current_waffle, ant2_original_waffle)

    ## comparisons ##

    if ant1_current_ant1_original < ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant2_copy.loc[dupl] = copy_data
        previous_detection = "ant2_needs_changing"

    elif ant1_current_ant1_original > ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant1_copy.loc[dupl] = copy_data
        previous_detection = "ant1_needs_changing"

    elif ant1_current_ant1_original == ant1_current_ant2_original:
        if abs(ant1_abx - ant1_abx_o) > abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant1_copy.loc[dupl] = copy_data
        elif abs(ant1_abx - ant1_abx_o) < abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant2_copy.loc[dupl] = copy_data
    else:
        weird_detections.append(dupl)


# ant1 - ant3

dupl_1_3 = []
ref_dupl_1_3 = []

for i in range(0, len(final)):

    ind1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i]
    ind1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i]

    ind3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
    ind3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]

    if math.isnan(ind1_abx):
        continue
    else:

        if ind1_abx == ind3_abx:
            if math.isnan(ind1_abx):
                continue
            else:
                dupl_1_3.append(i)


if len(dupl_1_3) > 0:
    ref_dupl_1_3.append(dupl_1_3[0])

for x, y in zip(dupl_1_3[::], dupl_1_3[1::]):
    if abs(x-y) > 1:
        ref_dupl_1_3.append(x)


if len(dupl_1_3) > 0:
    ref_dupl_1_3.append(dupl_1_3[-1])


if len(ref_dupl_1_3) != 0:
    ref_dupl_1_3 = list(set(ref_dupl_1_3))

ref_dupl_1_3 = sorted(ref_dupl_1_3)


for dupl in tqdm(dupl_1_3):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ## other detections ##

    ant1_abx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant2_thy_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ## thetas ##

    ant1_current_theta = davi.get_vector_angle(
        ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)

    ant1_original_theta = davi.get_vector_angle(
        ant1_abx_o, ant1_aby_o, ant1_thx_o, ant1_thy_o)
    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)

    ## waffles ##

    if ant1_current_theta == 0 or math.isnan(ant1_current_theta):
        ant1_current_ant1_original = np.nan
        ant1_current_ant2_original = np.nan

    elif ant1_current_theta != 0:
        ant1_current_waffle = davi.create_waffle(
            ant1_abx, ant1_aby, ant1_current_theta, pix)
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)

        if ant1_original_theta == 0 or math.isnan(ant1_original_theta):
            ant1_current_ant1_original = np.nan

        elif ant1_original_theta != 0:
            ant1_original_waffle = davi.create_waffle(
                ant1_abx_o, ant1_aby_o, ant1_original_theta, pix)
            ant1_current_ant1_original = davi.subtract_colours_2(
                ant1_current_waffle, ant1_original_waffle)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant1_current_ant2_original = np.nan

        elif ant2_original_theta != 0:

            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant1_current_ant2_original = davi.subtract_colours_2(
                ant1_current_waffle, ant2_original_waffle)

    ## comparisons ##

    if ant1_current_ant1_original < ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant3_copy.loc[dupl] = copy_data
        previous_detection = "ant2_needs_changing"

    elif ant1_current_ant1_original > ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant1_copy.loc[dupl] = copy_data
        previous_detection = "ant1_needs_changing"

    elif ant1_current_ant1_original == ant1_current_ant2_original:
        if abs(ant1_abx - ant1_abx_o) > abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant1_copy.loc[dupl] = copy_data
        elif abs(ant1_abx - ant1_abx_o) < abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant3_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant1 - ant4

dupl_1_4 = []
ref_dupl_1_4 = []

for i in range(0, len(final)):

    ind1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i]
    ind1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i]

    ind4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
    ind4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]

    if math.isnan(ind1_abx):
        continue
    else:

        if ind1_abx == ind4_abx:
            if math.isnan(ind1_abx):
                continue
            else:
                dupl_1_4.append(i)


if len(dupl_1_4) > 0:
    ref_dupl_1_4.append(dupl_1_4[0])

for x, y in zip(dupl_1_4[::], dupl_1_4[1::]):
    if abs(x-y) > 1:
        ref_dupl_1_4.append(x)


if len(dupl_1_4) > 0:
    ref_dupl_1_4.append(dupl_1_4[-1])


if len(ref_dupl_1_4) != 0:
    ref_dupl_1_4 = list(set(ref_dupl_1_4))

ref_dupl_1_4 = sorted(ref_dupl_1_4)

for dupl in tqdm(dupl_1_4):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ## other detections ##

    ant1_abx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant2_thy_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ## thetas ##

    ant1_current_theta = davi.get_vector_angle(
        ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)

    ant1_original_theta = davi.get_vector_angle(
        ant1_abx_o, ant1_aby_o, ant1_thx_o, ant1_thy_o)
    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)

    ## waffles ##

    if ant1_current_theta == 0 or math.isnan(ant1_current_theta):
        ant1_current_ant1_original = np.nan
        ant1_current_ant2_original = np.nan
        continue
    elif ant1_current_theta != 0:
        ant1_current_waffle = davi.create_waffle(
            ant1_abx, ant1_aby, ant1_current_theta, pix)
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)

        if ant1_original_theta == 0 or math.isnan(ant1_original_theta):
            ant1_current_ant1_original = np.nan

        elif ant1_original_theta != 0:
            ant1_original_waffle = davi.create_waffle(
                ant1_abx_o, ant1_aby_o, ant1_original_theta, pix)
            ant1_current_ant1_original = davi.subtract_colours_2(
                ant1_current_waffle, ant1_original_waffle)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant1_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant1_current_ant2_original = davi.subtract_colours_2(
                ant1_current_waffle, ant2_original_waffle)

    ## comparisons ##

    if ant1_current_ant1_original < ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant4_copy.loc[dupl] = copy_data
        previous_detection = "ant2_needs_changing"

    elif ant1_current_ant1_original > ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant1_copy.loc[dupl] = copy_data
        previous_detection = "ant1_needs_changing"

    elif ant1_current_ant1_original == ant1_current_ant2_original:
        if abs(ant1_abx - ant1_abx_o) > abs(ant1_abx - ant2_abx_o):

            copy_data = list(empty_row)
            ant1_copy.loc[dupl] = copy_data
        elif abs(ant1_abx - ant1_abx_o) < abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant4_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant1 - ant5


dupl_1_5 = []
ref_dupl_1_5 = []

for i in range(0, len(final)):

    ind1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i]
    ind1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i]

    ind5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
    ind5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

    if math.isnan(ind1_abx):
        continue
    else:

        if ind1_abx == ind5_abx:
            if math.isnan(ind1_abx):
                continue
            else:
                dupl_1_5.append(i)


if len(dupl_1_5) > 0:
    ref_dupl_1_5.append(dupl_1_5[0])

for x, y in zip(dupl_1_5[::], dupl_1_5[1::]):
    if abs(x-y) > 1:
        ref_dupl_1_5.append(x)


if len(dupl_1_5) > 0:
    ref_dupl_1_5.append(dupl_1_5[-1])


if len(ref_dupl_1_5) != 0:
    ref_dupl_1_5 = list(set(ref_dupl_1_5))

ref_dupl_1_5 = sorted(ref_dupl_1_5)


for dupl in tqdm(dupl_1_5):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## other detections ##

    ant1_abx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant2_thy_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## thetas ##

    ant1_current_theta = davi.get_vector_angle(
        ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)

    ant1_original_theta = davi.get_vector_angle(
        ant1_abx_o, ant1_aby_o, ant1_thx_o, ant1_thy_o)
    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)

    ## waffles ##

    if ant1_current_theta == 0 or math.isnan(ant1_current_theta):
        ant1_current_ant1_original = np.nan
        ant1_current_ant2_original = np.nan
        continue
    elif ant1_current_theta != 0:
        ant1_current_waffle = davi.create_waffle(
            ant1_abx, ant1_aby, ant1_current_theta, pix)
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)

        if ant1_original_theta == 0 or math.isnan(ant1_original_theta):
            ant1_current_ant1_original = np.nan

        elif ant1_original_theta != 0:
            ant1_original_waffle = davi.create_waffle(
                ant1_abx_o, ant1_aby_o, ant1_original_theta, pix)
            ant1_current_ant1_original = davi.subtract_colours_2(
                ant1_current_waffle, ant1_original_waffle)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant1_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant1_current_ant2_original = davi.subtract_colours_2(
                ant1_current_waffle, ant2_original_waffle)

    ## comparisons ##

    if ant1_current_ant1_original < ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant5_copy.loc[dupl] = copy_data
        previous_detection = "ant2_needs_changing"

    elif ant1_current_ant1_original > ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant1_copy.loc[dupl] = copy_data
        previous_detection = "ant1_needs_changing"

    elif ant1_current_ant1_original == ant1_current_ant2_original:
        if abs(ant1_abx - ant1_abx_o) > abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant1_copy.loc[dupl] = copy_data
        elif abs(ant1_abx - ant1_abx_o) < abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant5_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant1 - ant6

dupl_1_6 = []
ref_dupl_1_6 = []

for i in range(0, len(final)):

    ind1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i]
    ind1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i]

    ind6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i]
    ind6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i]

    if math.isnan(ind1_abx):
        continue
    else:

        if ind1_abx == ind6_abx:
            if math.isnan(ind1_abx):
                continue
            else:
                dupl_1_6.append(i)


if len(dupl_1_6) > 0:
    ref_dupl_1_6.append(dupl_1_6[0])

for x, y in zip(dupl_1_6[::], dupl_1_6[1::]):
    if abs(x-y) > 1:
        ref_dupl_1_6.append(x)


if len(dupl_1_6) > 0:
    ref_dupl_1_6.append(dupl_1_6[-1])


if len(ref_dupl_1_6) != 0:
    ref_dupl_1_6 = list(set(ref_dupl_1_6))

ref_dupl_1_6 = sorted(ref_dupl_1_6)


for dupl in tqdm(dupl_1_6):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant1_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## other detections ##

    ant1_abx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][dupl]
    ant1_aby_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][dupl]
    ant1_thx_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][dupl]
    ant1_thy_o = ant1[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][dupl]

    ant2_abx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant2_thy_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## thetas ##

    ant1_current_theta = davi.get_vector_angle(
        ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)

    ant1_original_theta = davi.get_vector_angle(
        ant1_abx_o, ant1_aby_o, ant1_thx_o, ant1_thy_o)
    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)

    ## waffles ##

    if ant1_current_theta == 0 or math.isnan(ant1_current_theta):
        ant1_current_ant1_original = np.nan
        ant1_current_ant2_original = np.nan
        continue
    elif ant1_current_theta != 0:
        ant1_current_waffle = davi.create_waffle(
            ant1_abx, ant1_aby, ant1_current_theta, pix)
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)

        if ant1_original_theta == 0 or math.isnan(ant1_original_theta):
            ant1_current_ant1_original = np.nan
            continue
        elif ant1_original_theta != 0:
            ant1_original_waffle = davi.create_waffle(
                ant1_abx_o, ant1_aby_o, ant1_original_theta, pix)
            ant1_current_ant1_original = davi.subtract_colours_2(
                ant1_current_waffle, ant1_original_waffle)

        if ant2_original_theta == 0:
            ant1_current_ant2_original = np.nan
            continue

        elif ant2_original_theta != 0 or math.isnan(ant2_original_theta):
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant1_current_ant2_original = davi.subtract_colours_2(
                ant1_current_waffle, ant2_original_waffle)

    ## comparisons ##

    if ant1_current_ant1_original < ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant6_copy.loc[dupl] = copy_data
        previous_detection = "ant2_needs_changing"

    elif ant1_current_ant1_original > ant1_current_ant2_original:

        copy_data = list(empty_row)
        ant1_copy.loc[dupl] = copy_data
        previous_detection = "ant1_needs_changing"

    elif ant1_current_ant1_original == ant1_current_ant2_original:
        if abs(ant1_abx - ant1_abx_o) > abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant1_copy.loc[dupl] = copy_data
        elif abs(ant1_abx - ant1_abx_o) < abs(ant1_abx - ant2_abx_o):
            copy_data = list(empty_row)
            ant6_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant2 - ant3

dupl_2_3 = []
ref_dupl_2_3 = []

for i in range(0, len(final)):

    ind2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
    ind2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]

    ind3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
    ind3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]

    if math.isnan(ind2_abx):
        continue
    else:

        if ind2_abx == ind3_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                dupl_2_3.append(i)


if len(dupl_2_3) > 0:
    ref_dupl_2_3.append(dupl_2_3[0])

for x, y in zip(dupl_2_3[::], dupl_2_3[1::]):
    if abs(x-y) > 2:
        ref_dupl_2_3.append(x)


if len(dupl_2_3) > 0:
    ref_dupl_2_3.append(dupl_2_3[-1])


if len(ref_dupl_2_3) != 0:
    ref_dupl_2_3 = list(set(ref_dupl_2_3))

ref_dupl_2_3 = sorted(ref_dupl_2_3)

for dupl in tqdm(dupl_2_3):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ## other detections ##

    ant2_abx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant3_abx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ## thetas ##

    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)
    ant3_current_theta = davi.get_vector_angle(
        ant3_abx, ant3_aby, ant3_thx, ant3_thy)

    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)
    ant3_original_theta = davi.get_vector_angle(
        ant3_abx_o, ant3_aby_o, ant3_thx_o, ant3_thy_o)

    ## waffles ##

    if ant2_current_theta == 0 or math.isnan(ant2_current_theta):
        ant2_current_ant2_original = np.nan
        ant2_current_ant3_original = np.nan

    elif ant2_current_theta != 0:
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)
        ant3_current_waffle = davi.create_waffle(
            ant3_abx, ant3_aby, ant3_current_theta, pix)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant2_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant2_current_ant2_original = davi.subtract_colours_2(
                ant2_current_waffle, ant2_original_waffle)

        if ant3_original_theta == 0 or math.isnan(ant3_original_theta):
            ant2_current_ant3_original = np.nan

        elif ant3_original_theta != 0:
            ant3_original_waffle = davi.create_waffle(
                ant3_abx_o, ant3_aby_o, ant3_original_theta, pix)
            ant2_current_ant3_original = davi.subtract_colours_2(
                ant2_current_waffle, ant3_original_waffle)

    ## comparisons ##

    if ant2_current_ant2_original < ant2_current_ant3_original:

        copy_data = list(empty_row)
        ant3_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original > ant2_current_ant3_original:

        copy_data = list(empty_row)
        ant2_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original == ant2_current_ant3_original:
        if abs(ant2_abx - ant2_abx_o) > abs(ant2_abx - ant3_abx_o):
            copy_data = list(empty_row)
            ant2_copy.loc[dupl] = copy_data

        elif abs(ant2_abx - ant2_abx_o) < abs(ant2_abx - ant3_abx_o):
            copy_data = list(empty_row)
            ant3_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant2 - ant4

dupl_2_4 = []
ref_dupl_2_4 = []

for i in range(0, len(final)):

    ind2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
    ind2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]

    ind4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
    ind4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]

    if math.isnan(ind2_abx):
        continue
    else:

        if ind2_abx == ind4_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                dupl_2_4.append(i)


if len(dupl_2_4) > 0:
    ref_dupl_2_4.append(dupl_2_4[0])

for x, y in zip(dupl_2_4[::], dupl_2_4[1::]):
    if abs(x-y) > 2:
        ref_dupl_2_4.append(x)


if len(dupl_2_4) > 0:
    ref_dupl_2_4.append(dupl_2_4[-1])


if len(ref_dupl_2_4) != 0:
    ref_dupl_2_4 = list(set(ref_dupl_2_4))

ref_dupl_2_4 = sorted(ref_dupl_2_4)


for dupl in tqdm(dupl_2_4):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ## other detections ##

    ant2_abx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant4_abx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ## thetas ##

    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)
    ant4_current_theta = davi.get_vector_angle(
        ant4_abx, ant4_aby, ant4_thx, ant4_thy)

    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)
    ant4_original_theta = davi.get_vector_angle(
        ant4_abx_o, ant4_aby_o, ant4_thx_o, ant4_thy_o)

    ## waffles ##

    if ant2_current_theta == 0 or math.isnan(ant2_current_theta):
        ant2_current_ant2_original = np.nan
        ant2_current_ant4_original = np.nan

    elif ant2_current_theta != 0:
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)
        ant4_current_waffle = davi.create_waffle(
            ant4_abx, ant4_aby, ant4_current_theta, pix)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant2_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant2_current_ant2_original = davi.subtract_colours_2(
                ant2_current_waffle, ant2_original_waffle)

        if ant4_original_theta == 0 or math.isnan(ant4_original_theta):
            ant2_current_ant4_original = np.nan

        elif ant4_original_theta != 0:
            ant4_original_waffle = davi.create_waffle(
                ant4_abx_o, ant4_aby_o, ant4_original_theta, pix)
            ant2_current_ant4_original = davi.subtract_colours_2(
                ant2_current_waffle, ant4_original_waffle)

    ## comparisons ##

    if ant2_current_ant2_original < ant2_current_ant4_original:

        copy_data = list(empty_row)
        ant4_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original > ant2_current_ant4_original:

        copy_data = list(empty_row)
        ant2_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original == ant2_current_ant4_original:
        if abs(ant2_abx - ant2_abx_o) > abs(ant2_abx - ant4_abx_o):
            copy_data = list(empty_row)
            ant2_copy.loc[dupl] = copy_data

        elif abs(ant2_abx - ant2_abx_o) < abs(ant2_abx - ant4_abx_o):
            copy_data = list(empty_row)
            ant4_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant2 - ant5

dupl_2_5 = []
ref_dupl_2_5 = []

for i in range(0, len(final)):

    ind2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
    ind2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]

    ind5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
    ind5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

    if math.isnan(ind2_abx):
        continue
    else:

        if ind2_abx == ind5_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                dupl_2_5.append(i)


if len(dupl_2_5) > 0:
    ref_dupl_2_5.append(dupl_2_5[0])

for x, y in zip(dupl_2_5[::], dupl_2_5[1::]):
    if abs(x-y) > 2:
        ref_dupl_2_5.append(x)


if len(dupl_2_5) > 0:
    ref_dupl_2_5.append(dupl_2_5[-1])


if len(ref_dupl_2_5) != 0:
    ref_dupl_2_5 = list(set(ref_dupl_2_5))

ref_dupl_2_5 = sorted(ref_dupl_2_5)


for dupl in tqdm(dupl_2_5):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## other detections ##

    ant2_abx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant5_abx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## thetas ##

    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)
    ant5_current_theta = davi.get_vector_angle(
        ant5_abx, ant5_aby, ant5_thx, ant5_thy)

    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)
    ant5_original_theta = davi.get_vector_angle(
        ant5_abx_o, ant5_aby_o, ant5_thx_o, ant5_thy_o)

    ## waffles ##

    if ant2_current_theta == 0 or math.isnan(ant2_current_theta):
        ant2_current_ant2_original = np.nan
        ant2_current_ant5_original = np.nan

    elif ant2_current_theta != 0:
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)
        ant5_current_waffle = davi.create_waffle(
            ant5_abx, ant5_aby, ant5_current_theta, pix)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant2_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant2_current_ant2_original = davi.subtract_colours_2(
                ant2_current_waffle, ant2_original_waffle)

        if ant5_original_theta == 0 or math.isnan(ant5_original_theta):
            ant2_current_ant5_original = np.nan

        elif ant5_original_theta != 0:
            ant5_original_waffle = davi.create_waffle(
                ant5_abx_o, ant5_aby_o, ant5_original_theta, pix)
            ant2_current_ant5_original = davi.subtract_colours_2(
                ant2_current_waffle, ant5_original_waffle)

    ## comparisons ##

    if ant2_current_ant2_original < ant2_current_ant5_original:

        copy_data = list(empty_row)
        ant5_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original > ant2_current_ant5_original:

        copy_data = list(empty_row)
        ant2_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original == ant2_current_ant5_original:
        if abs(ant2_abx - ant2_abx_o) > abs(ant2_abx - ant5_abx_o):
            copy_data = list(empty_row)
            ant2_copy.loc[dupl] = copy_data

        elif abs(ant2_abx - ant2_abx_o) < abs(ant2_abx - ant5_abx_o):
            copy_data = list(empty_row)
            ant5_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant2 - ant6

dupl_2_6 = []
ref_dupl_2_6 = []

for i in range(0, len(final)):

    ind2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
    ind2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]

    ind6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i]
    ind6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i]

    if math.isnan(ind2_abx):
        continue
    else:

        if ind2_abx == ind6_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                dupl_2_6.append(i)


if len(dupl_2_6) > 0:
    ref_dupl_2_6.append(dupl_2_6[0])

for x, y in zip(dupl_2_6[::], dupl_2_6[1::]):
    if abs(x-y) > 2:
        ref_dupl_2_6.append(x)


if len(dupl_2_6) > 0:
    ref_dupl_2_6.append(dupl_2_6[-1])


if len(ref_dupl_2_6) != 0:
    ref_dupl_2_6 = list(set(ref_dupl_2_6))

ref_dupl_2_6 = sorted(ref_dupl_2_6)


for dupl in tqdm(dupl_2_6):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant2_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## other detections ##

    ant2_abx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][dupl]
    ant2_aby_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][dupl]
    ant2_thx_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][dupl]
    ant2_thy_o = ant2[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][dupl]

    ant6_abx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## thetas ##

    ant2_current_theta = davi.get_vector_angle(
        ant2_abx, ant2_aby, ant2_thx, ant2_thy)
    ant6_current_theta = davi.get_vector_angle(
        ant6_abx, ant6_aby, ant6_thx, ant6_thy)

    ant2_original_theta = davi.get_vector_angle(
        ant2_abx_o, ant2_aby_o, ant2_thx_o, ant2_thy_o)
    ant6_original_theta = davi.get_vector_angle(
        ant6_abx_o, ant6_aby_o, ant6_thx_o, ant6_thy_o)

    ## waffles ##

    if ant2_current_theta == 0 or math.isnan(ant2_current_theta):
        ant2_current_ant2_original = np.nan
        ant2_current_ant6_original = np.nan

    elif ant2_current_theta != 0:
        ant2_current_waffle = davi.create_waffle(
            ant2_abx, ant2_aby, ant2_current_theta, pix)
        ant6_current_waffle = davi.create_waffle(
            ant6_abx, ant6_aby, ant6_current_theta, pix)

        if ant2_original_theta == 0 or math.isnan(ant2_original_theta):
            ant2_current_ant2_original = np.nan

        elif ant2_original_theta != 0:
            ant2_original_waffle = davi.create_waffle(
                ant2_abx_o, ant2_aby_o, ant2_original_theta, pix)
            ant2_current_ant2_original = davi.subtract_colours_2(
                ant2_current_waffle, ant2_original_waffle)

        if ant6_original_theta == 0 or math.isnan(ant6_original_theta):
            ant2_current_ant6_original = np.nan

        elif ant6_original_theta != 0:
            ant6_original_waffle = davi.create_waffle(
                ant6_abx_o, ant6_aby_o, ant6_original_theta, pix)
            ant2_current_ant6_original = davi.subtract_colours_2(
                ant2_current_waffle, ant6_original_waffle)

    ## comparisons ##

    if ant2_current_ant2_original < ant2_current_ant6_original:

        copy_data = list(empty_row)
        ant6_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original > ant2_current_ant6_original:

        copy_data = list(empty_row)
        ant2_copy.loc[dupl] = copy_data

    elif ant2_current_ant2_original == ant2_current_ant6_original:
        if abs(ant2_abx - ant2_abx_o) > abs(ant2_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant2_copy.loc[dupl] = copy_data

        elif abs(ant2_abx - ant2_abx_o) < abs(ant2_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant6_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant3 - ant4

dupl_3_4 = []
ref_dupl_3_4 = []

for i in range(0, len(final)):

    ind3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
    ind3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]

    ind4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
    ind4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]

    if math.isnan(ind3_abx):
        continue
    else:

        if ind3_abx == ind4_abx:
            if math.isnan(ind3_abx):
                continue
            else:
                dupl_3_4.append(i)


if len(dupl_3_4) > 0:
    ref_dupl_3_4.append(dupl_3_4[0])

for x, y in zip(dupl_3_4[::], dupl_3_4[1::]):
    if abs(x-y) > 3:
        ref_dupl_3_4.append(x)


if len(dupl_3_4) > 0:
    ref_dupl_3_4.append(dupl_3_4[-1])


if len(ref_dupl_3_4) != 0:
    ref_dupl_3_4 = list(set(ref_dupl_3_4))

ref_dupl_3_4 = sorted(ref_dupl_3_4)


for dupl in tqdm(dupl_3_4):
    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()
    ## currents ##

    ant3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ant4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ## other detections ##

    ant3_abx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ant4_abx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ## thetas ##

    ant3_current_theta = davi.get_vector_angle(
        ant3_abx, ant3_aby, ant3_thx, ant3_thy)
    ant4_current_theta = davi.get_vector_angle(
        ant4_abx, ant4_aby, ant4_thx, ant4_thy)

    ant3_original_theta = davi.get_vector_angle(
        ant3_abx_o, ant3_aby_o, ant3_thx_o, ant3_thy_o)
    ant4_original_theta = davi.get_vector_angle(
        ant4_abx_o, ant4_aby_o, ant4_thx_o, ant4_thy_o)

    ## waffles ##

    if ant3_current_theta == 0 or math.isnan(ant3_current_theta):
        ant3_current_ant3_original = np.nan
        ant3_current_ant4_original = np.nan

    elif ant3_current_theta != 0:
        ant3_current_waffle = davi.create_waffle(
            ant3_abx, ant3_aby, ant3_current_theta, pix)
        ant4_current_waffle = davi.create_waffle(
            ant4_abx, ant4_aby, ant4_current_theta, pix)

        if ant3_original_theta == 0 or math.isnan(ant3_original_theta):
            ant3_current_ant3_original = np.nan

        elif ant3_original_theta != 0:
            ant3_original_waffle = davi.create_waffle(
                ant3_abx_o, ant3_aby_o, ant3_original_theta, pix)
            ant3_current_ant3_original = davi.subtract_colours_2(
                ant3_current_waffle, ant3_original_waffle)

        if ant4_original_theta == 0 or math.isnan(ant4_original_theta):
            ant3_current_ant4_original = np.nan

        elif ant4_original_theta != 0:
            ant4_original_waffle = davi.create_waffle(
                ant4_abx_o, ant4_aby_o, ant4_original_theta, pix)
            ant3_current_ant4_original = davi.subtract_colours_2(
                ant3_current_waffle, ant4_original_waffle)

    ## comparisons ##

    if ant3_current_ant3_original < ant3_current_ant4_original:

        copy_data = list(empty_row)
        ant4_copy.loc[dupl] = copy_data

    elif ant3_current_ant3_original > ant3_current_ant4_original:

        copy_data = list(empty_row)
        ant3_copy.loc[dupl] = copy_data

    elif ant3_current_ant3_original == ant3_current_ant4_original:
        if abs(ant3_abx - ant3_abx_o) > abs(ant3_abx - ant4_abx_o):
            copy_data = list(empty_row)
            ant3_copy.loc[dupl] = copy_data

        elif abs(ant3_abx - ant3_abx_o) < abs(ant3_abx - ant4_abx_o):
            copy_data = list(empty_row)
            ant4_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant3 - ant5

dupl_3_5 = []
ref_dupl_3_5 = []

for i in range(0, len(final)):

    ind3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
    ind3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]

    ind5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
    ind5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

    if math.isnan(ind3_abx):
        continue
    else:

        if ind3_abx == ind5_abx:
            if math.isnan(ind3_abx):
                continue
            else:
                dupl_3_5.append(i)


if len(dupl_3_5) > 0:
    ref_dupl_3_5.append(dupl_3_5[0])

for x, y in zip(dupl_3_5[::], dupl_3_5[1::]):
    if abs(x-y) > 3:
        ref_dupl_3_5.append(x)


if len(dupl_3_5) > 0:
    ref_dupl_3_5.append(dupl_3_5[-1])


if len(ref_dupl_3_5) != 0:
    ref_dupl_3_5 = list(set(ref_dupl_3_5))

ref_dupl_3_5 = sorted(ref_dupl_3_5)


for dupl in tqdm(dupl_3_5):
    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ant5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## other detections ##

    ant3_abx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ant5_abx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## thetas ##

    ant3_current_theta = davi.get_vector_angle(
        ant3_abx, ant3_aby, ant3_thx, ant3_thy)
    ant5_current_theta = davi.get_vector_angle(
        ant5_abx, ant5_aby, ant5_thx, ant5_thy)

    ant3_original_theta = davi.get_vector_angle(
        ant3_abx_o, ant3_aby_o, ant3_thx_o, ant3_thy_o)
    ant5_original_theta = davi.get_vector_angle(
        ant5_abx_o, ant5_aby_o, ant5_thx_o, ant5_thy_o)

    ## waffles ##

    if ant3_current_theta == 0 or math.isnan(ant3_current_theta):
        ant3_current_ant3_original = np.nan
        ant3_current_ant5_original = np.nan

    elif ant3_current_theta != 0:
        ant3_current_waffle = davi.create_waffle(
            ant3_abx, ant3_aby, ant3_current_theta, pix)
        ant5_current_waffle = davi.create_waffle(
            ant5_abx, ant5_aby, ant5_current_theta, pix)

        if ant3_original_theta == 0 or math.isnan(ant3_original_theta):
            ant3_current_ant3_original = np.nan

        elif ant3_original_theta != 0:
            ant3_original_waffle = davi.create_waffle(
                ant3_abx_o, ant3_aby_o, ant3_original_theta, pix)
            ant3_current_ant3_original = davi.subtract_colours_2(
                ant3_current_waffle, ant3_original_waffle)

        if ant5_original_theta == 0 or math.isnan(ant5_original_theta):
            ant3_current_ant5_original = np.nan

        elif ant5_original_theta != 0:
            ant5_original_waffle = davi.create_waffle(
                ant5_abx_o, ant5_aby_o, ant5_original_theta, pix)
            ant3_current_ant5_original = davi.subtract_colours_2(
                ant3_current_waffle, ant5_original_waffle)

    ## comparisons ##

    if ant3_current_ant3_original < ant3_current_ant5_original:

        copy_data = list(empty_row)
        ant5_copy.loc[dupl] = copy_data

    elif ant3_current_ant3_original > ant3_current_ant5_original:

        copy_data = list(empty_row)
        ant3_copy.loc[dupl] = copy_data

    elif ant3_current_ant3_original == ant3_current_ant5_original:
        if abs(ant3_abx - ant3_abx_o) > abs(ant3_abx - ant5_abx_o):
            copy_data = list(empty_row)
            ant3_copy.loc[dupl] = copy_data

        elif abs(ant3_abx - ant3_abx_o) < abs(ant3_abx - ant5_abx_o):
            copy_data = list(empty_row)
            ant5_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant3 - ant6

dupl_3_6 = []
ref_dupl_3_6 = []

for i in range(0, len(final)):

    ind3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
    ind3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]

    ind6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i]
    ind6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i]

    if math.isnan(ind3_abx):
        continue
    else:

        if ind3_abx == ind6_abx:
            if math.isnan(ind3_abx):
                continue
            else:
                dupl_3_6.append(i)


if len(dupl_3_6) > 0:
    ref_dupl_3_6.append(dupl_3_6[0])

for x, y in zip(dupl_3_6[::], dupl_3_6[1::]):
    if abs(x-y) > 3:
        ref_dupl_3_6.append(x)


if len(dupl_3_6) > 0:
    ref_dupl_3_6.append(dupl_3_6[-1])


if len(ref_dupl_3_6) != 0:
    ref_dupl_3_6 = list(set(ref_dupl_3_6))

ref_dupl_3_6 = sorted(ref_dupl_3_6)


for dupl in tqdm(dupl_3_6):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant3_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ant6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## other detections ##

    ant3_abx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][dupl]
    ant3_aby_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][dupl]
    ant3_thx_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][dupl]
    ant3_thy_o = ant3[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][dupl]

    ant6_abx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## thetas ##

    ant3_current_theta = davi.get_vector_angle(
        ant3_abx, ant3_aby, ant3_thx, ant3_thy)
    ant6_current_theta = davi.get_vector_angle(
        ant6_abx, ant6_aby, ant6_thx, ant6_thy)

    ant3_original_theta = davi.get_vector_angle(
        ant3_abx_o, ant3_aby_o, ant3_thx_o, ant3_thy_o)
    ant6_original_theta = davi.get_vector_angle(
        ant6_abx_o, ant6_aby_o, ant6_thx_o, ant6_thy_o)

    ## waffles ##

    if ant3_current_theta == 0 or math.isnan(ant3_current_theta):
        ant3_current_ant3_original = np.nan
        ant3_current_ant6_original = np.nan

    elif ant3_current_theta != 0:
        ant3_current_waffle = davi.create_waffle(
            ant3_abx, ant3_aby, ant3_current_theta, pix)
        ant6_current_waffle = davi.create_waffle(
            ant6_abx, ant6_aby, ant6_current_theta, pix)

        if ant3_original_theta == 0 or math.isnan(ant3_original_theta):
            ant3_current_ant3_original = np.nan

        elif ant3_original_theta != 0:
            ant3_original_waffle = davi.create_waffle(
                ant3_abx_o, ant3_aby_o, ant3_original_theta, pix)
            ant3_current_ant3_original = davi.subtract_colours_2(
                ant3_current_waffle, ant3_original_waffle)

        if ant6_original_theta == 0 or math.isnan(ant6_original_theta):
            ant3_current_ant6_original = np.nan

        elif ant6_original_theta != 0:
            ant6_original_waffle = davi.create_waffle(
                ant6_abx_o, ant6_aby_o, ant6_original_theta, pix)
            ant3_current_ant6_original = davi.subtract_colours_2(
                ant3_current_waffle, ant6_original_waffle)

    ## comparisons ##

    if ant3_current_ant3_original < ant3_current_ant6_original:

        copy_data = list(empty_row)
        ant6_copy.loc[dupl] = copy_data

    elif ant3_current_ant3_original > ant3_current_ant6_original:

        copy_data = list(empty_row)
        ant3_copy.loc[dupl] = copy_data

    elif ant3_current_ant3_original == ant3_current_ant6_original:
        if abs(ant3_abx - ant3_abx_o) > abs(ant3_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant3_copy.loc[dupl] = copy_data

        elif abs(ant3_abx - ant3_abx_o) < abs(ant3_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant6_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant4 - ant5

dupl_4_5 = []
ref_dupl_4_5 = []

for i in range(0, len(final)):

    ind4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
    ind4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]

    ind5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
    ind5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

    if math.isnan(ind4_abx):
        continue
    else:

        if ind4_abx == ind5_abx:
            if math.isnan(ind4_abx):
                continue
            else:
                dupl_4_5.append(i)


if len(dupl_4_5) > 0:
    ref_dupl_4_5.append(dupl_4_5[0])

for x, y in zip(dupl_4_5[::], dupl_4_5[1::]):
    if abs(x-y) > 4:
        ref_dupl_4_5.append(x)


if len(dupl_4_5) > 0:
    ref_dupl_4_5.append(dupl_4_5[-1])


if len(ref_dupl_4_5) != 0:
    ref_dupl_4_5 = list(set(ref_dupl_4_5))

ref_dupl_4_5 = sorted(ref_dupl_4_5)


for dupl in tqdm(dupl_4_5):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ant5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## other detections ##

    ant4_abx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ant5_abx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ## thetas ##

    ant4_current_theta = davi.get_vector_angle(
        ant4_abx, ant4_aby, ant4_thx, ant4_thy)
    ant5_current_theta = davi.get_vector_angle(
        ant5_abx, ant5_aby, ant5_thx, ant5_thy)

    ant4_original_theta = davi.get_vector_angle(
        ant4_abx_o, ant4_aby_o, ant4_thx_o, ant4_thy_o)
    ant5_original_theta = davi.get_vector_angle(
        ant5_abx_o, ant5_aby_o, ant5_thx_o, ant5_thy_o)

    ## waffles ##

    if ant4_current_theta == 0 or math.isnan(ant4_current_theta):
        ant4_current_ant4_original = np.nan
        ant4_current_ant5_original = np.nan

    elif ant4_current_theta != 0:
        ant4_current_waffle = davi.create_waffle(
            ant4_abx, ant4_aby, ant4_current_theta, pix)
        ant5_current_waffle = davi.create_waffle(
            ant5_abx, ant5_aby, ant5_current_theta, pix)

        if ant4_original_theta == 0 or math.isnan(ant4_original_theta):
            ant4_current_ant4_original = np.nan

        elif ant4_original_theta != 0:
            ant4_original_waffle = davi.create_waffle(
                ant4_abx_o, ant4_aby_o, ant4_original_theta, pix)
            ant4_current_ant4_original = davi.subtract_colours_2(
                ant4_current_waffle, ant4_original_waffle)

        if ant5_original_theta == 0 or math.isnan(ant5_original_theta):
            ant4_current_ant5_original = np.nan

        elif ant5_original_theta != 0:
            ant5_original_waffle = davi.create_waffle(
                ant5_abx_o, ant5_aby_o, ant5_original_theta, pix)
            ant4_current_ant5_original = davi.subtract_colours_2(
                ant4_current_waffle, ant5_original_waffle)

    ## comparisons ##

    if ant4_current_ant4_original < ant4_current_ant5_original:

        copy_data = list(empty_row)
        ant5_copy.loc[dupl] = copy_data

    elif ant4_current_ant4_original > ant4_current_ant5_original:

        copy_data = list(empty_row)
        ant4_copy.loc[dupl] = copy_data

    elif ant4_current_ant4_original == ant4_current_ant5_original:
        if abs(ant4_abx - ant4_abx_o) > abs(ant4_abx - ant5_abx_o):
            copy_data = list(empty_row)
            ant4_copy.loc[dupl] = copy_data

        elif abs(ant4_abx - ant4_abx_o) < abs(ant4_abx - ant5_abx_o):
            copy_data = list(empty_row)
            ant5_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant4 - ant6

dupl_4_6 = []
ref_dupl_4_6 = []

for i in range(0, len(final)):

    ind4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
    ind4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]

    ind6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i]
    ind6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i]

    if math.isnan(ind4_abx):
        continue
    else:

        if ind4_abx == ind6_abx:
            if math.isnan(ind4_abx):
                continue
            else:
                dupl_4_6.append(i)


if len(dupl_4_6) > 0:
    ref_dupl_4_6.append(dupl_4_6[0])

for x, y in zip(dupl_4_6[::], dupl_4_6[1::]):
    if abs(x-y) > 4:
        ref_dupl_4_6.append(x)


if len(dupl_4_6) > 0:
    ref_dupl_4_6.append(dupl_4_6[-1])


if len(ref_dupl_4_6) != 0:
    ref_dupl_4_6 = list(set(ref_dupl_4_6))

ref_dupl_4_6 = sorted(ref_dupl_4_6)

for dupl in tqdm(dupl_4_6):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant4_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ant6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## other detections ##

    ant4_abx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][dupl]
    ant4_aby_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][dupl]
    ant4_thx_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][dupl]
    ant4_thy_o = ant4[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][dupl]

    ant6_abx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## thetas ##

    ant4_current_theta = davi.get_vector_angle(
        ant4_abx, ant4_aby, ant4_thx, ant4_thy)
    ant6_current_theta = davi.get_vector_angle(
        ant6_abx, ant6_aby, ant6_thx, ant6_thy)

    ant4_original_theta = davi.get_vector_angle(
        ant4_abx_o, ant4_aby_o, ant4_thx_o, ant4_thy_o)
    ant6_original_theta = davi.get_vector_angle(
        ant6_abx_o, ant6_aby_o, ant6_thx_o, ant6_thy_o)

    ## waffles ##

    if ant4_current_theta == 0 or math.isnan(ant4_current_theta):
        ant4_current_ant4_original = np.nan
        ant4_current_ant6_original = np.nan

    elif ant4_current_theta != 0:
        ant4_current_waffle = davi.create_waffle(
            ant4_abx, ant4_aby, ant4_current_theta, pix)
        ant6_current_waffle = davi.create_waffle(
            ant6_abx, ant6_aby, ant6_current_theta, pix)

        if ant4_original_theta == 0 or math.isnan(ant4_original_theta):
            ant4_current_ant4_original = np.nan

        elif ant4_original_theta != 0:
            ant4_original_waffle = davi.create_waffle(
                ant4_abx_o, ant4_aby_o, ant4_original_theta, pix)
            ant4_current_ant4_original = davi.subtract_colours_2(
                ant4_current_waffle, ant4_original_waffle)

        if ant6_original_theta == 0 or math.isnan(ant6_original_theta):
            ant4_current_ant6_original = np.nan

        elif ant6_original_theta != 0:
            ant6_original_waffle = davi.create_waffle(
                ant6_abx_o, ant6_aby_o, ant6_original_theta, pix)
            ant4_current_ant6_original = davi.subtract_colours_2(
                ant4_current_waffle, ant6_original_waffle)

    ## comparisons ##

    if ant4_current_ant4_original < ant4_current_ant6_original:
        copy_data = list(empty_row)
        ant6_copy.loc[dupl] = copy_data

    elif ant4_current_ant4_original > ant4_current_ant6_original:

        copy_data = list(empty_row)
        ant4_copy.loc[dupl] = copy_data

    elif ant4_current_ant4_original == ant4_current_ant6_original:
        if abs(ant4_abx - ant4_abx_o) > abs(ant4_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant4_copy.loc[dupl] = copy_data

        elif abs(ant4_abx - ant4_abx_o) < abs(ant4_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant6_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


# ant5 - ant6

dupl_5_6 = []
ref_dupl_5_6 = []

for i in range(0, len(final)):

    ind5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
    ind5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

    ind6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i]
    ind6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i]

    if math.isnan(ind5_abx):
        continue
    else:

        if ind5_abx == ind6_abx:
            if math.isnan(ind5_abx):
                continue
            else:
                dupl_5_6.append(i)


if len(dupl_5_6) > 0:
    ref_dupl_5_6.append(dupl_5_6[0])

for x, y in zip(dupl_5_6[::], dupl_5_6[1::]):
    if abs(x-y) > 5:
        ref_dupl_5_6.append(x)


if len(dupl_5_6) > 0:
    ref_dupl_5_6.append(dupl_5_6[-1])


if len(ref_dupl_5_6) != 0:
    ref_dupl_5_6 = list(set(ref_dupl_5_6))

ref_dupl_5_6 = sorted(ref_dupl_5_6)


for dupl in tqdm(dupl_5_6):

    frame_id = dupl

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + "-" + str(frame_id) + form

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if ret:
        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(colour_converted)

    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    contrast = ImageEnhance.Contrast(im_output)
    factor_c = 1.2
    im_output_T = contrast.enhance(factor_c)

    pix = im_output_T.load()

    ## currents ##

    ant5_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ant6_abx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy = final[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## other detections ##

    ant5_abx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][dupl]
    ant5_aby_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][dupl]
    ant5_thx_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][dupl]
    ant5_thy_o = ant5[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][dupl]

    ant6_abx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][dupl]
    ant6_aby_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][dupl]
    ant6_thx_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][dupl]
    ant6_thy_o = ant6[(
        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][dupl]

    ## thetas ##

    ant5_current_theta = davi.get_vector_angle(
        ant5_abx, ant5_aby, ant5_thx, ant5_thy)
    ant6_current_theta = davi.get_vector_angle(
        ant6_abx, ant6_aby, ant6_thx, ant6_thy)

    ant5_original_theta = davi.get_vector_angle(
        ant5_abx_o, ant5_aby_o, ant5_thx_o, ant5_thy_o)
    ant6_original_theta = davi.get_vector_angle(
        ant6_abx_o, ant6_aby_o, ant6_thx_o, ant6_thy_o)

    ## waffles ##

    if ant5_current_theta == 0 or math.isnan(ant5_current_theta):
        ant5_current_ant5_original = np.nan
        ant5_current_ant6_original = np.nan

    elif ant5_current_theta != 0:
        ant5_current_waffle = davi.create_waffle(
            ant5_abx, ant5_aby, ant5_current_theta, pix)
        ant6_current_waffle = davi.create_waffle(
            ant6_abx, ant6_aby, ant6_current_theta, pix)

        if ant5_original_theta == 0 or math.isnan(ant5_original_theta):
            ant5_current_ant5_original = np.nan

        elif ant5_original_theta != 0:
            ant5_original_waffle = davi.create_waffle(
                ant5_abx_o, ant5_aby_o, ant5_original_theta, pix)
            ant5_current_ant5_original = davi.subtract_colours_2(
                ant5_current_waffle, ant5_original_waffle)

        if ant6_original_theta == 0 or math.isnan(ant6_original_theta):
            ant5_current_ant6_original = np.nan

        elif ant6_original_theta != 0:
            ant6_original_waffle = davi.create_waffle(
                ant6_abx_o, ant6_aby_o, ant6_original_theta, pix)
            ant5_current_ant6_original = davi.subtract_colours_2(
                ant5_current_waffle, ant6_original_waffle)

    ## comparisons ##

    if ant5_current_ant5_original < ant5_current_ant6_original:

        copy_data = list(empty_row)
        ant6_copy.loc[dupl] = copy_data

    elif ant5_current_ant5_original > ant5_current_ant6_original:

        copy_data = list(empty_row)
        ant5_copy.loc[dupl] = copy_data

    elif ant5_current_ant5_original == ant5_current_ant6_original:
        if abs(ant5_abx - ant5_abx_o) > abs(ant5_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant5_copy.loc[dupl] = copy_data

        elif abs(ant5_abx - ant5_abx_o) < abs(ant5_abx - ant6_abx_o):
            copy_data = list(empty_row)
            ant6_copy.loc[dupl] = copy_data

    else:
        weird_detections.append(dupl)


final1 = pd.concat([ant1_copy, ant2_copy, ant3_copy, ant4_copy,
                   ant5_copy, ant6_copy], axis=1, join='inner')
                   
                   
                   
#file_name = os.path.basename(davi_output_path)


#if dupl_split == 1:


#	final1.to_hdf(os.path.join(davi_output_path, 'davi_split1_'+file_name), key="changed_names", format="fixed")


#elif dupl_split == 2:


#	final1.to_hdf(os.path.join(davi_output_path, 'davi_split2_'+file_name), key="changed_names", format="fixed")

#else:

final1.to_hdf(os.path.join(davi_output_path, 'davi_'+file_name), key="changed_names", format="fixed")
	
	
	
	
print("all finished! Check if davi was successful by making a labelled video! fingers crossed...")

globals().clear()
