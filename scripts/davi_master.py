##########################################################
##########################################################
################### DAVI MASTER SCRIPT ###################
##########################################################
##########################################################



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
import gc

import sys
import time
import climage
	
os.system("pokemonsay -p Charmander -t 'It is better to imitate ancient than modern work'")


#output = climage.convert('/home/rb17990/Documents/TEST VIDS/davi scripts/Zoidberg.png')
#print(output)

time.sleep(1)

print('welcome to davi: the restitching pipeline for deeplabcut trajectories!')

time.sleep(2)

print("there is some information needed before we begin...")

time.sleep(2)


original_detections_path = input(
    "first of all, what is the path to the original detections? \n")


video_path = input("aaand what is the path to the original video? \n")


davi_output_path = input("aaaaaaaaand what is the path for the davi files? \n")


data = pd.read_hdf(original_detections_path)


print("great! now we just need to correct for number of ants...")


import nb_ants
time.sleep(2)


original_detections_path = os.path.join(davi_output_path, 'all_6_ants.h5')

time.sleep(1)

print("okay, now time to start the tracking...")

import dist_ant1
import dist_ant2
import dist_ant3
import dist_ant4
import dist_ant5
import dist_ant6

print("TRACKING COMPLETE")

time.sleep(1)

print("now to clean up any leftover errors in assignment...")
import dupl



print("okay, the next step is to create a labeled video from the davi detections, and record down the identities of each ant. Then, in the terminal, type 'python split_stitch.py) and follow that script! :)))" )

time.sleep(3)


print("Thank you and bye bye!")

time.sleep(2)


os.system("pokemonsay -p Squirtle -t 'The noblest pleasure is the joy of understanding'")









gc.collect()
