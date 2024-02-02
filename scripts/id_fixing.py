#########################################
####### final stitching of splits #######
#########################################

### INCOMPLETE, NEEDS TO BE ADJUSTED FOR NB SPLITS ###


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
import time

#import deeplabcut

#davi_output_path = input("what is the path to the folder containing all output for davi? \n")

#split1 = pd.read_hdf(davi_output_path)

print("this step is to reassign the now complete trajectories to the correct ant by the paint spot colour, so we know who is who for real :)")

time.sleep(2)


print("The most important part of this step is understanding the colours from deeplabcut! When checking the ids with a labeled video, the skeleton colours indicate which individual is which. If it is obvious that is great, if not, for the following questions, just put NA. It may mean watching the entire video, but find the first seen identities for each ant.")


time.sleep(3)


davi_output_path = input("what is the path to davi output? \n")

file_name = os.path.basename(davi_output_path)


time.sleep(1)

print("for each colour, write blue, red, yellow, green, white or pink, no capitals or anything else (I'm too lazy to account for these small differences) :)")

print("LUKE: there is no pink ant, so write purple :)")

time.sleep(1)

ant1 = input("what colour is the paint spot for the ant with the PURPLE detections? \n")

ant2 = input("what colour is the paint spot for the ant with the BLUE detections? \n")

ant3 = input("what colour is the paint spot for the ant with the CYAN detections? \n")

ant4 = input("what colour is the paint spot for the ant with the YELLOW detections? \n")

ant5 = input("what colour is the paint spot for the ant with the ORANGE detections? \n")

ant6 = input("what colour is the paint spot for the ant with the RED detections? \n")


##ant1

time.sleep(2)

print("great, now lets do some fixinnn")

data = pd.read_hdf(davi_output_path) 	

for col in data.columns:
    
    col = list(col)
        
    if col[1] == "ind1":
        
        new_name = col[1].replace("ind1", ant1)

        
        data.rename(columns={col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]},
                   inplace=True)
        
    elif col[1] == "ind2":
        
        new_name = col[1].replace("ind2", ant2)

        
        data.rename(columns={col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]},
                   inplace=True)        
        
    elif col[1] == "ind3":
        
        new_name = col[1].replace("ind3", ant3)

        
        data.rename(columns={col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]},
                   inplace=True)        
        

    elif col[1] == "ind4":
        
        new_name = col[1].replace("ind4", ant4)

        
        data.rename(columns={col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]},
                   inplace=True)    

    elif col[1] == "ind5":
        
        new_name = col[1].replace("ind5", ant5)

        
        data.rename(columns={col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]},
                   inplace=True)    



    elif col[1] == "ind6":
        
        new_name = col[1].replace("ind6", ant6)

        
        data.rename(columns={col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]},
                   inplace=True)    





print(data.head())

time.sleep(2)

print("finished! Enjoy analysing the data :))")

new_name = "ID_fixed_" + file_name

output_path = os.path.dirname(davi_output_path)

data.to_hdf(os.path.join(output_path, new_name), key="changed_names", format="fixed")

time.sleep(1)

os.system("pokemonsay -p Cubone -t 'Arrivaderci' ")






