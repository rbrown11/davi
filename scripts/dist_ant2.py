
###########################################################################
######################### DISTANCE TRACKING ANT 2 #########################
###########################################################################

## imports

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


## paths

print("ANT 2")

path = original_detections_path

#data = pd.read_hdf(path)

data_copy = pd.DataFrame().reindex(columns=data.columns)

ant1 = data.xs('ind1', level='individuals', axis=1, drop_level=False)
ant2 = data.xs('ind2', level='individuals', axis=1, drop_level=False)
ant3 = data.xs('ind3', level='individuals', axis=1, drop_level=False)
ant4 = data.xs('ind4', level='individuals', axis=1, drop_level=False)
ant5 = data.xs('ind5', level='individuals', axis=1, drop_level=False)
ant6 = data.xs('ind6', level='individuals', axis=1, drop_level=False)



ant2_copy = pd.DataFrame().reindex(columns = ant2.columns)
ant2_copy.loc[0] = ant2.loc[0]
full_ids = ["ind2_ind2"]


previous_detection = "first"


for i in tqdm(range(0, len(data))):

    new_row_ant1 = ant1.loc[i]
    new_row_ant2 = ant2.loc[i]
    new_row_ant3 = ant3.loc[i]
    new_row_ant4 = ant4.loc[i]
    new_row_ant5 = ant5.loc[i]
    new_row_ant6 = ant6.loc[i]
            

    if i != 0: 

        ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
        ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]

            
        
        if previous_detection == "first":

            
            ind2_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind2_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]


        elif previous_detection == "ind2_changed_id":
            
        
            ind2_abx_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind2_aby_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]
            
        
        distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, ind2_abx_p, ind2_aby, ind2_aby_p))


        if math.isnan(ind2_abx) or math.isnan(ind2_aby): 
            test2 = list(new_row_ant2)
            ant2_copy.loc[i] = test2

        elif math.isnan(ind2_abx_p) or math.isnan(ind2_aby_p):

            for j in reversed(range(0, i)):
                if previous_detection == "ind2_changed_id":
                    j_ind2_abx = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][j]
                else:    

                    j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][j]

                if math.isnan(j_ind2_abx):
                    continue
                else:
                    
                    if previous_detection == "ind2_changed_id":
                        
                        j_ind2_abx = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][j]
                        j_ind2_aby = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][j]
    
                    else:
                        j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][j]
                        j_ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][j]
        
                    break

            if j != 0:
                distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, j_ind2_abx, ind2_aby, j_ind2_aby))
                
            else:
            	distance_ind2_ind2 = 11
#                test2 = list(new_row_ant2)
#                ant2_copy.loc[i] = test2



        if distance_ind2_ind2 < 10:
            test2 = list(new_row_ant2)
            ant2_copy.loc[i] = test2



        elif distance_ind2_ind2 > 10:

            ## compare to ant 1 from the original detections ##

            ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i-1]
            ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind1_abx_p) or math.isnan(ind1_aby_p):

                for k in reversed(range(0, i)):
                    j_ind1_abx = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]

                    if math.isnan(j_ind1_abx):
                        
                        continue

                    else:
                        j_ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]
                        j_ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind1_original = abs(davi.find_distance(j_ind2_abx, j_ind1_abx_p, j_ind2_aby, j_ind1_aby_p))

                    else:

                        distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, j_ind1_abx_p, ind2_aby, j_ind1_aby_p))

                else:
                    distance_ind2_ind1_original = 1000

            else:
                distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, ind1_abx_p, ind2_aby, ind1_aby_p))
                
            
            
            ## compare to ant 3 ##

            ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i-1]
            ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind3_abx_p) or math.isnan(ind3_aby_p):

                for k in reversed(range(0, i)):
                    j_ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]

                    if math.isnan(j_ind3_abx):
                        
                        continue

                    else:
                        j_ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]
                        j_ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind3 = abs(davi.find_distance(j_ind2_abx, j_ind3_abx_p, j_ind2_aby, j_ind3_aby_p))

                    else:

                        distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, j_ind3_abx_p, ind2_aby, j_ind3_aby_p))

                else:
                    distance_ind2_ind3 = 1000

            else:
                distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, ind3_abx_p, ind2_aby, ind3_aby_p))
                
                
                
                
            ## compare to ant 4 ##

            ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i-1]
            ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind4_abx_p) or math.isnan(ind4_aby_p):

                for k in reversed(range(0, i)):

                    j_ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]

                    if math.isnan(j_ind4_abx):
                        
                        continue

                    else:
                        

                        j_ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]
                        j_ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    
                    if math.isnan(ind2_abx):

                        distance_ind2_ind4 = abs(davi.find_distance(j_ind2_abx, j_ind4_abx_p, j_ind2_aby, j_ind4_aby_p))

                    else:

                        distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, j_ind4_abx_p, ind2_aby, j_ind4_aby_p))

                else:
                    distance_ind2_ind4 = 1000

            else:
                distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, ind4_abx_p, ind2_aby, ind4_aby_p))


                
                
            ## compare to ant 5 ##

            ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i-1]
            ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind5_abx_p) or math.isnan(ind5_aby_p):

                for k in reversed(range(0, i)):

                    j_ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]

                    if math.isnan(j_ind5_abx):
                        
                        continue

                    else:

                        j_ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]
                        j_ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind5 = abs(davi.find_distance(j_ind2_abx, j_ind5_abx_p, j_ind2_aby, j_ind5_aby_p))

                    else:

                        distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, j_ind5_abx_p, ind2_aby, j_ind5_aby_p))

                else:
                    distance_ind2_ind5 = 1000

            else:
                distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, ind5_abx_p, ind2_aby, ind5_aby_p))


                
                
                
            ## compare to ant 6 ##

            ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i-1]
            ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind6_abx_p) or math.isnan(ind6_aby_p):

                for k in reversed(range(0, i)):
                    
                    j_ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]

                    if math.isnan(j_ind6_abx):
                        
                        continue

                    else:

                        j_ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]
                        j_ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind6 = abs(davi.find_distance(j_ind2_abx, j_ind6_abx_p, j_ind2_aby, j_ind6_aby_p))

                    else:

                        distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, j_ind6_abx_p, ind2_aby, j_ind6_aby_p))

                else:
                    distance_ind2_ind6 = 1000

            else:
                distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, ind6_abx_p, ind2_aby, ind6_aby_p))




            other_ant_distances = {  "ind2_ind2" : distance_ind2_ind2,
                                     "ind2_ind5" : distance_ind2_ind5,
                                     "ind2_ind3" : distance_ind2_ind3,
                                      "ind2_ind4" : distance_ind2_ind4,
                                      "ind2_ind6" : distance_ind2_ind6,
                                   "ind2_ind1" : distance_ind2_ind1_original
                                  }


            optimal_match_value = min(other_ant_distances.values())

            multi_distance = []


            #### FIX THIS SECTION ####

            for dist in other_ant_distances:
                if other_ant_distances[dist] == optimal_match_value:
                    multi_distance.append(dist)

            if len(multi_distance) > 1:
                optimal_match = full_ids[-1]
            else:
                optimal_match = min(other_ant_distances, key=other_ant_distances.get)
                
            
            full_ids.append(optimal_match)



            if optimal_match == "ind2_ind1":

                test2 = list(new_row_ant1)
                ant2_copy.loc[i] = test2
                previous_detection = "ind2_changed_id"

            if optimal_match == "ind2_ind2":
                
                test2 = list(new_row_ant2)
                ant2_copy.loc[i] = test2

            if optimal_match == 'ind2_ind3':
                
                test = list(new_row_ant3)
                ant2_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind4':
                
                test = list(new_row_ant4)
                ant2_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind5':
                
                test = list(new_row_ant5)
                ant2_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind6':
                
                test = list(new_row_ant6)
                ant2_copy.loc[i] = test
                previous_detection = "ind2_changed_id"
            


print("ant2 done") 



### ant 3 detections


ant3_copy = pd.DataFrame().reindex(columns = ant3.columns)
ant3_copy.loc[0] = ant3.loc[0]

full_ids = []


previous_detection = "first"


for i in tqdm(range(0, len(data))):

    
    new_row_ant1 = ant1.loc[i]    
    new_row_ant2 = ant2.loc[i]
    new_row_ant3 = ant3.loc[i]
    new_row_ant4 = ant4.loc[i]
    new_row_ant5 = ant5.loc[i]
    new_row_ant6 = ant6.loc[i]
            


    if i != 0: 
            

        ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
        ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]


    
        if previous_detection == "first":
    

            ind2_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i-1]
            ind2_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i-1]



            
        if previous_detection == "ind2_changed_id":
            
        
            ind2_abx_p = ant3_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i-1]
            ind2_aby_p = ant3_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i-1]
            
        
        distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, ind2_abx_p, ind2_aby, ind2_aby_p))


        if math.isnan(ind2_abx) or math.isnan(ind2_aby): 
            test2 = list(new_row_ant3)
            ant3_copy.loc[i] = test2

        elif math.isnan(ind2_abx_p) or math.isnan(ind2_aby_p):

            for j in reversed(range(0, i)):
                
                if previous_detection == "ind2_changed_id":
                    j_ind2_abx = ant3_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][j]
                else:    

                    j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][j]

                if math.isnan(j_ind2_abx):
                    continue
                else:
                    
                    if previous_detection == "ind2_changed_id":
                        
                        j_ind2_abx = ant3_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][j]
                        j_ind2_aby = ant3_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][j]
    
                    else:
                        j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][j]
                        j_ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][j]
        
                    break

            if j != 0:
                distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, j_ind2_abx, ind2_aby, j_ind2_aby))
                
            else:
            	distance_ind2_ind2 = 11
#                test2 = list(new_row_ant3)
#                ant3_copy.loc[i] = test2



        if distance_ind2_ind2 < 10:
            test2 = list(new_row_ant3)
            ant3_copy.loc[i] = test2



        elif distance_ind2_ind2 > 10:


            
                        ## compare to ant 1 from the original detections ##

            ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i-1]
            ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind1_abx_p) or math.isnan(ind1_aby_p):

                for k in reversed(range(0, i)):
                    j_ind1_abx = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]

                    if math.isnan(j_ind1_abx):
                        
                        continue

                    else:
                        j_ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]
                        j_ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    
                    if math.isnan(ind2_abx):

                        distance_ind2_ind1_original = abs(davi.find_distance(j_ind2_abx, j_ind1_abx_p, j_ind2_aby, j_ind1_aby_p))

                    else:

                        distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, j_ind1_abx_p, ind2_aby, j_ind1_aby_p))

                else:
                    distance_ind2_ind1_original = 1000

            else:
                distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, ind1_abx_p, ind2_aby, ind1_aby_p))
            
            ## compare to ant 2 ##

            ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind3_abx_p) or math.isnan(ind3_aby_p):

                for k in reversed(range(0, i)):
                    j_ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]

                    if math.isnan(j_ind3_abx):
                        
                        continue

                    else:
                        j_ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]
                        j_ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    
                    if math.isnan(ind2_abx):

                        distance_ind2_ind3 = abs(davi.find_distance(j_ind2_abx, j_ind3_abx_p, j_ind2_aby, j_ind3_aby_p))

                    else:

                        distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, j_ind3_abx_p, ind2_aby, j_ind3_aby_p))

                else:
                    distance_ind2_ind3 = 1000
            else:
                distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, ind3_abx_p, ind2_aby, ind3_aby_p))
                
                
                
                
            ## compare to ant 4 ##

            ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i-1]
            ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind4_abx_p) or math.isnan(ind4_aby_p):

                for k in reversed(range(0, i)):

                    j_ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]

                    if math.isnan(j_ind4_abx):
                        
                        continue

                    else:
                        

                        j_ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]
                        j_ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind4 = abs(davi.find_distance(j_ind2_abx, j_ind4_abx_p, j_ind2_aby, j_ind4_aby_p))

                    else:

                        distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, j_ind4_abx_p, ind2_aby, j_ind4_aby_p))

                else:
                    distance_ind2_ind4 = 1000

            else:
                distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, ind4_abx_p, ind2_aby, ind4_aby_p))


                
                
            ## compare to ant 5 ##

            ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i-1]
            ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind5_abx_p) or math.isnan(ind5_aby_p):

                for k in reversed(range(0, i)):

                    j_ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]

                    if math.isnan(j_ind5_abx):
                        
                        continue

                    else:


                        j_ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]
                        j_ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    
                    if math.isnan(ind2_abx):

                        distance_ind2_ind5 = abs(davi.find_distance(j_ind2_abx, j_ind5_abx_p, j_ind2_aby, j_ind5_aby_p))

                    else:

                        distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, j_ind5_abx_p, ind2_aby, j_ind5_aby_p))

                else:
                    distance_ind2_ind5 = 1000
            else:
                distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, ind5_abx_p, ind2_aby, ind5_aby_p))


                
                
                
            ## compare to ant 6 ##

            ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i-1]
            ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind6_abx_p) or math.isnan(ind6_aby_p):

                for k in reversed(range(0, i)):

                    j_ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]

                    if math.isnan(j_ind6_abx):
                        
                        continue

                    else:
                        
                        j_ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]
                        j_ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    

                    if math.isnan(ind2_abx):

                        distance_ind2_ind6 = abs(davi.find_distance(j_ind2_abx, j_ind6_abx_p, j_ind2_aby, j_ind6_aby_p))

                    else:

                        distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, j_ind6_abx_p, ind2_aby, j_ind6_aby_p))

                else:
                    distance_ind2_ind6 = 1000

            else:
                distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, ind6_abx_p, ind2_aby, ind6_aby_p))




            other_ant_distances = { "ind2_ind2" : distance_ind2_ind2,
                                      "ind2_ind3" : distance_ind2_ind3,
                                      "ind2_ind4" : distance_ind2_ind4,
                                      "ind2_ind5" : distance_ind2_ind5,
                                      "ind2_ind6" : distance_ind2_ind6,
                                     "ind2_ind1" : distance_ind2_ind1_original

                                  }


            optimal_match = min(other_ant_distances, key=other_ant_distances.get)
            optimal_match_value = min(other_ant_distances.values())

            multi_distance = []


            ### FIX THIS SECTION ####

            for dist in other_ant_distances:
                if other_ant_distances[dist] == optimal_match_value:

                    multi_distance.append(dist)

            if len(multi_distance) > 1:
                optimal_match = full_ids[-1]
            else:
                optimal_match_value = min(other_ant_distances.values())

            full_ids.append(optimal_match)


            if optimal_match == "ind2_ind2":
                
                test2 = list(new_row_ant3)
                ant3_copy.loc[i] = test2

            if optimal_match == 'ind2_ind3':
                
                test = list(new_row_ant2)
                ant3_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind4':
                
                test = list(new_row_ant4)
                ant3_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind5':
                
                test = list(new_row_ant5)
                ant3_copy.loc[i] = test
                previous_detection = "ind2_changed_id"
  
            if optimal_match == 'ind2_ind6':
                
                test = list(new_row_ant6)
                ant3_copy.loc[i] = test
                previous_detection = "ind2_changed_id"



            if optimal_match == "ind2_ind1":

                test2 = list(new_row_ant1)
                ant3_copy.loc[i] = test2
            



print("ant 3 done")


## ant 4 detections


ant4_copy = pd.DataFrame().reindex(columns = ant4.columns)
ant4_copy.loc[0] = ant4.loc[0]

full_ids = []

previous_detection = "first"


for i in tqdm(range(0, len(data))):

    new_row_ant1 = ant1.loc[i]
    new_row_ant2 = ant2.loc[i]
    new_row_ant3 = ant3.loc[i]
    new_row_ant4 = ant4.loc[i]
    new_row_ant5 = ant5.loc[i]
    new_row_ant6 = ant6.loc[i]

    if i != 0: 

        ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
        ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]


    
        if previous_detection == "first":
    

            ind2_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i-1]
            ind2_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i-1]

            
        elif previous_detection == "ind2_changed_id":
            
        
            ind2_abx_p = ant4_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i-1]
            ind2_aby_p = ant4_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i-1]
            
        
        distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, ind2_abx_p, ind2_aby, ind2_aby_p))


        if math.isnan(ind2_abx) or math.isnan(ind2_aby): 
            test2 = list(new_row_ant4)
            ant4_copy.loc[i] = test2

        elif math.isnan(ind2_abx_p) or math.isnan(ind2_aby_p):

            for j in reversed(range(0, i)):
                
                if previous_detection == "ind2_changed_id":
                    j_ind2_abx = ant4_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][j]
                else:    

                    j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][j]

                if math.isnan(j_ind2_abx):
                    continue
                else:

                    
                    if previous_detection == "ind2_changed_id":
                        
                        j_ind2_abx = ant4_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][j]
                        j_ind2_aby = ant4_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][j]
    
                    else:
                        j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][j]
                        j_ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][j]
        
                    break

            if j != 0:
                distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, j_ind2_abx, ind2_aby, j_ind2_aby))
                
            else:
            	distance_ind2_ind2 = 11
#                test2 = list(new_row_ant4)
#                ant4_copy.loc[i] = test2



        if distance_ind2_ind2 < 10:
            test2 = list(new_row_ant4)
            ant4_copy.loc[i] = test2



        elif distance_ind2_ind2 > 10:


                                   ## compare to ant 1 from the original detections ##

            ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i-1]
            ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind1_abx_p) or math.isnan(ind1_aby_p):

                for k in reversed(range(0, i)):
                    j_ind1_abx = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]

                    if math.isnan(j_ind1_abx):
                        
                        continue

                    else:
                        j_ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]
                        j_ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind1_original = abs(davi.find_distance(j_ind2_abx, j_ind1_abx_p, j_ind2_aby, j_ind1_aby_p))

                    else:

                        distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, j_ind1_abx_p, ind2_aby, j_ind1_aby_p))

                else:
                    distance_ind2_ind1_original = 1000

            else:
                distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, ind1_abx_p, ind2_aby, ind1_aby_p))
            
            
            
            ## compare to ant 3 ##

            ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i-1]
            ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind3_abx_p) or math.isnan(ind3_aby_p):

                for k in reversed(range(0, i)):
                    j_ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]

                    if math.isnan(j_ind3_abx):
                        
                        continue

                    else:
                        j_ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]
                        j_ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind3 = abs(davi.find_distance(j_ind2_abx, j_ind3_abx_p, j_ind2_aby, j_ind3_aby_p))

                    else:

                        distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, j_ind3_abx_p, ind2_aby, j_ind3_aby_p))

                else:
                    distance_ind2_ind3 = 1000
            else:
                distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, ind3_abx_p, ind2_aby, ind3_aby_p))
                
                
                
                
            ## compare to ant 4 ##

            ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind4_abx_p) or math.isnan(ind4_aby_p):

                for k in reversed(range(0, i)):

                    j_ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]

                    if math.isnan(j_ind4_abx):
                        
                        continue

                    else:
                        

                        j_ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]
                        j_ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind4 = abs(davi.find_distance(j_ind2_abx, j_ind4_abx_p, j_ind2_aby, j_ind4_aby_p))

                    else:

                        distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, j_ind4_abx_p, ind2_aby, j_ind4_aby_p))

                else:
                    distance_ind2_ind4 = 1000
            else:
                distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, ind4_abx_p, ind2_aby, ind4_aby_p))


                
                
            ## compare to ant 5 ##

            ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i-1]
            ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind5_abx_p) or math.isnan(ind5_aby_p):

                for k in reversed(range(0, i)):

                    j_ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]

                    if math.isnan(j_ind5_abx):
                        
                        continue

                    else:

                        j_ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]
                        j_ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:


                    if math.isnan(ind2_abx):

                        distance_ind2_ind5 = abs(davi.find_distance(j_ind2_abx, j_ind5_abx_p, j_ind2_aby, j_ind5_aby_p))

                    else:

                        distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, j_ind5_abx_p, ind2_aby, j_ind5_aby_p))

                else:
                    distance_ind2_ind5 = 1000

            else:
                distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, ind5_abx_p, ind2_aby, ind5_aby_p))


                
                
                
            ## compare to ant 6 ##

            ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i-1]
            ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind6_abx_p) or math.isnan(ind6_aby_p):

                for k in reversed(range(0, i)):

                    j_ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]

                    if math.isnan(j_ind6_abx):
                        
                        continue

                    else:

                        j_ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]
                        j_ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind6 = abs(davi.find_distance(j_ind2_abx, j_ind6_abx_p, j_ind2_aby, j_ind6_aby_p))

                    else:

                        distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, j_ind6_abx_p, ind2_aby, j_ind6_aby_p))

                else:
                    distance_ind2_ind6 = 1000

            else:
                distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, ind6_abx_p, ind2_aby, ind6_aby_p))




            other_ant_distances = { "ind2_ind2" : distance_ind2_ind2,
                                      "ind2_ind3" : distance_ind2_ind3,
                                      "ind2_ind4" : distance_ind2_ind4,
                                      "ind2_ind5" : distance_ind2_ind5,
                                      "ind2_ind6" : distance_ind2_ind6,
                                      "ind2_ind1" : distance_ind2_ind1_original

                                  }

            
            optimal_match_value = min(other_ant_distances.values())

            multi_distance = []

            ### FIX THIS SECTION ####

            for dist in other_ant_distances:
                if other_ant_distances[dist] == optimal_match_value:

                    multi_distance.append(dist)

            if len(multi_distance) > 1:
                optimal_match = full_ids[-1]
            else:
                optimal_match = min(other_ant_distances, key=other_ant_distances.get)

            full_ids.append(optimal_match)



            if optimal_match == "ind2_ind2":
                
                test2 = list(new_row_ant4)
                ant4_copy.loc[i] = test2

            if optimal_match == 'ind2_ind3':
                
                test = list(new_row_ant3)
                ant4_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind4':
                
                test = list(new_row_ant2)
                ant4_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind5':
                
                test = list(new_row_ant5)
                ant4_copy.loc[i] = test
                previous_detection = "ind2_changed_id"
  
            if optimal_match == 'ind2_ind6':
                
                test = list(new_row_ant6)
                ant4_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

            if optimal_match == "ind2_ind1":

                test2 = list(new_row_ant1)
                ant4_copy.loc[i] = test2

                previous_detection = "ind2_changed_id"
               

            



print("ant 4 done")


### ant 5 detections

ant5_copy = pd.DataFrame().reindex(columns = ant5.columns)
ant5_copy.loc[0] = ant5.loc[0]



previous_detection = "first"
full_ids = []

for i in tqdm(range(0, len(data))):
    
    new_row_ant1 = ant1.loc[i]
    new_row_ant2 = ant2.loc[i]
    new_row_ant3 = ant3.loc[i]
    new_row_ant4 = ant4.loc[i]
    new_row_ant5 = ant5.loc[i]
    new_row_ant6 = ant6.loc[i]
            

    if i != 0: 
        

        ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
        ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

        
        if previous_detection == "first":

            ind2_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i-1]
            ind2_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i-1]
            
        elif previous_detection == "ind2_changed_id":
    
            ind2_abx_p = ant5_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i-1]
            ind2_aby_p = ant5_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i-1]
            
        
        distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, ind2_abx_p, ind2_aby, ind2_aby_p))



        if math.isnan(ind2_abx) or math.isnan(ind2_aby): 
            test2 = list(new_row_ant5)
            ant5_copy.loc[i] = test2

        elif math.isnan(ind2_abx_p) or math.isnan(ind2_aby_p):

            for j in reversed(range(0, i)):
                
                if previous_detection == "ind2_changed_id":
                    j_ind2_abx = ant5_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][j]
                else:    

                    j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][j]

                if math.isnan(j_ind2_abx):
                    continue
                else:
                    
                    if previous_detection == "ind2_changed_id":
                        
                        j_ind2_abx = ant5_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][j]
                        j_ind2_aby = ant5_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][j]
    
                    else:
                        j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][j]
                        j_ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][j]
        
                    break

            if j != 0:
                distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, j_ind2_abx, ind2_aby, j_ind2_aby))
                
            else:
            	distance_ind2_ind2 =11
#                test2 = list(new_row_ant5)
#                ant5_copy.loc[i] = test2


        if distance_ind2_ind2 < 10:
            test2 = list(new_row_ant5)
            ant5_copy.loc[i] = test2

        elif distance_ind2_ind2 > 10:

                                   ## compare to ant 1 from the original detections ##

            ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i-1]
            ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind1_abx_p) or math.isnan(ind1_aby_p):

                for k in reversed(range(0, i)):
                    j_ind1_abx = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]

                    if math.isnan(j_ind1_abx):
                        
                        continue

                    else:
                        j_ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]
                        j_ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    
                    if math.isnan(ind2_abx):

                        distance_ind2_ind1_original = abs(davi.find_distance(j_ind2_abx, j_ind1_abx_p, j_ind2_aby, j_ind1_aby_p))

                    else:

                        distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, j_ind1_abx_p, ind2_aby, j_ind1_aby_p))

                else:
                    distance_ind2_ind1_original = 1000

            else:
                distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, ind1_abx_p, ind2_aby, ind1_aby_p))
            
            
            
            ## compare to ant 3 ##

            ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i-1]
            ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind3_abx_p) or math.isnan(ind3_aby_p):

                for k in reversed(range(0, i)):
                    j_ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]

                    if math.isnan(j_ind3_abx):
                        
                        continue

                    else:
                        j_ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]
                        j_ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:
                    
                    if math.isnan(ind2_abx):

                        distance_ind2_ind3 = abs(davi.find_distance(j_ind2_abx, j_ind3_abx_p, j_ind2_aby, j_ind3_aby_p))

                    else:

                        distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, j_ind3_abx_p, ind2_aby, j_ind3_aby_p))

                else:
                    distance_ind2_ind3 = 1000
            else:
                distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, ind3_abx_p, ind2_aby, ind3_aby_p))
                
                
                
                
            ## compare to ant 4 ##

            ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i-1]
            ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind4_abx_p) or math.isnan(ind4_aby_p):

                for k in reversed(range(0, i)):

                    j_ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]

                    if math.isnan(j_ind4_abx):
                        
                        continue

                    else:
                        

                        j_ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]
                        j_ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind4 = abs(davi.find_distance(j_ind2_abx, j_ind4_abx_p, j_ind2_aby, j_ind4_aby_p))

                    else:

                        distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, j_ind4_abx_p, ind2_aby, j_ind4_aby_p))

                else:
                    distance_ind2_ind4 = 1000
            else:
                distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, ind4_abx_p, ind2_aby, ind4_aby_p))


                
                
            ## compare to ant 5 ##

            ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind5_abx_p) or math.isnan(ind5_aby_p):

                for k in reversed(range(0, i)):

                    j_ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]

                    if math.isnan(j_ind5_abx):
                        
                        continue

                    else:
                        
                        j_ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]
                        j_ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind5 = abs(davi.find_distance(j_ind2_abx, j_ind5_abx_p, j_ind2_aby, j_ind5_aby_p))

                    else:

                        distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, j_ind5_abx_p, ind2_aby, j_ind5_aby_p))

                else:
                    distance_ind2_ind5 = 1000
            else:
                distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, ind5_abx_p, ind2_aby, ind5_aby_p))


                
                
                
            ## compare to ant 6 ##

            ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i-1]
            ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind6_abx_p) or math.isnan(ind6_aby_p):

                for k in reversed(range(0, i)):

                    j_ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]

                    if math.isnan(j_ind6_abx):
                        
                        continue

                    else:
                        
                        j_ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][k]
                        j_ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind6 = abs(davi.find_distance(j_ind2_abx, j_ind6_abx_p, j_ind2_aby, j_ind6_aby_p))

                    else:

                        distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, j_ind6_abx_p, ind2_aby, j_ind6_aby_p))

                else:
                    distance_ind2_ind6 = 1000

            else:
                distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, ind6_abx_p, ind2_aby, ind6_aby_p))




            other_ant_distances = { "ind2_ind2" : distance_ind2_ind2,
                                      "ind2_ind3" : distance_ind2_ind3,
                                      "ind2_ind4" : distance_ind2_ind4,
                                      "ind2_ind5" : distance_ind2_ind5,
                                      "ind2_ind6" : distance_ind2_ind6,
                                       "ind2_ind1" : distance_ind2_ind1_original
                                  }


            optimal_match_value = min(other_ant_distances.values())

            multi_distance = []


            ### FIX THIS SECTION ####

            for dist in other_ant_distances:
                if other_ant_distances[dist] == optimal_match_value:

                    multi_distance.append(dist)

            if len(multi_distance) > 1:
                optimal_match = full_ids[-1]
            else:
                optimal_match = min(other_ant_distances, key=other_ant_distances.get)

                
            full_ids.append(optimal_match)

            if optimal_match == "ind2_ind2":
                
                test2 = list(new_row_ant5)
                ant5_copy.loc[i] = test2

            if optimal_match == 'ind2_ind3':
                
                test = list(new_row_ant3)
                ant5_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind4':
                
                test = list(new_row_ant4)
                ant5_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind5':
                
                test = list(new_row_ant2)
                ant5_copy.loc[i] = test
                previous_detection = "ind2_changed_id"
  
            if optimal_match == 'ind2_ind6':
                
                test = list(new_row_ant6)
                ant5_copy.loc[i] = test
                previous_detection = "ind2_changed_id"


            if optimal_match == "ind2_ind1":

                test2 = list(new_row_ant1)
                ant5_copy.loc[i] = test2

                previous_detection = "ind2_changed_id"
            

print("ant 5 done")


## ant 6 detections


ant6_copy = pd.DataFrame().reindex(columns = ant6.columns)
ant6_copy.loc[0] = ant6.loc[0]



previous_detection = "first"


for i in tqdm(range(0, len(data))):

    new_row_ant1 = ant1.loc[i]
    new_row_ant2 = ant2.loc[i]
    new_row_ant3 = ant3.loc[i]
    new_row_ant4 = ant4.loc[i]
    new_row_ant5 = ant5.loc[i]
    new_row_ant6 = ant6.loc[i]
            
    if i != 0: 


        ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
        ind6_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]
        
        if previous_detection == "first":
            
            ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]
            
        if previous_detection == "ind2_changed_id":
            
            ind2_abx_p = ant6_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i-1]
            ind2_aby_p = ant6_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i-1]
            
        
        distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, ind2_abx_p, ind2_aby, ind2_aby_p))


        if math.isnan(ind2_abx) or math.isnan(ind2_aby): 
            test2 = list(new_row_ant6)
            ant6_copy.loc[i] = test2

        elif math.isnan(ind2_abx_p) or math.isnan(ind2_aby_p):

            for j in reversed(range(0, i)):
                
                if previous_detection == "ind2_changed_id":
                    j_ind2_abx = ant6_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][j]
                else:    

                    j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][j]

                if math.isnan(j_ind2_abx):
                    continue
                else:
                    
                    if previous_detection == "ind2_changed_id":
                        
                        j_ind2_abx = ant6_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][j]
                        j_ind2_aby = ant6_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][j]
    
                    else:
                        j_ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][j]
                        j_ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][j]
        
                    break

            if j != 0:
                distance_ind2_ind2 = abs(davi.find_distance(ind2_abx, j_ind2_abx, ind2_aby, j_ind2_aby))
                
            else:
            	distance_ind2_ind2 = 11
#                test2 = list(new_row_ant6)
#                ant6_copy.loc[i] = test2



        if distance_ind2_ind2 < 10:
            test2 = list(new_row_ant6)
            ant6_copy.loc[i] = test2



        elif distance_ind2_ind2 > 10:


                                              ## compare to ant 1 from the original detections ##

            ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i-1]
            ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind1_abx_p) or math.isnan(ind1_aby_p):

                for k in reversed(range(0, i)):
                    j_ind1_abx = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]

                    if math.isnan(j_ind1_abx):
                        
                        continue

                    else:
                        j_ind1_abx_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][k]
                        j_ind1_aby_p = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind1_original = abs(davi.find_distance(j_ind2_abx, j_ind1_abx_p, j_ind2_aby, j_ind1_aby_p))

                    else:

                        distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, j_ind1_abx_p, ind2_aby, j_ind1_aby_p))

                else:
                    distance_ind2_ind1_original = 1000

            else:
                distance_ind2_ind1_original = abs(davi.find_distance(ind2_abx, ind1_abx_p, ind2_aby, ind1_aby_p))
            
            
            
            ## compare to ant 3 ##

            ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i-1]
            ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind3_abx_p) or math.isnan(ind3_aby_p):

                for k in reversed(range(0, i)):
                    j_ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]

                    if math.isnan(j_ind3_abx):
                        
                        continue

                    else:
                        j_ind3_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][k]
                        j_ind3_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind3 = abs(davi.find_distance(j_ind2_abx, j_ind3_abx_p, j_ind2_aby, j_ind3_aby_p))

                    else:

                        distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, j_ind3_abx_p, ind2_aby, j_ind3_aby_p))

                else:
                    distance_ind2_ind3 = 1000

            else:
                distance_ind2_ind3 = abs(davi.find_distance(ind2_abx, ind3_abx_p, ind2_aby, ind3_aby_p))
                
                
                
                
            ## compare to ant 4 ##

            ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i-1]
            ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind4_abx_p) or math.isnan(ind4_aby_p):

                for k in reversed(range(0, i)):

                    j_ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]

                    if math.isnan(j_ind4_abx):
                        
                        continue

                    else:
                        

                        j_ind4_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][k]
                        j_ind4_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind4 = abs(davi.find_distance(j_ind2_abx, j_ind4_abx_p, j_ind2_aby, j_ind4_aby_p))

                    else:

                        distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, j_ind4_abx_p, ind2_aby, j_ind4_aby_p))

                else:
                    distance_ind2_ind4 = 1000
            else:
                distance_ind2_ind4 = abs(davi.find_distance(ind2_abx, ind4_abx_p, ind2_aby, ind4_aby_p))


                
                
            ## compare to ant 5 ##

            ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i-1]
            ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind5_abx_p) or math.isnan(ind5_aby_p):

                for k in reversed(range(0, i)):

                    j_ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]

                    if math.isnan(j_ind5_abx):
                        
                        continue

                    else:

                        j_ind5_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][k]
                        j_ind5_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind5 = abs(davi.find_distance(j_ind2_abx, j_ind5_abx_p, j_ind2_aby, j_ind5_aby_p))

                    else:

                        distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, j_ind5_abx_p, ind2_aby, j_ind5_aby_p))

                else:
                    distance_ind2_ind5 = 1000

            else:
                distance_ind2_ind5 = abs(davi.find_distance(ind2_abx, ind5_abx_p, ind2_aby, ind5_aby_p))


                
                
                
            ## compare to ant 6 ##

            ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i-1]
            ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i-1]
                

            if math.isnan(ind6_abx_p) or math.isnan(ind6_aby_p):

                for k in reversed(range(0, i)):

                    j_ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]

                    if math.isnan(j_ind6_abx):
                        
                        continue

                    else:

                        j_ind6_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][k]
                        j_ind6_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][k]
                        
                        break 

                if k!= 0:

                    if math.isnan(ind2_abx):

                        distance_ind2_ind6 = abs(davi.find_distance(j_ind2_abx, j_ind6_abx_p, j_ind2_aby, j_ind6_aby_p))

                    else:

                        distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, j_ind6_abx_p, ind2_aby, j_ind6_aby_p))

                else:
                    distance_ind2_ind6 = 1000

            else:
                distance_ind2_ind6 = abs(davi.find_distance(ind2_abx, ind6_abx_p, ind2_aby, ind6_aby_p))




            other_ant_distances = { "ind2_ind2" : distance_ind2_ind2,
                                      "ind2_ind3" : distance_ind2_ind3,
                                      "ind2_ind4" : distance_ind2_ind4,
                                      "ind2_ind5" : distance_ind2_ind5,
                                      "ind2_ind6" : distance_ind2_ind6,
                                      "ind2_ind1" : distance_ind2_ind1_original
                                  }


            optimal_match_value = min(other_ant_distances.values())

            multi_distance = []

            ### FIX THIS SECTION ####

            for dist in other_ant_distances:
                if other_ant_distances[dist] == optimal_match_value:

                    multi_distance.append(dist)

            if len(multi_distance) > 1:
                optimal_match = full_ids[-1]
            else:
                optimal_match = min(other_ant_distances, key=other_ant_distances.get)


            full_ids.append(optimal_match)

            if optimal_match == "ind2_ind2":
                
                test2 = list(new_row_ant6)
                ant6_copy.loc[i] = test2

            if optimal_match == 'ind2_ind3':
                
                test = list(new_row_ant3)
                ant6_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind4':
                
                test = list(new_row_ant4)
                ant6_copy.loc[i] = test
                previous_detection = "ind2_changed_id"

                
            if optimal_match == 'ind2_ind5':
                
                test = list(new_row_ant5)
                ant6_copy.loc[i] = test
                previous_detection = "ind2_changed_id"
  
            if optimal_match == 'ind2_ind6':
                
                test = list(new_row_ant2)
                ant6_copy.loc[i] = test
                previous_detection = "ind2_changed_id"


            if optimal_match == "ind2_ind1":

                test2 = list(new_row_ant1)
                ant6_copy.loc[i] = test2
            
            
            



print("ant 6 done")    


df1 = pd.concat([ant2_copy, ant3_copy, ant4_copy, ant5_copy, ant6_copy], axis = 1, join = 'inner')
 
#df1.to_hdf('stage1_ant2.h5', key="changed_names", format="fixed")


print("ant 2 detections and comparisons are complete")


time.sleep(1)

print("now to compare colours :)")


#vidname = os.path.basename(video_path)


#vidtype = '.mp4'

#video_name = video_name + vidtype

#video_path = os.path.join(video_path, video_name)

#vidname = video_name
 
ant1 = data.xs('ind1', level='individuals', axis=1, drop_level=False)
ant2 = data.xs('ind2', level='individuals', axis=1, drop_level=False)
ant3 = data.xs('ind3', level='individuals', axis=1, drop_level=False)
ant4 = data.xs('ind4', level='individuals', axis=1, drop_level=False)
ant5 = data.xs('ind5', level='individuals', axis=1, drop_level=False)
ant6 = data.xs('ind6', level='individuals', axis=1, drop_level=False)


ant2_copy = df1.xs('ind2', level='individuals', axis=1, drop_level=False)
 
 
 
ant2_ant3_mix = []
ant2_ant4_mix = []
ant2_ant5_mix = []
ant2_ant6_mix = []

ant2_ant1_mix= []
refined_ant2_ant1_mix = []

refined_ant2_ant3_mix = []
refined_ant2_ant4_mix = []
refined_ant2_ant5_mix = []
refined_ant2_ant6_mix = []


for i in tqdm(range(0, len(df1))):

    ind1_abx = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i]
    ind1_aby = ant1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][i]
    
    ind2_abx = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
    ind2_aby = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][i]
    
    ind3_abx = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i]
    ind3_aby = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][i]

    ind4_abx = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i]
    ind4_aby = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][i]

    ind5_abx = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i]
    ind5_aby = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][i]

    ind6_abx = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i]
    ind6_aby = df1[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][i]

    if math.isnan(ind2_abx):
        continue
    else:  

        if ind2_abx == ind1_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                ant2_ant1_mix.append(i)
   
        if ind2_abx == ind3_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                ant2_ant3_mix.append(i)
    
        elif ind2_abx == ind4_abx:
            if math.isnan(ind2_abx):
                continue
            else: 
                ant2_ant4_mix.append(i)

        elif ind2_abx == ind5_abx:
            if math.isnan(ind2_abx):
                continue
            else:
                ant2_ant5_mix.append(i)

        elif ind2_abx == ind6_abx:
            if math.isnan(ind2_abx):
                continue
            else:   
                ant2_ant6_mix.append(i)
                

if len(ant2_ant1_mix) > 0:
    refined_ant2_ant1_mix.append(ant2_ant1_mix[0])

if len(ant2_ant3_mix) > 0:
    refined_ant2_ant3_mix.append(ant2_ant3_mix[0])
if len(ant2_ant4_mix) > 0:  
    refined_ant2_ant4_mix.append(ant2_ant4_mix[0])
if len(ant2_ant5_mix) > 0:  
    refined_ant2_ant5_mix.append(ant2_ant5_mix[0])
if len(ant2_ant6_mix) > 0:  
    refined_ant2_ant6_mix.append(ant2_ant6_mix[0])


for x,y in zip(ant2_ant1_mix[::],ant2_ant1_mix[1::]): 
     if abs(x-y) > 1:
        refined_ant2_ant1_mix.append(x)
         
for x,y in zip(ant2_ant3_mix[::],ant2_ant3_mix[1::]): 
     if abs(x-y) > 1:
        refined_ant2_ant3_mix.append(x)

for x,y in zip(ant2_ant4_mix[::],ant2_ant4_mix[1::]): 
     if abs(x-y) > 1:
        refined_ant2_ant4_mix.append(x)
         
for x,y in zip(ant2_ant5_mix[::],ant2_ant5_mix[1::]): 
     if abs(x-y) > 1:
        refined_ant2_ant5_mix.append(x)

for x,y in zip(ant2_ant6_mix[::],ant2_ant6_mix[1::]): 
     if abs(x-y) > 1:
        refined_ant2_ant6_mix.append(x)



if len(ant2_ant1_mix) > 0:
    refined_ant2_ant1_mix.append(ant2_ant1_mix[-1])

if len(ant2_ant3_mix) > 0:
    refined_ant2_ant3_mix.append(ant2_ant3_mix[-1])
if len(ant2_ant4_mix) > 0:    
    refined_ant2_ant4_mix.append(ant2_ant4_mix[-1])
if len(ant2_ant5_mix) > 0:  
    refined_ant2_ant5_mix.append(ant2_ant5_mix[-1])
if len(ant2_ant6_mix) > 0:  
    refined_ant2_ant6_mix.append(ant2_ant6_mix[-1])


if len(refined_ant2_ant1_mix) != 0:
    refined_ant2_ant1_mix = list(set(refined_ant2_ant1_mix))

if len(refined_ant2_ant3_mix) != 0:
    refined_ant2_ant3_mix = list(set(refined_ant2_ant3_mix))
if len(refined_ant2_ant4_mix) != 0:
    refined_ant2_ant4_mix = list(set(refined_ant2_ant4_mix))
if len(refined_ant2_ant5_mix) != 0:
    refined_ant2_ant5_mix = list(set(refined_ant2_ant5_mix))
if len(refined_ant2_ant6_mix) != 0:
    refined_ant2_ant6_mix = list(set(refined_ant2_ant6_mix))

 
 
 
 

previous_detection = "first"
full_ids = ['ind2_ind2']

for mix in refined_ant2_ant1_mix, refined_ant2_ant3_mix, refined_ant2_ant4_mix, refined_ant2_ant5_mix, refined_ant2_ant6_mix:

    if len(mix) != 0:

        for index, i in enumerate(mix):

    
            frame_id = i
            
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
        
        
            length = 14
            width = 16
            extra = 4
    
            ##make waffles for the other ants detected on the frame to compare

            ind1_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][frame_id-1]
            ind1_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][frame_id-1]
            ind1_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][frame_id-1]
            ind1_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][frame_id-1]
            
            ind2_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][frame_id]
            ind2_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][frame_id]
    
            ind2_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][frame_id]
            ind2_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][frame_id]
        
            ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][frame_id-1]
            ind3_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][frame_id-1]
        
            ind3_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][frame_id-1]
            ind3_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][frame_id-1]
    
            ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][frame_id-1]
            ind4_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][frame_id-1]
    
            ind4_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][frame_id-1]
            ind4_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][frame_id-1]
    
            ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][frame_id-1]
            ind5_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][frame_id-1]
    
            ind5_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][frame_id-1]
            ind5_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][frame_id-1]
    
            ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][frame_id-1]
            ind6_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][frame_id-1]
    
            ind6_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][frame_id-1]
            ind6_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][frame_id-1]


            ind2_theta = davi.get_vector_angle(ind2_abx, ind2_aby, ind2_thx, ind2_thy)
            ind3_theta = davi.get_vector_angle(ind3_abx, ind3_aby, ind3_thx, ind3_thy)
            ind4_theta = davi.get_vector_angle(ind4_abx, ind4_aby, ind4_thx, ind4_thy)
            ind5_theta = davi.get_vector_angle(ind5_abx, ind5_aby, ind5_thx, ind5_thy)
            ind6_theta = davi.get_vector_angle(ind6_abx, ind6_aby, ind6_thx, ind6_thy)

            ind1_theta = davi.get_vector_angle(ind1_abx, ind1_aby, ind1_thx, ind1_thy)
            
        
        
            ind2_waffle = []
            ind3_waffle = []
            ind4_waffle = []
            ind5_waffle = []
            ind6_waffle = []

            ind1_waffle = []


            if ind2_theta == 0:


                test = list(ant2.loc[i])
                ant2_copy.loc[i] = test


            else:     
    
                ind2_waffle = davi.create_waffle(ind2_abx, ind2_aby, ind2_theta, pix)
    
        
        
                if ind3_theta == 0:
        
                    for r in reversed(range(0, frame_id)):
        
                        ind3_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][r]
                        ind3_checker2 = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][r]
        
                        if math.isnan(ind3_checker) or math.isnan(ind3_checker2):
                            continue
                        else:
                            ind3_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][r]
                            ind3_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'y')][r]
        
                            ind3_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'x')][r]
                            ind3_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'thorax', 'y')][r]
        
                            break
        
                ind3_theta = davi.get_vector_angle(ind3_abx, ind3_aby, ind3_thx, ind3_thy)
                            
                    
                if ind4_theta == 0:
        
                    for r in reversed(range(0, frame_id)):
        
                        ind4_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][r]
                        ind4_checker2 = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][r]
        
                        if math.isnan(ind4_checker) or math.isnan(ind4_checker2):
                            continue
                        else:
                            ind4_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][r]
                            ind4_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'y')][r]
        
                            ind4_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'x')][r]
                            ind4_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'thorax', 'y')][r]
                
                            break
        
                ind4_theta = davi.get_vector_angle(ind4_abx, ind4_aby, ind4_thx, ind4_thy)
        
                
                if ind5_theta == 0:
        
                    for r in reversed(range(0, frame_id)):
        
                        ind5_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][r]
                        ind5_checker2 = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][r]
        
                        if math.isnan(ind5_checker) or math.isnan(ind5_checker2):
                            continue
                        else:
                            ind5_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][r]
                            ind5_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'y')][r]
        
                            ind5_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'x')][r]
                            ind5_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'thorax', 'y')][r]
        
                            break
        
                ind5_theta = davi.get_vector_angle(ind5_abx, ind5_aby, ind5_thx, ind5_thy)
                #print("ind5_theta", ind5_theta)
        
                
                if ind6_theta == 0:
        
                    for r in reversed(range(0, frame_id)):
        
                        ind6_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][r]
                        ind6_checker2 = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][r]
        
                        if math.isnan(ind6_checker) or math.isnan(ind6_checker2):
                            continue
                        else:
                            ind6_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][r]
                            ind6_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'y')][r]
        
                            ind6_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'x')][r]
                            ind6_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'thorax', 'y')][r]
        
                            break
                            
                ind6_theta = davi.get_vector_angle(ind6_abx, ind6_aby, ind6_thx, ind6_thy)



                if ind1_theta == 0:
        
                    for r in reversed(range(0, frame_id)):
        
                        ind1_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][r]
                        ind1_checker2 = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][r]
        
                        if math.isnan(ind1_checker) or math.isnan(ind1_checker2):
                            continue
                        else:
                            ind1_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][r]
                            ind1_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'y')][r]
        
                            ind1_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'x')][r]
                            ind1_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'thorax', 'y')][r]
        
                            break
                            
                ind1_theta = davi.get_vector_angle(ind1_abx, ind1_aby, ind1_thx, ind1_thy)
        

                
                ## failsafe
        
                if previous_detection == "ind2_changed_id":

                    ind2_abx_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][frame_id-1]
                    ind2_aby_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][frame_id-1]
            
                    ind2_thx_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][frame_id-1]
                    ind2_thy_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][frame_id-1]
    
                else:
                    ind2_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][frame_id-1]
                    ind2_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][frame_id-1]
                    ind2_thx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][frame_id-1]
                    ind2_thy_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][frame_id-1]
        
                ind2_theta_p = davi.get_vector_angle(ind2_abx_p, ind2_aby_p, ind2_thx_p, ind2_thy_p)
        
                ind2_waffle_p = []
        
                if ind2_theta_p == 0:
        
                    for r in reversed(range(0, frame_id)):
    
                        if previous_detection == "ind2_changed_id":
                            
                            ind1_checker = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][r]
                            ind1_checker2 = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][r]
                            
                        else:
                            
                            ind1_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][r]
                            ind1_checker2 = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][r]
        
                        if math.isnan(ind1_checker) or math.isnan(ind1_checker2):
                            
                            continue
                            
                        else:
                            if previous_detection == "ind2_changed_id":
                                ind2_abx_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][r]
                                ind2_aby_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][r]
            
                                ind2_thx_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][r]
                                ind2_thy_p = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][r]
                                break
                        
                            else:    
                                ind2_abx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][r]
                                ind2_aby_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'y')][r]
            
                                ind2_thx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'x')][r]
                                ind2_thy_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'thorax', 'y')][r]
        
                                break
    
                    if r != 0:
                        ind2_theta_p = davi.get_vector_angle(ind2_abx_p, ind2_aby_p, ind2_thx_p, ind2_thy_p)
                                    
                        ind2_waffle_p = davi.create_waffle(ind2_abx_p, ind2_aby_p, ind2_theta_p, pix)
                        ind2_ind2 = davi.subtract_colours_2(ind2_waffle, ind2_waffle_p)
                        
                        
                    else:
                        ind2_waffle_p = davi.create_waffle(ind2_abx, ind2_aby, ind2_theta, pix)
                        ind2_ind2 = davi.subtract_colours_2(ind2_waffle, ind2_waffle_p)
    
                else:
                    if ind2_aby_p > 1070:
                        ind2_ind2 = 30
                    else:
                        ind2_waffle_p = davi.create_waffle(ind2_abx_p, ind2_aby_p, ind2_theta_p, pix)
                        
                                
                        ind2_ind2 = davi.subtract_colours_2(ind2_waffle, ind2_waffle_p)
                
    
#                ind2_waffle = davi.create_waffle(ind2_abx, ind2_aby, ind2_theta, pix)
#                 ind2_ind2 = davi.subtract_colours_2(ind2_waffle, ind2_waffle_p)
        
                ## make comparisons
                            
                if ind6_theta ==0:
                    ind2_ind6 = 30
                else:
                    ind6_waffle = davi.create_waffle(ind6_abx, ind6_aby, ind6_theta, pix)

                    ind2_ind6 = davi.subtract_colours_2(ind2_waffle, ind6_waffle)

                if ind1_theta ==0:
                    ind1_ind2 = 30
                else:
                    ind1_waffle = davi.create_waffle(ind1_abx, ind1_aby, ind1_theta, pix)        
                    ind2_ind1 = davi.subtract_colours_2(ind2_waffle, ind1_waffle)

                    
                if ind3_theta ==0:
                    ind2_ind3 = 30
                else:
                    ind3_waffle = davi.create_waffle(ind3_abx, ind3_aby, ind3_theta, pix)

                    ind2_ind3 = davi.subtract_colours_2(ind2_waffle, ind3_waffle)
    
                    
                if ind4_theta==0:
                    ind2_ind4 = 30
                else:
                    ind4_waffle = davi.create_waffle(ind4_abx, ind4_aby, ind4_theta, pix)
    
                    ind2_ind4 = davi.subtract_colours_2(ind2_waffle, ind4_waffle)

        
                if ind5_theta==0 or math.isnan(ind5_theta):
                    ind2_ind5 = 30
                else:
                    if ind5_aby > 1070:
                        ind2_ind5 = 30
                    else:

                        ind5_waffle = davi.create_waffle(ind5_abx, ind5_aby, ind5_theta, pix)

                        ind2_ind5 = davi.subtract_colours_2(ind2_waffle, ind5_waffle)
        

        
                optimal_match = { "ind2_ind3" : ind2_ind3,
                                 "ind2_ind4" : ind2_ind4,
                                 "ind2_ind5" : ind2_ind5,
                                 "ind2_ind6" : ind2_ind6,
                                "ind2_ind2": ind2_ind2,
                                 "ind2_ind1": ind2_ind1
                                }
        
                
                true_id_val = min(optimal_match.values())
            
                nub_match = []

                for match in optimal_match:
                    if optimal_match[match] == true_id_val:
                        nub_match.append(match)

                if len(nub_match) > 1:
                    
                    identities = []
                    
                    for match, value in optimal_match.items():
                        if value == true_id_val:
                            identities.append(match)
                    
                    ind2_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]
                    
                    if previous_detection == "ind2_changed_id":
                        ind2_checker = ant2_copy[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i]

                    distances = {}    
                    
                    for iden in identities:
                        if iden == "ind2_ind3":
                            
                            ind3_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind3', 'abdomen', 'x')][i+1]
                            
                            ind2_ind3_distance = abs(ind2_checker - ind3_checker)
                            distances['ind2_ind3']=ind2_ind3_distance
                            
                        elif iden == "ind2_ind1":
                            
                            ind1_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind1', 'abdomen', 'x')][i+1]
                            
                            ind2_ind1_distance = abs(ind2_checker - ind1_checker)
                            distances['ind2_ind1']=ind2_ind1_distance

                            
                        elif iden == "ind2_ind4":
                            
                            
                            ind4_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind4', 'abdomen', 'x')][i+1]
                            
                            ind2_ind4_distance = abs(ind2_checker - ind4_checker)
                            distances['ind2_ind4']=ind2_ind4_distance

                        elif iden == "ind2_ind5":
                            
                            ind5_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind5', 'abdomen', 'x')][i+1]
                            
                            ind2_ind5_distance = abs(ind2_checker - ind5_checker)
                            distances['ind2_ind5']=ind2_ind5_distance

                    
                        elif iden == "ind2_ind6":
                            
                            ind6_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind6', 'abdomen', 'x')][i+1]
                            
                            ind2_ind6_distance = abs(ind2_checker - ind6_checker)
                            distances['ind2_ind6']=ind2_ind6_distance
                            
                        elif iden == "ind2_ind2":
                            
                            ind6_checker = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', 'ind2', 'abdomen', 'x')][i+1]
                            
                            ind2_ind6_distance = abs(ind2_checker - ind6_checker)
                            distances['ind2_ind2']=ind2_ind6_distance
                        
                    
                    true_id = min(distances, key=distances.get)
                
                    
                else:
                    true_id = min(optimal_match, key=optimal_match.get)

                    
                    
                full_ids.append(true_id)

                
                if true_id == "ind2_ind2":

    
                    if i != mix[-1]:
                        next_elem = mix[index + 1]
    
                        for f in range(i, next_elem):
                            new_row_ant1 = ant2.loc[f]
                            test = list(new_row_ant1)
                            ant2_copy.loc[f] = test
    
                    
                    elif i == mix[-1]:
                        continue
                    
    
    
                
                if true_id == "ind2_ind3":
    
                    if i != mix[-1]:
                        next_elem = mix[index + 1]
    
                        for f in range(i, next_elem):
                            new_row_ant1 = ant3.loc[f]
                            test = list(new_row_ant1)
                            ant2_copy.loc[f] = test
    
                    
                    elif i == mix[-1]:
                        continue
                
                    previous_detection = "ind2_changed_id"
                    
                elif true_id == "ind2_ind4":

    
                    if i != mix[-1]:
                        next_elem = mix[index + 1]
                        for f in range(i, next_elem):
                            new_row_ant1 = ant4.loc[f]
                            test = list(new_row_ant1)
                            ant2_copy.loc[f] = test
                            
                    elif i == mix[-1]:
                        continue
    
                    previous_detection = "ind2_changed_id"
                    
        
                elif true_id == "ind2_ind5":

                    if i != mix[-1]:
                        next_elem = mix[index + 1]
    
                        for f in range(i, next_elem):
                            new_row_ant1 = ant5.loc[f]
                            test = list(new_row_ant1)
                            ant2_copy.loc[f] = test
    
                    
                    elif i == mix[-1]:
                        continue

                    previous_detection = "ind2_changed_id"    
        
                elif true_id == "ind2_ind6":
    
    
                    if i != mix[-1]:
                        next_elem = mix[index + 1]
    
                        for f in range(i, next_elem):
                            new_row_ant1 = ant6.loc[f]
                            test = list(new_row_ant1)
                            ant2_copy.loc[f] = test
                    
                    elif i == mix[-1]:
                        continue

                    previous_detection = "ind2_changed_id"    


                elif true_id == "ind2_ind1":

                    if i != mix[-1]:
                        next_elem = mix[index + 1]
    
                        for f in range(i, next_elem):
                            new_row_ant1 = ant1.loc[f]
                            test = list(new_row_ant1)
                            ant2_copy.loc[f] = test
                    
                    elif i == mix[-1]:
                        continue
                

        
                    previous_detection = "ind2_changed_id"    
     
 
 
 
 
     
print("colour checks are complete")


#ant2_copy.to_csv("ant2_detections.csv")
ant2_copy.to_hdf(os.path.join(davi_output_path, "ant2_detections.h5"), key="changed_names", format="fixed")
 
 
globals().clear()
 
 
