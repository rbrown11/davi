##################################
######## MODULES REQUIRED ########
##################################

import pandas as pd
import math
import davi
import random
import numpy as np
import hulls
from collections import Counter
import PySimpleGUI as sg
from PIL import Image, ImageTk, ImageEnhance
import os
import io
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt 
import flags_manual_check
from tqdm import tqdm
import itertools


##################################
########### FUNCTIONS ############
##################################

pd.options.mode.chained_assignment = None
    
def calculate_new_coords(x_coord, y_coord, angle, distance):
    new_coord_x = x_coord + distance * math.cos(angle)
    new_coord_y = y_coord + distance * math.sin(angle)
    return new_coord_x, new_coord_y
    
def find_abdomen(x_coord, y_coord, theta ):
    
    length=12
    width=8
    extra=2
    
    coord_list = []
    
    AN0 = calculate_new_coords(x_coord, y_coord, theta+(math.pi/2), length/2) 
    AA0 = calculate_new_coords(AN0[0], AN0[1], theta, width/2) #top left
    AT0 = calculate_new_coords(AN0[0], AN0[1], theta+math.pi, width/2)
    AX0 = calculate_new_coords(AT0[0], AT0[1], theta+math.pi, extra) #bottom left
    
    AN4 = calculate_new_coords(x_coord, y_coord, theta-(math.pi/2), length/2)
    AA4 = calculate_new_coords(AN4[0], AN4[1], theta, width/2) #top right
    AT4 = calculate_new_coords(AN4[0], AN4[1], theta+math.pi, width/2)
    AX4 = calculate_new_coords(AT4[0], AT4[1], theta+math.pi, extra) #bottom right
    
    coord_list.append(AA0)
    coord_list.append(AX0)
    coord_list.append(AA4)
    coord_list.append(AX4)
    
    return coord_list ### top left, bottom left, top right, bottom right
    

                
def set_row_to_nan(df, row_index):

    if row_index in df.index:
        df.loc[row_index] = np.nan

def assign_detections_from_key(dataframe_key, data, key, i):
    # First, assign data for individuals with known colors
    for k in key:  # Loop over each individual in the key
        if key[k] != 'unknown':  # If the individual has a known color
            color = key[k]  # Color (like 'blue', 'red', etc.)
            if color in dataframe_key:
                try:
                    # Select the specific row for this individual and assign it to the corresponding DataFrame
                    row_to_copy = data.xs(k, level='individuals', axis=1, drop_level=False).loc[i]
                    dataframe_key[color].loc[i] = row_to_copy.values  # Assign the row to the corresponding color DataFrame
                except KeyError as e:
                    print(f"KeyError for {k} at frame {i}: {e}")
                    
    # After assigning, check the length of each DataFrame and append NaN rows where necessary
    for d in dataframe_key:
        if len(dataframe_key[d]) <= i:  # If the DataFrame is shorter than expected 
            # Append a row of NaN to match the correct length
            row_to_copy = data.xs('ind1', level='individuals', axis=1, drop_level=False).loc[i]
            dataframe_key[d].loc[i] = row_to_copy.values
                
            set_row_to_nan(dataframe_key[d], i)


                
def find_colour_match(frame_id, ant, video_path, data, hull_dict, key):
    """Find the color match for the ant without saving images to disk."""
    
    # Remove certain colors from comparison
#    keys_to_remove = [key[k] for k in key if key[k] in hull_dict]
#    for key_to_remove in keys_to_remove:
#        hull_dict.pop(key_to_remove, None)

    # Load the specific frame from the video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    
    if not ret:
        return {}

    # Convert the frame to RGB format
    colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(colour_converted)

    # Enhance the image (brightness and contrast)
    enhancer = ImageEnhance.Brightness(im)
    im_output = enhancer.enhance(1.5)
    contrast = ImageEnhance.Contrast(im_output)
    im_output_T = contrast.enhance(1.2)
    
    # Crop the image around the ant's location
    ant1_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][frame_id]
    ant1_aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'y')][frame_id]
    ant1_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][frame_id]
    ant1_thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][frame_id]
    
    ant1_theta = davi.get_vector_angle(ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    
    # Check for invalid theta
    if ant1_theta == 0 or math.isnan(ant1_theta) or ant1_aby > 1065:
        return {}
    
    # Calculate the crop region based on abdomen and thorax position
    ant1_waffle = find_abdomen(ant1_abx, ant1_aby, ant1_theta)
    top_left_x = int(min([ant1_waffle[0][0], ant1_waffle[1][0], ant1_waffle[2][0], ant1_waffle[3][0]]))
    top_left_y = int(min([ant1_waffle[0][1], ant1_waffle[1][1], ant1_waffle[2][1], ant1_waffle[3][1]]))
    bot_right_x = int(max([ant1_waffle[0][0], ant1_waffle[1][0], ant1_waffle[2][0], ant1_waffle[3][0]]))
    bot_right_y = int(max([ant1_waffle[0][1], ant1_waffle[1][1], ant1_waffle[2][1], ant1_waffle[3][1]]))
    
    # Crop the image in memory
    ant1_crop = im_output_T.crop((top_left_x, top_left_y, bot_right_x, bot_right_y))
    
    # Convert the cropped image to a format for processing
    ant1_crop_np = np.array(ant1_crop)

    # Process the cropped image to find colors
    modified_image = ant1_crop_np.reshape(ant1_crop_np.shape[0]*ant1_crop_np.shape[1], 3)
    clf = KMeans(n_clusters=10, n_init=10)
    Klabels = clf.fit_predict(modified_image)
    counts = Counter(Klabels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[k] for k in counts.keys()]
    
    if len(ordered_colors) < 10:
        return {'blue': 0, 'red': 0, 'yellow': 0, 'green': 0, 'white': 0, 'pink': 0}
        cap.release()
        del ant1_crop
        del im_output_T
        del im_output
        del im

    else:
    
        hex_colors = [hulls.RGB2HEX(ordered_colors[k]) for k in counts.keys()]
    
    # Match colors with hulls
        hull_matches = []
        for hexcode in hex_colors:
            rgb = list(hulls.hex_to_rgb(hexcode))
            for color, hull in hull_dict.items():
                if hulls.is_rgb_in_hull(rgb, hull):
                    hull_matches.append(color)
    
    # Count the occurrences of matched colors
  #  sorted_counts = dict(sorted(Counter(hull_matches).items()))
    
    
        blue_cnt = hull_matches.count('blue')
        red_cnt = hull_matches.count('red')                                   
        yellow_cnt = hull_matches.count('yellow')                                  
        green_cnt = hull_matches.count('green')  
        white_cnt = hull_matches.count('white') 
        pink_cnt = hull_matches.count('pink')
    
        col_cnts = { 'blue' : int(blue_cnt), 'red' : int(red_cnt), 'yellow' : int(yellow_cnt), 'green' : int(green_cnt), 'white' : int(white_cnt), 'pink': int(pink_cnt)}    
    
    # Clean up by closing the video capture object and deleting images
        cap.release()
        del ant1_crop
        del im_output_T
        del im_output
        del im

        print(col_cnts)
        return col_cnts
    




def find_duplicate_colours(key):

    colour_counts = Counter(key.values())
    
    # Find colours that appear more than once
    duplicates = [colour for colour, count in colour_counts.items() if count > 1]
    
    return duplicates               



##################################
########## MAKING HULLS ##########
##################################



hex_data = pd.read_csv('/home/rb17990/Documents/colour_detection/Kmeans_col_sep_first_round.csv')

hex_data = hex_data.drop(columns = ['Unnamed: 0'], axis=1)
hex_data.drop(hex_data[hex_data.ant_colours == 'BAD'].index, inplace=True)
hex_data = hex_data.reset_index()
hex_data = hex_data.drop(columns = ['index'], axis=1)


blue = []
red  = []
yellow = []
green = []
white = []
pink = []


for i in range(0, len(hex_data)):
    
    colour = hex_data['ant_colours'][i]
	    
    if colour == 'blue':
        blue.append(hex_data['best_hex'][i])
        
    elif colour == 'red':
        red.append(hex_data['best_hex'][i])
    
    elif colour == 'yellow ':
        yellow.append(hex_data['best_hex'][i])
    
    elif colour == 'green':
        green.append(hex_data['best_hex'][i])
    
    elif colour == 'white':
        white.append(hex_data['best_hex'][i])
    
    elif colour == 'pink':
        pink.append(hex_data['best_hex'][i])
    
       
white_rgb_values = np.array([hulls.hex_to_rgb(code_w) for code_w in white])
blue_rgb_values = np.array([hulls.hex_to_rgb(code_b) for code_b in blue])
red_rgb_values = np.array([hulls.hex_to_rgb(code_r) for code_r in red])
yellow_rgb_values = np.array([hulls.hex_to_rgb(code_y) for code_y in yellow])
green_rgb_values = np.array([hulls.hex_to_rgb(code_g) for code_g in green])
pink_rgb_values = np.array([hulls.hex_to_rgb(code_p) for code_p in pink])



white_hull = ConvexHull(white_rgb_values)
blue_hull = ConvexHull(blue_rgb_values)
red_hull = ConvexHull(red_rgb_values)
yellow_hull = ConvexHull(yellow_rgb_values)
green_hull = ConvexHull(green_rgb_values)
pink_hull = ConvexHull(pink_rgb_values)

hull_dict = {'blue' : blue_hull,
            'red' : red_hull,
            'yellow' : yellow_hull,
            'green' : green_hull,
            'white' : white_hull,
            'pink' : pink_hull}



##################################
######### PATHS REQUIRED #########
##################################


path = '/media/rb17990/DATA/EXP1/6ants analysis/dlc_h5_to_be_fixed/original_R23-5P-2DLC_dlcrnetms5_full-modelJan10shuffle1_100000.h5'
data = pd.read_hdf(path)

video_path = '/media/rb17990/DATA/EXP1/6ants/R23-5P-2.mp4'

folder_path = '/media/rb17990/DATA/EXP1/6ants analysis/DAVI_OUTPUT'

file_name = os.path.basename(path)


##################################
####### REMOVE DUPLICATES ########
##################################

ants1 = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']

comp_dist = {}

for combo in itertools.combinations(ants1, 2):
    
    ant1 = combo[0]
    ant2 = combo[1]

    
    ant2_subset = data.xs(ant2, level='individuals', axis=1, drop_level=False)        
        
    for i in range(0, len(data)):
        ant1_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant1, 'thorax', 'x')][i]
        ant2_thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant2, 'thorax', 'x')][i]
            
            
        if math.isnan(ant1_thx):
            continue
        else: 
            if ant1_thx == ant2_thx:
                
                
                ant2_subset.loc[i] = np.nan
                
                
                               
    data.loc[:, pd.IndexSlice[:, ant2, :]] = ant2_subset
           
                

data.to_hdf(os.path.join(folder_path, 'duplicates_removed.h5'), format='fixed', key='changed_names')

path_to_dupl_file = os.path.join(folder_path, 'duplicates_removed.h5')

print('data has been cleaned, start extracting colours from this h5 file: ', path_to_dupl_file)


##################################
######### FLAGGING FRAMES ########
##################################



def check_frame(i, data):

    ants = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']

    flags = []

    for ant in ants:
#        for i in range(0, len(data)):
    
        thx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][i]
        thy = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][i]
        
        
        if math.isnan(thx):
            if i not in flags: # remove duplicates
                flags.append(i)

        else:


            if i != 0:
        
                thx_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][i-1]
                thy_p = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][i-1]
                if math.isnan(thx_p):
                    if i not in flags:
                        flags.append(i)
                        
                else: 
                        
                    distance = abs(davi.find_distance(thx, thx_p, thy, thy_p))

            
                    if distance > 15: 
                        if i not in flags:
                        
                            flags.append(i)
             

    return len(flags)





##################################
########## FIRST COLOURS #########
##################################

print(path)
print(video_path)


ind1_id = input(" what colour is ind1? \n")
ind2_id = input(" what colour is ind2? \n")
ind3_id = input(" what colour is ind3? \n")
ind4_id = input(" what colour is ind4? \n")
ind5_id = input(" what colour is ind5? \n")
ind6_id = input(" what colour is ind6? \n")


frame_of_flip = int(input("what row has the first detections of all 6 ants? (i.e. i?) \n"))


fof_to_end = data[data.index >= int(frame_of_flip)].reset_index(drop=True)
print(fof_to_end.loc[0])
## fof_to_end is now a dataframe that starts with the frame_of_flip until the end of the dataset

new_data = pd.DataFrame().reindex(columns=(fof_to_end).columns)

for col in new_data.columns:
    col = list(col)
    
    if col[1] == 'ind1':
        new_name = col[1].replace('ind1', ind1_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)
        
    if col[1] == 'ind2':
        new_name = col[1].replace('ind2', ind2_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)        
        
    if col[1] == 'ind3':
        new_name = col[1].replace('ind3', ind3_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)
            
    if col[1] == 'ind4':
        new_name = col[1].replace('ind4', ind4_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)
        
    if col[1] == 'ind5':
        new_name = col[1].replace('ind5', ind5_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)        
        
    if col[1] == 'ind6':
        new_name = col[1].replace('ind6', ind6_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)        
        

blue = new_data.xs('blue', level='individuals', axis=1, drop_level=False)
red = new_data.xs('red', level='individuals', axis=1, drop_level=False)
yellow = new_data.xs('yellow', level='individuals', axis=1, drop_level=False)
green = new_data.xs('green', level='individuals', axis=1, drop_level=False)
white = new_data.xs('white', level='individuals', axis=1, drop_level=False)
pink = new_data.xs('pink', level='individuals', axis=1, drop_level=False)

## these are now empty dataframes that will be populated with data between frame_of_flip and end

##################################
######### INITIALISE KEYS ########
##################################



dataframe_key = { 'blue' : blue, 'red' : red, 'yellow' : yellow, 'green' : green, 'white' : white, 'pink' : pink }


key = { 'ind1' : ind1_id, 'ind2' : ind2_id, 'ind3' : ind3_id, 'ind4' : ind4_id, 'ind5' : ind5_id, 'ind6' : ind6_id }

colours = ['blue', 'red', 'yellow', 'green', 'white', 'pink']



##################################
########### FOF TO END ###########
##################################


#data = fof_to_end



assign_detections_from_key(dataframe_key, fof_to_end, key, 0)     
print(len(blue))



for i in tqdm(range(1, len(fof_to_end))):


    flag = check_frame(i, fof_to_end)
   # print(flag)
    
   # if i not in flags: 
    if flag == 0:
       assign_detections_from_key(dataframe_key, fof_to_end, key, i)

    else:
           
        uncertain_ids = []
                
        key = {} # reset the key
        
        
        # distance tracking
        
        ants_to_check = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']
        
        for ant in ants_to_check:
            #print(ant)
            
        
            distances = {}
        
            thx = fof_to_end[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][i]
            thy = fof_to_end[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][i]
           
            #print(thx, thy)
            
            if math.isnan(thx):

                key.update( { ant : 'unknown' } )

                
            else:
                      
                for d in dataframe_key:
                   
                    test_dataframe = dataframe_key[d]
                
                    colour = d
            
                    second_thx = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'x')][i-1]
                    second_thy = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'y')][i-1]
                
                    if math.isnan(second_thx): #find last seen detections

                        for r in reversed(range( 0 , i)): 
               #             print(r)

                            second_thx = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'x')][r]
                            
                            if math.isnan(second_thx):
                                continue          
                                
                            else:
                                second_thx = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'x')][r]
                                second_thy = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'y')][r]                            
                                break
                                

                        if r == 0 and math.isnan(second_thx):
                            #uncertain_ids.append(ant) # we cannot distance track but we can colour track
                            current_dist = 1000
                            
                        else:    
                            current_dist = abs(davi.find_distance(thx, second_thx, thy, second_thy))                  

                    else:
                    
                        current_dist = abs(davi.find_distance(thx, second_thx, thy, second_thy))
                
                    current_comparison = { colour : current_dist } 
                
                    distances.update( current_comparison )


                closest_match = min(distances, key=distances.get)
                closest_match_value = min(distances.values())
       #         print(distances, closest_match)
            
                if closest_match_value > 25:
            
                    uncertain_ids.append(ant)
                
                
                elif closest_match_value < 25 or closest_match_value == 25:
            
                    key.update( { ant : closest_match } )
        
        #print(key)
              #  print(uncertain_ids)
                    
                   
        if len(uncertain_ids) == 1 and 'unknown' not in key.values(): # if all ants are detected but one is uncertain
           # print('uncertain ids == 1 and no unknown')
            for col in colours:
                if col not in key.values():

                    key.update( { uncertain_ids[0] : col } )
                 #   print(uncertain_ids[0], col)
                    break
                   
            
        elif len(uncertain_ids) > 1 or (len(uncertain_ids) ==1 and 'unknown' in key.values()): # if more than one ant uncertain
                
            ## colour tracking ##    
                
            second_uncertain_ids = []
      #      print('COLOUR TRACKING')
            for iden in uncertain_ids:
            
             #   print('iden: ' , iden)
                    
                thx = fof_to_end[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', iden, 'thorax', 'x')][i]   
                thy = fof_to_end[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', iden, 'thorax', 'y')][i]                         
                                              
                if math.isnan(thx): #this should not happen
                
                    second_uncertain_ids.append(iden)
                            
                else: #this all needs checking
                        
                    col_counts = find_colour_match(i, iden, video_path, fof_to_end, hull_dict, key)
               #     print('col counts: ' , col_counts)
               #     print(len(col_counts))
                    
                    if len(col_counts) == 0:
                        second_uncertain_ids.append(iden)
                        
                    else:
                    
                        closest_colour = max(col_counts, key=col_counts.get)
                        highest_count = max(col_counts.values())
                            
                        multiple_col_matches = []
                        
                        if highest_count == 0:
                            second_uncertain_ids.append(iden)
                        
                            
                        for colour in col_counts:
                            if col_counts[colour] > 0:
                                multiple_col_matches.append(colour)
                                    
                        if len(multiple_col_matches) == 1: #if there is only one colour match
                            
                            if closest_colour not in key: 
                            
                                key.update({ iden : closest_colour })
                                    
                            else:
                                    
                                second_uncertain_ids.append(iden)
                                     
                        elif len(multiple_col_matches) > 1:
                    
                            max_counts = []
                        
                            for cnt in col_counts:
                                if col_counts[cnt] == highest_count:
                                    max_counts.append(cnt)
                                 
                            if len(max_counts) == 1:
                        
                            # we have a clear winner
                                for col in colours:
                                    if col not in key:
                                        key.update( { iden : closest_colour } )
                                               
                            
                            elif len(max_counts) > 1 or highest_count == 0:
                        
                            # no clear winner or no paint spot
                                second_uncertain_ids.append(iden)
                            
            if len(second_uncertain_ids) == 1 and 'unknown' not in key.values():
            
                for col in colours:
                    if col not in key:
                        key.update( { second_uncertain_ids[0] : col } )         
            
   
   
   
            elif len(second_uncertain_ids) > 1 or (len(second_uncertain_ids) ==1 and 'unknown' in key.values()):
            
        #        print(second_uncertain_ids)
        #        print(key)
                flags_manual_check.run_manual_check((frame_of_flip + i), data, video_path, second_uncertain_ids, key, folder_path) ## outline folder path
        



        if len(key) == 6:
            
            duplicates = find_duplicate_colours(key)
            if 'unknown' in duplicates:
                duplicates.remove('unknown')
                
            #print(duplicates)   
            if len(duplicates) > 0:
            
#                dupl_ids = []
 #               for d in duplicates:
#                    for k in key:
#                        if d == key[k]:
#                            dupl_ids.append(k)
                            
           #     print('duplicates:', dupl_ids)
            
                dupl_ids = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']            
                flags_manual_check.run_manual_check((frame_of_flip +i), data, video_path, dupl_ids, key, folder_path) 
            
      

             #   print('key from manual', key)

            unknown_count = list(key.values()).count('unknown')
            if unknown_count == 1:
              #  print('1 unknown found')
                unknown_cols = ['blue', 'red', 'yellow', 'green', 'white', 'pink']

                for u in unknown_cols:
                    if u not in key.values(): #check
                        col_to_update = u
                        #print('col_to_update', u)
                        
                        break

                for k in key:
                    if key[k] == 'unknown':
                        key.update( { k : col_to_update } )
                        break
                        
        #    elif unknown_count > 1:
        #        print('lotso unknowns kiddo')
            
            
            
            
            #print('final key:', key)
            assign_detections_from_key(dataframe_key, fof_to_end, key, i)
         #   print(len(blue), len(red), len(yellow), len(green), len(white), len(pink))



final_fof_to_end = pd.concat([blue, red, yellow, green, white, pink], join='inner', axis=1)
print(final_fof_to_end)

final_fof_to_end.to_hdf(os.path.join(folder_path, 'davi_fof_to_end_' + file_name), key='changed_names', format='fixed')





##################################
########## FOF TO START ##########
##################################

print('changing direction...')

fof_to_start = data[data.index <= int(frame_of_flip)]

fof_to_start = fof_to_start.iloc[::-1].reset_index(drop=True)

print(fof_to_start.loc[0])
## lets flip fof_to_start so that the loop runs with the same logic, then flip again before final concat

new_data = pd.DataFrame().reindex(columns=(fof_to_start).columns)

for col in new_data.columns:
    col = list(col)
    
    if col[1] == 'ind1':
        new_name = col[1].replace('ind1', ind1_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)
        
    if col[1] == 'ind2':
        new_name = col[1].replace('ind2', ind2_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)        
        
    if col[1] == 'ind3':
        new_name = col[1].replace('ind3', ind3_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)
            
    if col[1] == 'ind4':
        new_name = col[1].replace('ind4', ind4_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)
        
    if col[1] == 'ind5':
        new_name = col[1].replace('ind5', ind5_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)        
        
    if col[1] == 'ind6':
        new_name = col[1].replace('ind6', ind6_id)
        
        new_data.rename(columns = {col[0]:col[0], col[1]:new_name, col[2]:col[2], col[3]:col[3]}, inplace=True)        
        

blue = new_data.xs('blue', level='individuals', axis=1, drop_level=False)
red = new_data.xs('red', level='individuals', axis=1, drop_level=False)
yellow = new_data.xs('yellow', level='individuals', axis=1, drop_level=False)
green = new_data.xs('green', level='individuals', axis=1, drop_level=False)
white = new_data.xs('white', level='individuals', axis=1, drop_level=False)
pink = new_data.xs('pink', level='individuals', axis=1, drop_level=False)





dataframe_key = { 'blue' : blue, 'red' : red, 'yellow' : yellow, 'green' : green, 'white' : white, 'pink' : pink }


key = { 'ind1' : ind1_id, 'ind2' : ind2_id, 'ind3' : ind3_id, 'ind4' : ind4_id, 'ind5' : ind5_id, 'ind6' : ind6_id }

colours = ['blue', 'red', 'yellow', 'green', 'white', 'pink']


#data = fof_to_start

assign_detections_from_key(dataframe_key, fof_to_start, key, 0)     


 

for i in tqdm(range(1, len(fof_to_start))):


    flag = check_frame(i, fof_to_start)
   # print(flag)
    
   # if i not in flags: 
    if flag == 0:
       assign_detections_from_key(dataframe_key, fof_to_start, key, i)

    else:
           
        uncertain_ids = []
                
        key = {} # reset the key
        
        
        # distance tracking
        
        ants_to_check = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']
        
        for ant in ants_to_check:
            #print(ant)
            
        
            distances = {}
        
            thx = fof_to_start[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][i]
            thy = fof_to_start[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][i]
           
            #print(thx, thy)
            
            if math.isnan(thx):

                key.update( { ant : 'unknown' } )

                
            else:
                      
                for d in dataframe_key:
                   
                    test_dataframe = dataframe_key[d]
                
                    colour = d
            
                    second_thx = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'x')][i-1]
                    second_thy = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'y')][i-1]
                
                    if math.isnan(second_thx): #find last seen detections

                        for r in reversed(range(0, i)): ## i think the problem is here
                #            print(r)

                            second_thx = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'x')][r]
                            
                            if math.isnan(second_thx):
                                continue          
                                
                            else:
                                second_thx = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'x')][r]
                                second_thy = test_dataframe[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', d, 'thorax', 'y')][r]                            
                                break
                                

                        if r == 0 and math.isnan(second_thx):
                            #uncertain_ids.append(ant) # we cannot distance track but we can colour track
                            current_dist = 1000
                            
                        else:    
                            current_dist = abs(davi.find_distance(thx, second_thx, thy, second_thy))                  

                    else:
                    
                        current_dist = abs(davi.find_distance(thx, second_thx, thy, second_thy))
                
                    current_comparison = { colour : current_dist } 
                
                    distances.update( current_comparison )


                closest_match = min(distances, key=distances.get)
                closest_match_value = min(distances.values())
       #         print(distances, closest_match)
            
                if closest_match_value > 25:
            
                    uncertain_ids.append(ant)
                
                
                elif closest_match_value < 25 or closest_match_value == 25:
            
                    key.update( { ant : closest_match } )
        
        #print(key)
              #  print(uncertain_ids)
                    
                   
        if len(uncertain_ids) == 1 and 'unknown' not in key.values(): # if all ants are detected but one is uncertain
           # print('uncertain ids == 1 and no unknown')
            for col in colours:
                if col not in key.values():

                    key.update( { uncertain_ids[0] : col } )
                 #   print(uncertain_ids[0], col)
                    break
                   
            
        elif len(uncertain_ids) > 1 or (len(uncertain_ids) ==1 and 'unknown' in key.values()): # if more than one ant uncertain
                
            ## colour tracking ##    
                
            second_uncertain_ids = []
     #       print('COLOUR TRACKING')
            for iden in uncertain_ids:
            
             #   print('iden: ' , iden)
                    
                thx = fof_to_start[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', iden, 'thorax', 'x')][i]   
                thy = fof_to_start[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', iden, 'thorax', 'y')][i]                         
                                              
                if math.isnan(thx): #this should not happen
                
                    second_uncertain_ids.append(iden)
                            
                else: #this all needs checking
                        
                    col_counts = find_colour_match(i, iden, video_path, fof_to_start, hull_dict, key)
               #     print('col counts: ' , col_counts)
               #     print(len(col_counts))
                    
                    if len(col_counts) == 0:
                        second_uncertain_ids.append(iden)
                        
                    else:
                    
                        closest_colour = max(col_counts, key=col_counts.get)
                        highest_count = max(col_counts.values())
                            
                        multiple_col_matches = []
                        
                        if highest_count == 0:
                            second_uncertain_ids.append(iden)
                        
                            
                        for colour in col_counts:
                            if col_counts[colour] > 0:
                                multiple_col_matches.append(colour)
                                    
                        if len(multiple_col_matches) == 1: #if there is only one colour match
                            
                            if closest_colour not in key: 
                            
                                key.update({ iden : closest_colour })
                                    
                            else:
                                    
                                second_uncertain_ids.append(iden)
                                     
                        elif len(multiple_col_matches) > 1:
                    
                            max_counts = []
                        
                            for cnt in col_counts:
                                if col_counts[cnt] == highest_count:
                                    max_counts.append(cnt)
                                 
                            if len(max_counts) == 1:
                        
                            # we have a clear winner
                                for col in colours:
                                    if col not in key:
                                        key.update( { iden : closest_colour } )
                                               
                            
                            elif len(max_counts) > 1 or highest_count == 0:
                        
                            # no clear winner or no paint spot
                                second_uncertain_ids.append(iden)
                            
            if len(second_uncertain_ids) == 1 and 'unknown' not in key.values():
            
                for col in colours:
                    if col not in key:
                        key.update( { second_uncertain_ids[0] : col } )         
            
   
   
   
            elif len(second_uncertain_ids) > 1 or (len(second_uncertain_ids) ==1 and 'unknown' in key.values()):
            
              #  print(second_uncertain_ids)
             #   print(key)
                flags_manual_check.run_manual_check((frame_of_flip-i), data, video_path, second_uncertain_ids, key, folder_path) ## outline folder path
        



        if len(key) == 6:
            
            duplicates = find_duplicate_colours(key)
            if 'unknown' in duplicates:
                duplicates.remove('unknown')
                
            #print(duplicates)   
            if len(duplicates) > 0:
            
#                dupl_ids = []
 #               for d in duplicates:
#                    for k in key:
#                        if d == key[k]:
#                            dupl_ids.append(k)
                            
           #     print('duplicates:', dupl_ids)
            
                dupl_ids = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']            
                flags_manual_check.run_manual_check((frame_of_flip-i), data, video_path, dupl_ids, key, folder_path) 
            
      

             #   print('key from manual', key)

            unknown_count = list(key.values()).count('unknown')
            if unknown_count == 1:
              #  print('1 unknown found')
                unknown_cols = ['blue', 'red', 'yellow', 'green', 'white', 'pink']

                for u in unknown_cols:
                    if u not in key.values(): #check
                        col_to_update = u
                        #print('col_to_update', u)
                        
                        break

                for k in key:
                    if key[k] == 'unknown':
                        key.update( { k : col_to_update } )
                        break
                        
        #    elif unknown_count > 1:
        #        print('lotso unknowns kiddo')
            
            
            
            
            #print('final key:', key)
            assign_detections_from_key(dataframe_key, fof_to_start, key, i)
          #  print(len(blue), len(red), len(yellow), len(green), len(white), len(pink))





final_fof_to_start = pd.concat([blue, red, yellow, green, white, pink], join='inner', axis=1)
print(final_fof_to_start)

final_fof_to_start = final_fof_to_start.iloc[::-1].reset_index(drop=True)

final_fof_to_start.to_hdf(os.path.join(folder_path, 'davi_fof_to_start_' + file_name), key='changed_names', format='fixed')
final_dataset = pd.concat([final_fof_to_start.iloc[1:], final_fof_to_end]).reset_index(drop=True)



final_dataset.to_csv(os.path.join(folder_path, 'davi_check_flip.csv'))
final_dataset.to_hdf(os.path.join(folder_path, 'davi_' + file_name), key='changed_names', format='fixed')
