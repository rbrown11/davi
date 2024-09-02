import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import cv2
import davi
import os
import random
from PIL import Image, ImageEnhance
import math
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    
    
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    

def hex_to_rgb(hex_code):

    hex_code = hex_code.lstrip('#')

    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    
    return rgb


    
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
    
    
def is_rgb_in_hull(rgb, hull):
    point = np.array(rgb)
    delaunay = Delaunay(hull.points)  # Create Delaunay triangulation from hull points
    return delaunay.find_simplex(point) >= 0

# Function to find the closest convex hull to a point
def closest_convex_hull_name(point, hulls):
    min_distance = float('inf')
    closest_name = None
    
    for hull in hulls:
        if is_point_in_hull(point, hull):
            return hull.name
        else:
            # Calculate distance to hull's centroid
            centroid = np.mean(hull.points[hull.vertices], axis=0)
            distance = np.linalg.norm(point - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_name = hull.name
    
    return closest_name

def is_point_in_hull(point, hull):
    for simplex in hull.simplices:
        normal = np.cross(hull.points[simplex[1]] - hull.points[simplex[0]], hull.points[simplex[2]] - hull.points[simplex[0]])
        d = -np.sum(normal * hull.points[simplex[0]])
        if np.dot(point, normal) + d > 0:
            return False
    return True


def closest_convex_hull_index(point, hulls):
    min_distance = float('inf')
    closest_index = -1
    
    for i, hull in enumerate(hulls):
        if hull.find_simplex(point) >= 0:
            return i
    
    return closest_index



def find_closest_hull(rgb_value, hull_dict):
    min_dist = float('inf')
    closest_ant = None
    for ant, hull in hull_dict.items():
        dist = np.min(distance.cdist([rgb_value], hull.points[hull.vertices], 'euclidean'))
        if dist < min_dist:
            min_dist = dist
            closest_ant = ant
    return closest_ant


def find_hex_colours(frame_id, ant, video_path, data, hull_dict):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    folder_path = os.path.dirname(video_path)
    form = '.png'
    
    vidname = os.path.basename(video_path)
    
    full_vidname = vidname.strip('.mp4') + '-' + str(frame_id) + form
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
    
    im_output_T.save(os.path.join(folder_path, full_vidname))
    
    image_path = os.path.join(folder_path, full_vidname)
    
    image1 = get_image(image_path) #hulls function
    
    ant1_waffle = []
    hex_colors =[]
    
    ant1_abx = data[(
                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][frame_id]
    ant1_aby = data[(
                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'y')][frame_id]
    ant1_thx = data[(
                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][frame_id]
    ant1_thy = data[(
                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][frame_id]
                         

    ant1_theta = davi.get_vector_angle(ant1_abx, ant1_aby, ant1_thx, ant1_thy)
    ant1_waffle = find_abdomen(ant1_abx, ant1_aby, ant1_theta)


    if ant1_theta != 0 or not math.isnan(ant1_theta) or ant1_aby < 1065:
        print(ant1_theta)
    
        top_left_x = int(min([ant1_waffle[0][0], ant1_waffle[1][0], ant1_waffle[2][0], ant1_waffle[3][0]]))
        top_left_y = int(min([ant1_waffle[0][1], ant1_waffle[1][1], ant1_waffle[2][1], ant1_waffle[3][1]]))
                    
        bot_right_x = int(max([ant1_waffle[0][0], ant1_waffle[1][0], ant1_waffle[2][0], ant1_waffle[3][0]]))
        bot_right_y = int(max([ant1_waffle[0][1], ant1_waffle[1][1], ant1_waffle[2][1], ant1_waffle[3][1]]))
    
    
        ant1_crop = image1[top_left_y : bot_right_y, top_left_x : bot_right_x ] 
    
        ant1_crop = cv2.cvtColor(ant1_crop, cv2.COLOR_BGR2RGB)
        
        image_name = ant + '-' + full_vidname
    
        cv2.imwrite(os.path.join(folder_path, image_name), ant1_crop)
    
        hex_colors = []
        ant1_crop = get_image(os.path.join(folder_path, image_name)) ## hulls function
                            
        modified_image = ant1_crop.reshape(ant1_crop.shape[0]*ant1_crop.shape[1],3)
                            
        clf = KMeans(n_clusters = 5, n_init = 10)
                            
        Klabels = clf.fit_predict(modified_image)
                            
        counts = Counter(Klabels)
                            
        center_colors = clf.cluster_centers_
                            
        ordered_colors = [center_colors[k] for k in counts.keys()]
                            
        hex_colors = [RGB2HEX(ordered_colors[k]) for k in counts.keys()] ## hulls function
                            
        rgb_colors = [ordered_colors[k] for k in counts.keys()]    
            
    #fake_data = np.array([10,10,10,10,10,10,10,10,10,10])
        fake_data = np.array([20,20,20,20,20])
        plt.title('Colour Detection', fontsize=20)
        plt.pie(fake_data, labels = hex_colors, colors = hex_colors)                      
        plt.savefig(os.path.join(folder_path, (image_name + '-graph.png')))
        plt.close()
                            
        hull_matches = []
    
        for hexcode in hex_colors:
            rgb = list(hex_to_rgb(hexcode)) ## hulls function

        
        
        
            for colour, hull in hull_dict.items():
                hull_id = is_rgb_in_hull(rgb, hull) ## hulls function
            #hull_id = find_closest_hull(rgb, hull)
            
                if hull_id:

                    hull_matches.append(colour)
                
                
        blue_cnt = hull_matches.count('blue')
        red_cnt = hull_matches.count('red')                                   
        yellow_cnt = hull_matches.count('yellow')                                  
        green_cnt = hull_matches.count('green')  
        white_cnt = hull_matches.count('white') 
        pink_cnt = hull_matches.count('pink')
    
        col_cnts = { 'blue' : int(blue_cnt), 'red' : int(red_cnt), 'yellow' : int(yellow_cnt), 'green' : int(green_cnt), 'white' : int(white_cnt), 'pink': int(pink_cnt)}
    
    
        return col_cnts    
        
        
    elif ant1_theta == 0 or math.isnan(ant1_theta) or ant1_aby > 1065:
        col_cnts = {}
    
        return col_cnts





#    if ant1_theta == 0 or math.isnan(ant1_theta) or ant1_aby > 1065:
#        print('ant1 theta not able to be made')
        
        
        ## if no detections for whatever reason, do we go backwards or forwards??
        ## FOR NOW: forwards ##
        

#        for fi in reversed(range(0, frame_id)):
        
#            forward_check = data[(
#                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][fi]
                
#            if math.isnan(forward_check):
#                continue
#            else:
                
#                forward_x = data[(
#                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][fi]                                        
                
#                forward_y = data[(
#                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'y')][fi]
                
#                forward_tx = data[(
#                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'x')][fi]                                        
                
#                forward_ty = data[(
#                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'thorax', 'y')][fi]        

#                break
#        print(fi)            
#        ant1_theta = davi.get_vector_angle(forward_x, forward_y, forward_tx, forward_ty)
#        ant1_waffle = find_abdomen(forward_x, forward_y, ant1_theta)  
#        print(ant1_waffle)      





def hull_double_checker(video_path, frame_id, ant, hull_dict):

    folder_path = os.path.dirname(video_path)
    vidname = os.path.basename(video_path)
    form = '.png'
    full_vidname = vidname.strip('.mp4') + '-' + str(frame_id) + form
    image_name = ant + '-' + full_vidname
    
    ant1_crop = get_image(os.path.join(folder_path, image_name)) ## hulls function    

    modified_image = ant1_crop.reshape(ant1_crop.shape[0]*ant1_crop.shape[1],3)
                            
    clf = KMeans(n_clusters = 10, n_init = 10)
                            
    Klabels = clf.fit_predict(modified_image)
                            
    counts = Counter(Klabels)
                            
    center_colors = clf.cluster_centers_
                            
    ordered_colors = [center_colors[k] for k in counts.keys()]
                            
    hex_colors = [RGB2HEX(ordered_colors[k]) for k in counts.keys()]
                            
    rgb_colors = [ordered_colors[k] for k in counts.keys()]

    #fake_data = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
    fake_data = np.array([10,10,10,10,10,10,10,10,10,10])
    plt.title('Hull double checker', fontsize=20)
    plt.pie(fake_data, labels = hex_colors, colors = hex_colors)                      
    plt.savefig(os.path.join(folder_path, (image_name + '-graph.png')))
    plt.close()        


    hull_matches = []
    
    for hexcode in hex_colors:
        rgb = list(hex_to_rgb(hexcode)) ## hulls function

        
        
        
        for colour, hull in hull_dict.items():
            hull_id = is_rgb_in_hull(rgb, hull) ## hulls function
            

            
            if hull_id:

                hull_matches.append(colour)
                
                
    blue_cnt = hull_matches.count('blue')
    red_cnt = hull_matches.count('red')                                   
    yellow_cnt = hull_matches.count('yellow')                                  
    green_cnt = hull_matches.count('green')  
    white_cnt = hull_matches.count('white') 
    pink_cnt = hull_matches.count('pink')
    
    col_cnts = { 'blue' : int(blue_cnt), 'red' : int(red_cnt), 'yellow' : int(yellow_cnt), 'green' : int(green_cnt), 'white' : int(white_cnt), 'pink': int(pink_cnt)}
    
    
    return col_cnts


