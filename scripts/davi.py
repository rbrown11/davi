from scipy.spatial import ConvexHull, Delaunay
import math 
from shapely.geometry import Polygon
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance

## Temporal tracking functions ##

def find_distance(x, x1, y, y1):
    distance = ((x - x1)**2  + (y - y1)**2) **0.5
    return distance

## Colour waffle functions ##

def get_vector_angle(origin_x, origin_y, target_x, target_y):
    sign_angle=0
    if ((origin_x == target_x) and (origin_y + target_y)):
        attractor_direction_angle = np.nan
        return attractor_direction_angle
        
    else: 
        dx = target_x - origin_x
        dy = target_y - origin_y
        
        asinus = math.asin( dy/math.sqrt(pow(dx, 2) + pow(dy,2)))        
        
        if asinus < 0:
            sign_angle = -1
        elif asinus == 0:
            sign_angle = 0
        elif asinus > 0:
            sign_angle = 1
        
        if sign_angle == 0:
            if dx < 0:
                attractor_direction_angle = math.pi
            else:
                attractor_direction_angle = 0
        else:

            attractor_direction_angle = sign_angle * math.acos( dx/math.sqrt(pow(dx, 2) + pow(dy, 2)))
        return attractor_direction_angle

    
    
def calculate_new_coords(x_coord, y_coord, angle, distance):
    new_coord_x = x_coord + distance * math.cos(angle)
    new_coord_y = y_coord + distance * math.sin(angle)
    return new_coord_x, new_coord_y
    

    
def find_colour_distance(hex_color1, hex_color2):
    r1, g1, b1 = int(hex_color1[1:3], 16), int(hex_color1[3:5], 16), int(hex_color1[5:7], 16)
    r2, g2, b2 = int(hex_color2[1:3], 16), int(hex_color2[3:5], 16), int(hex_color2[5:7], 16)
    distance = ((0.3 *(r1 - r2)) ** 2 + (0.59 * (g1 - g2)) ** 2 + (0.11 * (b1 - b2)) ** 2) ** 0.5
    return distance     


def subtract_colours_2(col1, col2):
    hull = ConvexHull(col1, qhull_options="QJ")
    hull_points = [tuple(point) for point in hull.points[hull.vertices]]
    hull2 = ConvexHull(col2, qhull_options="QJ")
    hull_points2 = [tuple(point) for point in hull2.points[hull2.vertices]]
    col_vals = len(list(set(hull_points)-set(hull_points2)))
    return col_vals

# def subtract_colours_2(col1, col2):
#     hull = ConvexHull(col1)
#     hull_points = [tuple(point) for point in hull.points[hull.vertices]]
#     col2 = [(float(x), float(y), float(z)) for x, y, z in col2]
#     col_vals = len(list(set(col2)-set(hull_points)))
#     return col_vals


def create_waffle(x_coord, y_coord, theta, pix):
    length = 14
    width = 16
    extra = 4
    
    
    AN0 = calculate_new_coords(x_coord, y_coord, theta+(math.pi/2), length/2)
    AN4 = calculate_new_coords(x_coord, y_coord, theta-(math.pi/2), length/2)

    AA0 = calculate_new_coords(AN0[0], AN0[1], theta, width/2)
    AT0 = calculate_new_coords(AN0[0], AN0[1], theta+math.pi, width/2)

    AA4 = calculate_new_coords(AN4[0], AN4[1], theta, width/2)
    AT4 = calculate_new_coords(AN4[0], AN4[1], theta+math.pi, width/2)

    AN2 = (x_coord, y_coord)
    AA2 = calculate_new_coords(AN2[0], AN2[1], theta, width/2)
    AT2 = calculate_new_coords(AN2[0], AN2[1], theta+math.pi, width/2)

    AA1 = calculate_new_coords(AA2[0], AA2[1], theta+(math.pi/2), length/4)
    AN1 = calculate_new_coords(AA1[0], AA1[1], theta+math.pi, width/2)
    AT1 = calculate_new_coords(AN1[0], AN1[1], theta+math.pi, width/2)

    AA3 = calculate_new_coords(AA2[0], AA2[1], theta-(math.pi/2), length/4)
    AN3 = calculate_new_coords(AA3[0], AA3[1], theta+math.pi, width/2)
    AT3 = calculate_new_coords(AN3[0], AN3[1], theta+math.pi, width/2)

    AR0 = calculate_new_coords(AN0[0], AN0[1], theta, width/4)
    AH0 = calculate_new_coords(AN1[0], AN1[1], theta, width/4)
    AY0 = calculate_new_coords(AN2[0], AN2[1], theta, width/4)
    AM0 = calculate_new_coords(AN3[0], AN3[1], theta, width/4)
    AE0 = calculate_new_coords(AN4[0], AN4[1], theta, width/4)

    AR1 = calculate_new_coords(AT0[0], AT0[1], theta, width/4)
    AH1 = calculate_new_coords(AT1[0], AT1[1], theta, width/4)
    AY1 = calculate_new_coords(AT2[0], AT2[1], theta, width/4)
    AM1 = calculate_new_coords(AT3[0], AT3[1], theta, width/4)
    AE1 = calculate_new_coords(AT4[0], AT4[1], theta, width/4)

    AX0 = calculate_new_coords(AT0[0], AT0[1], theta+math.pi, extra)
    AX1 = calculate_new_coords(AT1[0], AT1[1], theta+math.pi, extra)        
    AX2 = calculate_new_coords(AT2[0], AT2[1], theta+math.pi, extra)
    AX3 = calculate_new_coords(AT3[0], AT3[1], theta+math.pi, extra)
    AX4 = calculate_new_coords(AT4[0], AT4[1], theta+math.pi, extra)
    
    ## NEW COORDS ##
    
    AB0 = calculate_new_coords(AA1[0], AA1[1], theta+(math.pi/2), length/8)
    AB1 = calculate_new_coords(AB0[0], AB0[1], theta+math.pi, width/8)
    AB2 = calculate_new_coords(AB1[0], AB1[1], theta+math.pi, width/8)
    AB3 = calculate_new_coords(AB2[0], AB2[1], theta+math.pi, width/8)    
    AB4 = calculate_new_coords(AB3[0], AB3[1], theta+math.pi, width/8)
    AB5 = calculate_new_coords(AB4[0], AB4[1], theta+math.pi, width/8)
    AB6 = calculate_new_coords(AB5[0], AB5[1], theta+math.pi, width/8)
    AB7 = calculate_new_coords(AB6[0], AB6[1], theta+math.pi, width/8)    
    AB8 = calculate_new_coords(AB7[0], AB7[1], theta+math.pi, width/8)
    AB9 = calculate_new_coords(AB8[0], AB8[1], theta+math.pi, width/8)
    AB10 = calculate_new_coords(AB9[0], AB9[1], theta+math.pi, width/8)    
    
    AC0 = calculate_new_coords(AA2[0], AA2[1], theta+(math.pi/2), length/8)
    AC1 = calculate_new_coords(AC0[0], AC0[1], theta+math.pi, width/8)
    AC2 = calculate_new_coords(AC1[0], AC1[1], theta+math.pi, width/8)
    AC3 = calculate_new_coords(AC2[0], AC2[1], theta+math.pi, width/8)    
    AC4 = calculate_new_coords(AC3[0], AC3[1], theta+math.pi, width/8)
    AC5 = calculate_new_coords(AC4[0], AC4[1], theta+math.pi, width/8)
    AC6 = calculate_new_coords(AC5[0], AC5[1], theta+math.pi, width/8)
    AC7 = calculate_new_coords(AC6[0], AC6[1], theta+math.pi, width/8)    
    AC8 = calculate_new_coords(AC7[0], AC7[1], theta+math.pi, width/8)
    AC9 = calculate_new_coords(AC8[0], AC8[1], theta+math.pi, width/8)
    AC10 = calculate_new_coords(AC9[0], AC9[1], theta+math.pi, width/8)
    
    AD0 = calculate_new_coords(AA3[0], AA3[1], theta+(math.pi/2), length/8)
    AD1 = calculate_new_coords(AD0[0], AD0[1], theta+math.pi, width/8)
    AD2 = calculate_new_coords(AD1[0], AD1[1], theta+math.pi, width/8)
    AD3 = calculate_new_coords(AD2[0], AD2[1], theta+math.pi, width/8)    
    AD4 = calculate_new_coords(AD3[0], AD3[1], theta+math.pi, width/8)
    AD5 = calculate_new_coords(AD4[0], AD4[1], theta+math.pi, width/8)
    AD6 = calculate_new_coords(AD5[0], AD5[1], theta+math.pi, width/8)
    AD7 = calculate_new_coords(AD6[0], AD6[1], theta+math.pi, width/8)    
    AD8 = calculate_new_coords(AD7[0], AD7[1], theta+math.pi, width/8)
    AD9 = calculate_new_coords(AD8[0], AD8[1], theta+math.pi, width/8)
    AD10 = calculate_new_coords(AD9[0], AD9[1], theta+math.pi, width/8)
    
    AF0 = calculate_new_coords(AA4[0], AA4[1], theta+(math.pi/2), length/8)
    AF1 = calculate_new_coords(AF0[0], AF0[1], theta+math.pi, width/8)
    AF2 = calculate_new_coords(AF1[0], AF1[1], theta+math.pi, width/8)
    AF3 = calculate_new_coords(AF2[0], AF2[1], theta+math.pi, width/8)    
    AF4 = calculate_new_coords(AF3[0], AF3[1], theta+math.pi, width/8)
    AF5 = calculate_new_coords(AF4[0], AF4[1], theta+math.pi, width/8)
    AF6 = calculate_new_coords(AF5[0], AF5[1], theta+math.pi, width/8)
    AF7 = calculate_new_coords(AF6[0], AF6[1], theta+math.pi, width/8)    
    AF8 = calculate_new_coords(AF7[0], AF7[1], theta+math.pi, width/8)
    AF9 = calculate_new_coords(AF8[0], AF8[1], theta+math.pi, width/8)
    AF10 = calculate_new_coords(AF9[0], AF9[1], theta+math.pi, width/8)  
    
    BX0 = calculate_new_coords(AB1[0], AB1[1], theta+(math.pi/2), length/8)
    BX1 = calculate_new_coords(AC1[0], AC1[1], theta+(math.pi/2), length/8)
    BX2 = calculate_new_coords(AD1[0], AD1[1], theta+(math.pi/2), length/8)
    BX3 = calculate_new_coords(AF1[0], AF1[1], theta+(math.pi/2), length/8)
    BX4 = calculate_new_coords(AF1[0], AF1[1], theta-(math.pi/2), length/8)

    BC0 = calculate_new_coords(AB3[0], AB3[1], theta+(math.pi/2), length/8)
    BC1 = calculate_new_coords(AC3[0], AC3[1], theta+(math.pi/2), length/8)
    BC2 = calculate_new_coords(AD3[0], AD3[1], theta+(math.pi/2), length/8)
    BC3 = calculate_new_coords(AF3[0], AF3[1], theta+(math.pi/2), length/8)
    BC4 = calculate_new_coords(AF3[0], AF3[1], theta-(math.pi/2), length/8)
               
    BD0 = calculate_new_coords(AB5[0], AB5[1], theta+(math.pi/2), length/8)
    BD1 = calculate_new_coords(AC5[0], AC5[1], theta+(math.pi/2), length/8)
    BD2 = calculate_new_coords(AD5[0], AD5[1], theta+(math.pi/2), length/8)
    BD3 = calculate_new_coords(AF5[0], AF5[1], theta+(math.pi/2), length/8)
    BD4 = calculate_new_coords(AF5[0], AF5[1], theta-(math.pi/2), length/8)               

    BE0 = calculate_new_coords(AB7[0], AB7[1], theta+(math.pi/2), length/8)
    BE1 = calculate_new_coords(AC7[0], AC7[1], theta+(math.pi/2), length/8)
    BE2 = calculate_new_coords(AD7[0], AD7[1], theta+(math.pi/2), length/8)
    BE3 = calculate_new_coords(AF7[0], AF7[1], theta+(math.pi/2), length/8)
    BE4 = calculate_new_coords(AF7[0], AF7[1], theta-(math.pi/2), length/8) 

    BF0 = calculate_new_coords(AB9[0], AB9[1], theta+(math.pi/2), length/8)
    BF1 = calculate_new_coords(AC9[0], AC9[1], theta+(math.pi/2), length/8)
    BF2 = calculate_new_coords(AD9[0], AD9[1], theta+(math.pi/2), length/8)
    BF3 = calculate_new_coords(AF9[0], AF9[1], theta+(math.pi/2), length/8)
    BF4 = calculate_new_coords(AF9[0], AF9[1], theta-(math.pi/2), length/8) 
    
    col_list = []
    col_list.append(pix[AA0[0], AA0[1]])
    col_list.append(pix[AN0[0], AN0[1]])
    col_list.append(pix[AT0[0], AT0[1]])
    col_list.append(pix[AA1[0], AA1[1]])
    col_list.append(pix[AN1[0], AN1[1]])
    col_list.append(pix[AT1[0], AT1[1]])
    col_list.append(pix[AA2[0], AA2[1]])
    col_list.append(pix[AN2[0], AN2[1]])
    col_list.append(pix[AT2[0], AT2[1]])
    col_list.append(pix[AA3[0], AA3[1]])
    col_list.append(pix[AN3[0], AN3[1]])
    col_list.append(pix[AT3[0], AT3[1]])
    col_list.append(pix[AA4[0], AA4[1]])
    col_list.append(pix[AN4[0], AN4[1]])
    col_list.append(pix[AT4[0], AT4[1]])
    col_list.append(pix[AR0[0], AR0[1]])
    col_list.append(pix[AH0[0], AH0[1]])
    col_list.append(pix[AY0[0], AY0[1]])
    col_list.append(pix[AM0[0], AM0[1]])
    col_list.append(pix[AE0[0], AE0[1]])
    col_list.append(pix[AR1[0], AR1[1]])
    col_list.append(pix[AH1[0], AH1[1]])
    col_list.append(pix[AY1[0], AY1[1]])
    col_list.append(pix[AM1[0], AM1[1]])
    col_list.append(pix[AE1[0], AE1[1]])
    col_list.append(pix[AX0[0], AX0[1]])
    col_list.append(pix[AX1[0], AX1[1]])
    col_list.append(pix[AX2[0], AX2[1]])
    col_list.append(pix[AX3[0], AX3[1]])
    col_list.append(pix[AX4[0], AX4[1]])
    
    col_list.append(pix[AB0[0], AB0[1]])
    col_list.append(pix[AB1[0], AB1[1]])
    col_list.append(pix[AB2[0], AB2[1]])
    col_list.append(pix[AB3[0], AB3[1]])
    col_list.append(pix[AB4[0], AB4[1]])
    col_list.append(pix[AB5[0], AB5[1]])
    col_list.append(pix[AB6[0], AB6[1]])
    col_list.append(pix[AB7[0], AB7[1]])
    col_list.append(pix[AB8[0], AB8[1]])
    col_list.append(pix[AB9[0], AB9[1]])
    col_list.append(pix[AB10[0], AB10[1]])
    
    col_list.append(pix[AC0[0], AC0[1]])
    col_list.append(pix[AC1[0], AC1[1]])
    col_list.append(pix[AC2[0], AC2[1]])
    col_list.append(pix[AC3[0], AC3[1]])
    col_list.append(pix[AC4[0], AC4[1]])
    col_list.append(pix[AC5[0], AC5[1]])
    col_list.append(pix[AC6[0], AC6[1]])
    col_list.append(pix[AC7[0], AC7[1]])
    col_list.append(pix[AC8[0], AC8[1]])
    col_list.append(pix[AC9[0], AC9[1]])
    col_list.append(pix[AC10[0], AC10[1]])
    
    col_list.append(pix[AD0[0], AD0[1]])
    col_list.append(pix[AD1[0], AD1[1]])
    col_list.append(pix[AD2[0], AD2[1]])
    col_list.append(pix[AD3[0], AD3[1]])
    col_list.append(pix[AD4[0], AD4[1]])
    col_list.append(pix[AD5[0], AD5[1]])
    col_list.append(pix[AD6[0], AD6[1]])
    col_list.append(pix[AD7[0], AD7[1]])
    col_list.append(pix[AD8[0], AD8[1]])
    col_list.append(pix[AD9[0], AD9[1]])
    col_list.append(pix[AD10[0], AD10[1]])    

    col_list.append(pix[AF0[0], AF0[1]])
    col_list.append(pix[AF1[0], AF1[1]])
    col_list.append(pix[AF2[0], AF2[1]])
    col_list.append(pix[AF3[0], AF3[1]])
    col_list.append(pix[AF4[0], AF4[1]])
    col_list.append(pix[AF5[0], AF5[1]])
    col_list.append(pix[AF6[0], AF6[1]])
    col_list.append(pix[AF7[0], AF7[1]])
    col_list.append(pix[AF8[0], AF8[1]])
    col_list.append(pix[AF9[0], AF9[1]])
    col_list.append(pix[AF10[0], AF10[1]])    
    
    col_list.append(pix[BX0[0], BX0[1]])
    col_list.append(pix[BX1[0], BX1[1]])
    col_list.append(pix[BX2[0], BX2[1]])
    col_list.append(pix[BX3[0], BX3[1]])
    col_list.append(pix[BX4[0], BX4[1]])
    col_list.append(pix[BC0[0], BC0[1]])
    col_list.append(pix[BC1[0], BC1[1]])
    col_list.append(pix[BC2[0], BC2[1]])
    col_list.append(pix[BC3[0], BC3[1]])
    col_list.append(pix[BC4[0], BC4[1]])
    col_list.append(pix[BD0[0], BD0[1]])
    col_list.append(pix[BD1[0], BD1[1]])
    col_list.append(pix[BD2[0], BD2[1]])
    col_list.append(pix[BD3[0], BD3[1]])
    col_list.append(pix[BD4[0], BD4[1]])
    col_list.append(pix[BE0[0], BE0[1]])
    col_list.append(pix[BE1[0], BE1[1]])
    col_list.append(pix[BE2[0], BE2[1]])
    col_list.append(pix[BE3[0], BE3[1]])
    col_list.append(pix[BE4[0], BE4[1]])
    col_list.append(pix[BF0[0], BF0[1]])
    col_list.append(pix[BF1[0], BF1[1]])
    col_list.append(pix[BF2[0], BF2[1]])
    col_list.append(pix[BF3[0], BF3[1]])
    col_list.append(pix[BF4[0], BF4[1]])


    
    return col_list
    
    
def find_box_space(bbox):
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]
    
    coords = ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
    box = Polygon(coords)
    
    return box



def calculate_iou(box1, box2):
    iou = box1.intersection(box2).area / box1.union(box2).area
    return iou
    
      
    
    
def make_multiple_colour_comparisons(ant, ant_waffle, data_frame, index, video_path):
    waffles = ['ind1', 'ind2', 'ind3', 'ind4', 'ind5', 'ind6']
    
        
    if ant in waffles:
        waffles.remove(ant)
    
    
    vidname = os.path.basename(video_path)        
    frame_id = index
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    form = '.png'
    full_vidname = vidname.strip('.MTS') + "-" + str(frame_id) + form
        
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
             
    colour_comparisons = {}
            
    for waffle in waffles:
    
         #print(waffle)    
        
         waffle_abx = data_frame[(
                        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'abdomen', 'x')][index]
         waffle_aby = data_frame[(
                        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'abdomen', 'y')][index]
         waffle_thx = data_frame[(
                        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'thorax', 'x')][index]
         waffle_thy = data_frame[(
                        'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'thorax', 'y')][index]
             
             
         waffle_theta = get_vector_angle(
                        waffle_abx, waffle_aby, waffle_thx, waffle_thy)
                        
         waffle_waffle = []
                        
         if waffle_theta == 0 or math.isnan(waffle_theta):
             for r in reversed(range(0, index)):
                 w_check = data_frame[(
                                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'abdomen', 'x')][r]
                 w_check2 = data_frame[(
                                'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle,  'thorax', 'x')][r]

            
                 if math.isnan(w_check) or math.isnan(w_check2):
                     continue
                 else:

                     waffle_abx = data_frame[(
                                    'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'abdomen', 'x')][r]
                     waffle_aby = data_frame[(
                                    'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'abdomen', 'y')][r]
                     waffle_thx = data_frame[(
                                    'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'thorax', 'x')][r]
                     waffle_thy = data_frame[(
                                    'DLC_dlcrnetms5_full-modelJan10shuffle1_100000', waffle, 'thorax', 'y')][r]

                     break
                                    
             waffle_theta = get_vector_angle(waffle_abx, waffle_aby, waffle_thx, waffle_thy)
                 
             if waffle_theta == 0 or math.isnan(waffle_theta) or waffle_aby > 1065:
                 pair_ants = ant + '_' + waffle
                 pairwise_comparison = 30
                     
             else:
                 waffle_waffle = create_waffle(waffle_abx, waffle_aby, waffle_theta, pix)
                 #print('reverse loop theta:', waffle_theta)    
                 pair_ants = ant + '_' + waffle
                 
                 pairwise_comparison = subtract_colours_2(ant_waffle, waffle_waffle)
                     
#                 pairwise_comparison = subtract_colours_2(waffle_waffle, ant_waffle)

         else:
             #print('reverse loop theta:', waffle_theta)    
             waffle_waffle = create_waffle(waffle_abx, waffle_aby, waffle_theta, pix)
                     
             pair_ants = ant + '_' + waffle
             
             pairwise_comparison = subtract_colours_2(ant_waffle, waffle_waffle)
                     
 #            pairwise_comparison = subtract_colours_2(waffle_waffle, ant_waffle)
                     
         

         current = { pair_ants : pairwise_comparison }
                 
         #print(current)        
         colour_comparisons.update(current)
                     
                     
    return colour_comparisons                    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
