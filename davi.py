from scipy.spatial import ConvexHull, Delaunay
import math 
from shapely.geometry import Polygon
import numpy as np

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
    
