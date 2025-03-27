
import pandas as pd
import math
import davi
import random
import numpy as np
import hulls
from collections import Counter
import PySimpleGUI as sg
from PIL import Image, ImageTk, ImageEnhance
import io
import os
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt 
import flags_manual_check


def load_image_in_memory(image):
    """Convert a PIL image to a format compatible with PySimpleGUI without saving to disk."""
    bio = io.BytesIO()
    image.save(bio, format="PNG")  # Save the image to an in-memory file
    return bio.getvalue()  # Return the in-memory file's contents as bytes

def load_image(path, window):
    try:
        image = Image.open(path)
        image.thumbnail((800, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
    except:
        print(f"Unable to open {path}!")


def load_image2(path, window):
    try:
        image = Image.open(path)
        image.thumbnail((720, 405))
        photo_img = ImageTk.PhotoImage(image)
        window["ant_image"].update(data=photo_img)
    except:
        print(f"Unable to open {path}!")



def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



def extract_frame(video_path, data, i):
    """Extract the image of the ant without saving it to disk, and return it as a PIL image."""
    
    # Open the video file and go to the specific frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {i} from video.")
        return None

    # Convert the frame from BGR to RGB (cv2 uses BGR by default)
    colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a PIL image for further processing
    im = Image.fromarray(colour_converted)

    # Enhance the brightness and contrast
    enhancer = ImageEnhance.Brightness(im)
    im_output = enhancer.enhance(1.5)
    contrast = ImageEnhance.Contrast(im_output)
    image = contrast.enhance(1.2)
    image.thumbnail((1000, 600))
 
    # Release the video capture object
    cap.release()

    # Return the cropped and enhanced image as a PIL image object
    return image  

 
def extract_solo_ants(video_path, ant, data, i):
    """Extract the image of the ant without saving it to disk, and return it as a PIL image."""
    
    # Open the video file and go to the specific frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {i} from video.")
        return None

    # Convert the frame from BGR to RGB (cv2 uses BGR by default)
    colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a PIL image for further processing
    im = Image.fromarray(colour_converted)

    # Enhance the brightness and contrast
    enhancer = ImageEnhance.Brightness(im)
    im_output = enhancer.enhance(1.5)
    contrast = ImageEnhance.Contrast(im_output)
    im_output_T = contrast.enhance(1.2)
    
    # Determine the coordinates for cropping based on the ant's position
    abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][i]
    aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'y')][i]
    
    # Define the crop box (ensure you handle edges correctly)
    crop_parameter_lowx = max(0, abx - 40)
    crop_parameter_highx = min(frame.shape[1], abx + 40)
    crop_parameter_lowy = max(0, aby - 40)
    crop_parameter_highy = min(frame.shape[0], aby + 40)
    
    # Crop the image based on the calculated coordinates
    ROI = (crop_parameter_lowx, crop_parameter_lowy, crop_parameter_highx, crop_parameter_highy)
    ant_image = im_output_T.crop(ROI)
    
    # Release the video capture object
    cap.release()

    # Return the cropped and enhanced image as a PIL image object
    return ant_image  
  
###########################################  
#### this function works but is slow!! ####  
###########################################  
    
#def extract_solo_ants(video_path, image_path, ant, data, i, window):
    
#    cap = cv2.VideoCapture(video_path)
#    fps = cap.get(cv2.CAP_PROP_FPS)
    
#    folder_path = os.path.dirname(video_path)
#    form = '.png'
    
#    vidname = os.path.basename(video_path)
    
#    full_vidname = vidname.strip('.mp4') + '-' + str(i) + form
#    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#    ret, frame = cap.read()
    
#    if ret:
#        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        
#    im = Image.fromarray(colour_converted)
    
#    enhancer = ImageEnhance.Brightness(im)
#    factor = 1.5
#    im_output = enhancer.enhance(factor)
#    contrast = ImageEnhance.Contrast(im_output)
#    factor_c = 1.2
#    im_output_T = contrast.enhance(factor_c)
    
#    im_output_T.save(os.path.join(folder_path, full_vidname))
    
#    image_path = os.path.join(folder_path, full_vidname)
    
    
#    image = get_image(image_path) #hulls / maybe copy function here or write it out
        
#    abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][i]
#    aby = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'y')][i]
        
        
      
#    if abx < 40:
#        crop_parameter_lowx = 0
#    else:
#        crop_parameter_lowx = abx - 40
            
#    if abx > 1880:
#        crop_parameter_highx = 1920
#    else:
#        crop_parameter_highx = abx + 40
            
#    if aby < 40:
#        crop_parameter_lowy = 0
#    else: 
#        crop_parameter_lowy = aby - 40
        
#    if aby > 1040:
#        crop_parameter_highy = 1080
#    else: 
#        crop_parameter_highy = aby + 40        
    
#    im = Image.fromarray(image)
    
    
#    ROI = (crop_parameter_lowx, crop_parameter_lowy, crop_parameter_highx, crop_parameter_highy)
        
    
    
        
#    cropped = im_output_T.crop(ROI)
        
#    crop_ant_image = os.path.join(folder_path, ant)
#    cropped.save(crop_ant_image, 'png')
    #crop_path = os.path.join(image_path, ant+'.png')
    
#    load_image2(crop_ant_image, window)





#def run_manual_check(i, data, video_path, second_uncertain_ids, key, folder_path):

#    cap = cv2.VideoCapture(video_path)
#    fps = cap.get(cv2.CAP_PROP_FPS)

#    vidname = os.path.basename(video_path)
#    form = '.png'
#    full_vidname = vidname.strip('.mp4') + '-' + str(i) + form
#    image_path = os.path.join(folder_path, full_vidname)


#    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#    ret, frame = cap.read()
#    
#    if ret:
#        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        
#    im = Image.fromarray(colour_converted)
#    
#    enhancer = ImageEnhance.Brightness(im)
#    factor = 1.5
#    im_output = enhancer.enhance(factor)
#    contrast = ImageEnhance.Contrast(im_output)
#    factor_c = 1.2
#    im_output_T = contrast.enhance(factor_c)
    
#    im_output_T.save(os.path.join(folder_path, full_vidname))
    


#    def main():
#        unable_to_id = []
#        elements = [
#            [sg.Image(key='image')],
#            [sg.Image(key='ant_image')],
#            [sg.Text('Ant ID:', key='-TEXT-')],
#            [
#                sg.Button('Blue', key='blue', visible=False),
#                sg.Button('Red', key='red', visible=False),
#                sg.Button('Yellow', key='yellow', visible=False),
#                sg.Button('Green', key='green', visible=False),
#                sg.Button('White', key='white', visible=False),
#                sg.Button('Pink', key='pink', visible=False)
#            ],
#            [sg.Button('Error in key', visible=True)]
#        ]

#        window = sg.Window('Errored frame', elements, size=(1600, 1000), finalize=True)
        
#        image_frame = load_image(image_path, window)

#        for ant in list(second_uncertain_ids):  # Use a copy of the list to avoid mutation issues
        
#            check_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][i]
            
#            if math.isnan(check_abx):
#                unable_to_id.append(ant)
            
#                continue
#            else:
            
            
#                extract_solo_ants(video_path, image_path, ant, data, i, window)
#                window['-TEXT-'].update('Ant ID:' + ant)
                #window['ant_image'].update(filename=image_path)

            # Show buttons for colors not yet assigned to any ant
#                if 'blue' not in key.values():
#                    window['blue'].update(visible=True)
#                if 'red' not in key.values():
#                    window['red'].update(visible=True)
#                if 'yellow' not in key.values():
#                    window['yellow'].update(visible=True)
#                if 'green' not in key.values():
#                    window['green'].update(visible=True)
#                if 'white' not in key.values():
#                    window['white'].update(visible=True)
#                if 'pink' not in key.values():
#                    window['pink'].update(visible=True)

            # Event loop for user interaction
#                while True:
#                    event, values = window.read()
                
#                    if event in (sg.WIN_CLOSED, 'Error in key'):
#                        window.close()
#                        return key  # Exit if window is closed or an error is detected

#                    if event in ['blue', 'red', 'yellow', 'green', 'white', 'pink']:
#                        color = event
#                        key[ant] = color  # Update the key with the ant's ID and selected color

                    # Hide all buttons after selection
#                        for w in window.key_dict:
#                            ele = window[w]
#                            if isinstance(ele, sg.Button):
#                                ele.update(visible=False)

#                        second_uncertain_ids.remove(ant)  # Remove the ant from the list
#                        break
                        

                        #break  # Break to load the next ant




            
#            for a in unable_to_id:
#                key.update({ a : 'unknown' } )
                
                 
            
#        window.close()
#        return key

#    return main() ## check if we need this




def run_manual_check(i, data, video_path, second_uncertain_ids, key, folder_path):
    def main():
        unable_to_id = []
        elements = [
            [sg.Image(key='image')],
            [sg.Image(key='ant_image')],
            [sg.Text('Ant ID:', key='-TEXT-')],
            [
                sg.Button('Blue', key='blue', visible=False),
                sg.Button('Red', key='red', visible=False),
                sg.Button('Yellow', key='yellow', visible=False),
                sg.Button('Green', key='green', visible=False),
                sg.Button('White', key='white', visible=False),
                sg.Button('Pink', key='pink', visible=False), 
                sg.Button('Unknown', key='unknown', visible=False)
            ]#,
            #[sg.Button('Error in key', visible=True)]
        ]

        window = sg.Window('Errored frame', elements, size=(1600, 1000), finalize=True)

        # Track the in-memory image to delete it when needed
        ant_image_bytes = None
        frame_bytes = None
        
        
        frame = extract_frame(video_path, data, i)
        
        frame_bytes = load_image_in_memory(frame)
        window['image'].update(data=frame_bytes)

        for ant in list(second_uncertain_ids):  # Use a copy of the list to avoid mutation issues
        
            check_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][i]
            
            if math.isnan(check_abx):
                unable_to_id.append(ant)
                continue
            else:
                # Extract the ant image here (using your own function)
                ant_image = extract_solo_ants(video_path, ant, data, i)  # Modify to return a PIL image

                # Convert the PIL image to a format compatible with PySimpleGUI
                ant_image_bytes = load_image_in_memory(ant_image)

                # Update the image in the GUI window
                window['ant_image'].update(data=ant_image_bytes)
                window['-TEXT-'].update('Ant ID:' + ant)

                # Remove the in-memory image after use to free up memory
                #del ant_image_bytes
                #del ant_image

            # Show buttons for colors not yet assigned to any ant
                for color in ['blue', 'red', 'yellow', 'green', 'white', 'pink', 'unknown']:
#               # if color not in key.values():
                     window[color].update(visible=True)



            # Event loop for user interaction
            while True:
                event, values = window.read()
                if event in (sg.WIN_CLOSED, 'Error in key'):
                    window.close()
                    return key  # Exit if window is closed or an error is detected

                if event in ['blue', 'red', 'yellow', 'green', 'white', 'pink', 'unknown']:
                    color = event
                    key[ant] = color  # Update the key with the ant's ID and selected color

                    # Hide all buttons after selection
                    for btn in ['blue', 'red', 'yellow', 'green', 'white', 'pink', 'unknown']:
                        window[btn].update(visible=False)

                    second_uncertain_ids.remove(ant)  # Remove the ant from the list
                    break  # Break to load the next ant

        for a in unable_to_id:
            key.update({ a : 'unknown' } )
                
        window.close()

        # Ensure all in-memory images are deleted after the window closes
        if ant_image_bytes is not None:
            del ant_image_bytes
            del ant_image
            del frame_bytes
            del frame
            
            
        return key

    return main()




###########################################
#### this function works but is slow!! ####
###########################################


#def run_manual_check(i, data, video_path, second_uncertain_ids, key, folder_path):

#    vidname = os.path.basename(video_path)
#    form = '.png'
#    full_vidname = vidname.strip('.mp4') + '-' + str(i) + form
#    image_path = os.path.join(folder_path, full_vidname)

    # Video capture setup for extracting frames
#    cap = cv2.VideoCapture(video_path)
#    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#    ret, frame = cap.read()
    
#    if ret:
#        colour_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#    im = Image.fromarray(colour_converted)
#    enhancer = ImageEnhance.Brightness(im)
#    factor = 1.5
#    im_output = enhancer.enhance(factor)
#    contrast = ImageEnhance.Contrast(im_output)
#    factor_c = 1.2
#    im_output_T = contrast.enhance(factor_c)
#    im_output_T.save(os.path.join(folder_path, full_vidname))

#    def manual_check_loop():
#        unable_to_id = []
#        elements = [
#            [sg.Image(key='image')],
#            [sg.Image(key='ant_image')],
#            [sg.Text('Ant ID:', key='-TEXT-')],
#            [
#                sg.Button('Blue', key='blue', visible=False),
#                sg.Button('Red', key='red', visible=False),
#                sg.Button('Yellow', key='yellow', visible=False),
#                sg.Button('Green', key='green', visible=False),
#                sg.Button('White', key='white', visible=False),
#                sg.Button('Pink', key='pink', visible=False), 
#                sg.Button('Unknown', key='unknown', visible=False)
#            ],
#            [sg.Button('Error in key', visible=True)]
#        ]

        # Create and finalize the window
#        window = sg.Window('Errored frame', elements, size=(1600, 1000), finalize=True)
        
#        load_image(image_path, window)  # Load and display the initial image

#        for ant in list(second_uncertain_ids):  # Process each ant in the list

#            check_abx = data[('DLC_dlcrnetms5_full-modelJan10shuffle1_100000', ant, 'abdomen', 'x')][i]
            
#            if math.isnan(check_abx):
#                unable_to_id.append(ant)
#                continue  # Skip to the next ant if there's an issue
            
#            extract_solo_ants(video_path, image_path, ant, data, i, window)
#            window['-TEXT-'].update('Ant ID:' + ant)

            # Show buttons for colors not yet assigned to any ant
#            for color in ['blue', 'red', 'yellow', 'green', 'white', 'pink', 'unknown']:
#               # if color not in key.values():
#                window[color].update(visible=True)

            # Event loop for user interaction
#            while True:
#                event, values = window.read()

#                if event in (sg.WIN_CLOSED, 'Error in key'):
#                    window.close()
#                    return key  # Exit if the window is closed or an error is detected

#                if event in ['blue', 'red', 'yellow', 'green', 'white', 'pink', 'unknown']:
#                    color = event
#                    key[ant] = color  # Update the key with the ant's ID and selected color

                    # Hide all buttons after selection
#                    for btn in ['blue', 'red', 'yellow', 'green', 'white', 'pink', 'unknown']:
#                        window[btn].update(visible=False)

#                    second_uncertain_ids.remove(ant)  # Remove the processed ant from the list
#                    break  # Break to load the next ant

        # Handle any ants that couldn't be identified
#        for a in unable_to_id:
#            key.update({a: 'unknown'})

#        window.close()
#        return key

#    return manual_check_loop()

