#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  22 22:54:14 2021.

@author: seb

Utilites to load, convert and save images.
"""

import os
import cv2
import numpy as np

def load_image(file_path):
    """Load image from given file."""
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist: {}".format(file_path))
        
    
    image = cv2.imread(file_path)
    
    return(image)


def show_image(img, label=''):
    """
    Show an image in external window.
    
    Parameters
    ----------
    img : numpy array
        An image stored as numpy array.
        
    label : str, optional
        Extra info to show in window title. The default is ''.

    Returns
    -------
    None.

    """
    if type(img) is not np.ndarray:
        raise TypeError("Incorrect img format. Must be numpy.ndarray but it is {}".format(type(img)))
    window_title = 'Image {}'.format(label)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)
    
    pass
    
    
def show_blob(blob, label="Blob image"):
    """
    Show image transformed to blob.

    Parameters
    ----------
    img : numpy array
        An image which was transformed by cv2.dnn.blobFromImage function.
    label : str, optional
        Extra info to show in window title. The default is "Blob image".

    Returns
    -------
    None.

    """
    if type(blob) is not np.ndarray:
        raise TypeError("Incorrect img format. Must be numpy.ndarray but it is {}".format(type(blob)))
        
    blob_to_show = blob[0, :, :, :].transpose(1,2,0)
    
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.imshow(label, cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyWindow(label)
    
    pass


def get_image_size(img):
    
    if not isinstance(img, np.ndarray):
        raise TypeError("'img' must be numpy.ndarray type but it is {}.".format(type(img)))
    
    img_size = img.shape[:2]
        
    return(img_size)


def transform_image_size_to_multipliers(img_size):
    
    if type(img_size) is not tuple and type(img_size) is not list:
        raise TypeError("'img_size' must be tuple or list type but it is {}.".format(type(img_size)))
    
    img_size=[img_size[1],img_size[0], img_size[1], img_size[0]]
    
    img_multipliers = np.array(img_size).flatten()
    
    return(img_multipliers)


def generate_box_colors(n):
    """
    Generate random RGB colors for drawing boxes around detected objects.

    Parameters
    ----------
    n : int
        Number of colors to generate.

    Returns
    -------
    Numpy array.

    """
    if type(n) is not int:
        raise TypeError('n must be int type, but it is {}'.format(type(n)))
        
    rgb_box_colors = np.random.randint(0,255,size=(n, 3))
    
    return(rgb_box_colors)


def transform_predictions_into_boxes(predictions, 
                                     labels,
                                     min_probability=0.5):
    
    boxes = []
    confidences = []
    class_numbers = []
    
    for layer_pred in predictions:
        for detected_object in layer_pred:
            
            # parameters in detect_object:
                # 0:4 --> x_center, y_center, width, height
                # 5:  --> probailities if assumed labels found
            
            object_num, object_proba = get_label_num_and_proba(detected_object)
            
            if object_proba > min_probability:
                                
                x_min, y_min, width, height = get_box_size(detected_object)
                
                boxes.append([x_min, y_min, width, height])
                confidences.append(object_proba)
                class_numbers.append(object_num)
        
    return(boxes, confidences, class_numbers)


def get_label_num_and_proba(obj_params):
    
    scores = obj_params[5:]
    selected_object_num = np.argmax(scores)
    proba_of_selected_object = float(scores[selected_object_num])

    return(selected_object_num, proba_of_selected_object)


def get_box_size(obj_params):
    
    x_center, y_center, width, height = obj_params[0:4]
    x_min = (x_center - width/2)
    y_min = (y_center - height/2)
    width = width
    height = height
    
    return(x_min, y_min, width, height)


def transform_boxes_to_original_size(boxes, multipliers):
    
    boxes_original = [ (b * multipliers).astype(int) for b in boxes]
    
    return(boxes_original)
    

def get_parameters_of_boxes(boxes, labels, label_nums, proba, b_colors, b_filter=None):
    
    results = []
    
    if b_filter is not None and len(b_filter)>0:
        b_filter = b_filter.flatten()
        
        boxes = [boxes[f] for f in b_filter]
        label_nums = [label_nums[f] for f in b_filter]
        proba = [proba[f] for f in b_filter]
        
        labels = [labels[lab_num] for lab_num in label_nums]
        colors = [b_colors[lab_num] for lab_num in label_nums]
    
    
        for num in range(len(labels)):
            
            single_box = {'Label':labels[num],
                          'Coords':boxes[num],
                          'Proba':proba[num],
                          'Color':colors[num]}
            
            results.append(single_box) 
        
    return(results)


def transform_box_coords(box):
    
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    
    return(box)
    


def get_box_coords(boxes):
    
    boxes = [transform_box_coords(b) for b in boxes]
    
    return(boxes)
    

def show_image_with_detected_objects(image, detected_objects):
    
    if len(detected_objects) > 0:
        
        for det_obj in detected_objects:
        
            box_text = "{}: {:.4f}".format(det_obj['Label'], det_obj['Proba'])
            
            x_min = int(det_obj['Coords'][0])
            y_min = int(det_obj['Coords'][1])
            x_max = x_min + int(det_obj['Coords'][2])
            y_max = y_min + int(det_obj['Coords'][3])
            
            cv2.rectangle(img=image, 
                          pt1=(x_min, y_min), 
                          pt2=(x_max, y_max), 
                          color=det_obj['Color'].tolist(), thickness=2)
        
            cv2.putText(img=image, 
                        text=box_text, 
                        org=(x_min, y_min-5), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale=0.7, 
                        color=det_obj['Color'].tolist(), thickness=2)
        
    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Detections')
    
    pass
