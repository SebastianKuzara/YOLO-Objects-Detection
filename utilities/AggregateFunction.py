#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:11:42 2022.

@author: seb
"""


import cv2
from cv2.dnn import blobFromImage, readNetFromDarknet
from utilities.ImageManagement import load_image
from utilities.ImageManagement import generate_box_colors, get_image_size
from utilities.ImageManagement import transform_image_size_to_multipliers
from utilities.ImageManagement import transform_predictions_into_boxes
from utilities.ImageManagement import transform_boxes_to_original_size
from utilities.ImageManagement import get_parameters_of_boxes
from utilities.ImageManagement import show_image_with_detected_objects
from utilities.FileManagement import load_text_file
from utilities.NetworkManagement import predict_image_detection



def detect_objects_on_image(image_file, names_file, config_file, weights_file,
                            min_probability=0.5,
                            score_threshold=0.5, nms_threshold=0.3):
    
    image = load_image(image_file)
    
    img_original_size = get_image_size(image)
    img_multipliers = transform_image_size_to_multipliers(img_original_size)
    
    blob = blobFromImage(image, scalefactor=1./255., size=(512, 512),
                         swapRB=True, crop=False)
    
    all_labels = load_text_file(names_file)
    box_colors = generate_box_colors(n=int(len(all_labels)))
    
    network = readNetFromDarknet(config_file, weights_file)

    predictions = predict_image_detection(model=network, image=blob)
        
    boxes, confidences, label_nums = transform_predictions_into_boxes(predictions, all_labels)
            
    boxes = transform_boxes_to_original_size(boxes, img_multipliers)
    
    results = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, 
                               score_threshold=0.5, nms_threshold=0.3)
    
    print(results)
    
    box_params = get_parameters_of_boxes(boxes=boxes, 
                                         labels=all_labels, 
                                         label_nums=label_nums,
                                         proba=confidences, 
                                         b_colors=box_colors, 
                                         b_filter=results)
    
    show_image_with_detected_objects(image, box_params)
    
    pass

    