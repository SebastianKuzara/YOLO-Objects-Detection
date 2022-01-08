#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  22 22:50:46 2021.

@author: seb
"""

import cv2
from cv2.dnn import blobFromImage, readNetFromDarknet
from utilities.ImageManagement import load_image
from utilities.ImageManagement import show_image, show_blob
from utilities.ImageManagement import generate_box_colors, get_image_size
from utilities.ImageManagement import transform_image_size_to_multipliers
from utilities.ImageManagement import transform_predictions_into_boxes
from utilities.ImageManagement import transform_boxes_to_original_size
from utilities.ImageManagement import get_parameters_of_boxes
from utilities.ImageManagement import show_image_with_detected_objects
from utilities.FileManagement import load_text_file
from utilities.NetworkManagement import predict_image_detection

image = load_image("./images/wroclaw_street.jpg")
# image = load_image("./images/cat.jpg")
# show_image(image, label='Original image')

img_original_size = get_image_size(image)
img_multipliers = transform_image_size_to_multipliers(img_original_size)

blob = blobFromImage(image, scalefactor=1./255., size=(512, 512),
                     swapRB=True, crop=False)

# show_blob(blob, label='Blob image')

all_labels = load_text_file('./config/coco.names')
box_colors = generate_box_colors(n=int(len(all_labels)))


network = readNetFromDarknet('./config/yolov3.cfg', './config/yolov3.weights')

predictions = predict_image_detection(model=network, image=blob)

boxes, confidences, label_nums = transform_predictions_into_boxes(predictions, all_labels)

boxes_original = [ (b * img_multipliers).astype(int) for b in boxes]


boxes = transform_boxes_to_original_size(boxes, img_multipliers)

results = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, 
                           score_threshold=0.5, nms_threshold=0.3)

box_params = get_parameters_of_boxes(boxes=boxes, 
                                     labels=all_labels, 
                                     label_nums=label_nums,
                                     proba=confidences, 
                                     b_colors=box_colors, 
                                     b_filter=results)



show_image_with_detected_objects(image, box_params)


