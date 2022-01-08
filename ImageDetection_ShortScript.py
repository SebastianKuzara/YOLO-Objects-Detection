#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:23:37 2022.

@author: seb
"""

from utilities.AggregateFunction import detect_objects_on_image

detect_objects_on_image(image_file="./images/village.jpeg",
                        names_file="./config/coco.names",
                        config_file="./config/yolov3.cfg", 
                        weights_file="./config/yolov3.weights")
