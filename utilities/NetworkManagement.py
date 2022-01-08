#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:57:15 2021.

@author: seb
"""

# import cv2

def get_output_layer_names(net):
    """
    Get output layer names of neural net.

    Parameters
    ----------
    net : cv2.dnn_Net
        Product of cv2.dnn module functions.

    Returns
    -------
    Names of output layers.

    """
    output_layers_num = net.getUnconnectedOutLayers()[:,0] - 1
    layer_names = net.getLayerNames()
    
    output_layer_names = [layer_names[num] for num in output_layers_num]
    
    return(output_layer_names)


def predict_image_detection(model, image):
    """
    Predict object detecion on image using the model.

    Parameters
    ----------
    model : cv2.dnn_Net
        Product of cv2.dnn module functions.
    image : numpy.ndarray
        Image as numpy array. It should be a blob image.

    Returns
    -------
    A list with parameters of recognized object on an image.

    """
    model.setInput(image)
    
    output_layers = get_output_layer_names(model)
    
    predictions = model.forward(output_layers)
    
    return(predictions)