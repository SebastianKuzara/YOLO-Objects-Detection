#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  27 22:20:07 2021.

@author: seb

Utilites to load and save files.
"""

import os

def load_text_file(file_name):
    """
    Load text file as a list.

    Parameters
    ----------
    file_name : str
        Path to file.

    Returns
    -------
    List.

    """
    if not os.path.isfile(file_name):
        raise ValueError("File does not exist: {}".format(file_name))
        
    with open(file_name) as f:
        result = [line.strip() for line in f]
    
    return(result)



