'''
  File: display.py
 
  Author: Thomas Kost
  
  Date: 02 August 2021
  
  @breif display funciton to make showing an image easier for the application
 '''

from PIL import Image as im
import numpy as np
from numpy.core.defchararray import asarray
from numpy.matrixlib.defmatrix import matrix
import cv2 as cv

def display(array : np.array, mode : str = None, title : str = None):
    img = im.fromarray(array, mode)
    if title:
        img.save(title +'.jpg')
    img.show()
    return

def read(file_path: str)->np.array:
    img = cv.imread(file_path,cv.IMREAD_UNCHANGED)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return np.asarray(img)