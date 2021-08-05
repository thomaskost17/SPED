'''
  File: reconstruction.py
 
  Author: Thomas Kost
  
  Date: 02 August 2021
  
  @breif using sparse sensing to create reconstruction
 '''

import cvxpy as cp
import numpy as np
from utility import *

'''
General Math:
x = true image
P = general fourier/wavelet basis
y = measured image
C = measurement matrix
s = sparse vector

Formulation:
x = P*s
y = C*x = C*P*s

find s satisfying:

min. ||s||_1 st. ||y-C*P*s||< E ... for some E
'''

def optimize()->np.array:
    pass

def bayesian_measurement_matrix(probaility: float)->np.array:
    pass

def gaussian_measurement_matrix()->np.array:
    pass

def create_measurement_matrix(type:str)->np.array:
    pass

def create_fourier_basis()->np.array:
    pass

def create_discrete_cos_basis()->np.array:
    pass

def create_basis(type:str = "Fourier")->np.array:
    pass

if __name__ == "__main__":
    pass
