'''
  File: reconstruction.py
 
  Author: Thomas Kost
  
  Date: 02 August 2021
  
  @breif using sparse sensing to create reconstruction
 '''

import cvxpy as cp
import numpy as np
from scipy.fftpack import fft, dct, idct
from scipy.linalg import dft as DFT_mat
from utility import *
import cv2 as cv
import matplotlib.pyplot as plt
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

def optimize(y:np.array, C:np.array, epsilon: float, basis: str = "fourier", debug: bool = False)->np.array:
    # Calculate constants
    sz = C.shape
    s_sz = sz[1]
    P = create_basis(s_sz,basis)
    # setup LP
    s = cp.Variable(s_sz)
    objective = cp.Minimize(cp.norm(s,1))
    constraints = [cp.norm(y-C@P@s,1) <= epsilon]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, max_iters=200, verbose=debug)
    return s.value

def bayesian_measurement_matrix(n : int, m : int, probaility: float)->np.array:
    return np.random.rand(n,m) < probaility;

def gaussian_measurement_matrix(n : int, m : int)->np.array:
    pass

def create_measurement_matrix(type:str, n : int, m : int, probability : float = 0.5)->np.array:
    if type == "gaussian":
        return gaussian_measurement_matrix(n,m)
    elif type == "bayesian":
        return bayesian_measurement_matrix(n,m, probability)
    else:
        raise ValueError("Unsupported measurement matrix")

def create_fourier_basis(dimension:int)->np.array:
    return DFT_mat(dimension)

def create_discrete_cos_basis(dimension:int)->np.array:
    return dct(np.eye(dimension), axis=0)

def create_basis(dimension:int, type:str = "fourier")->np.array:
    if type == "fourier":
        return create_fourier_basis(dimension)
    elif type == "dct":
        return create_discrete_cos_basis(dimension)
    else:
        raise ValueError("Unsupported basis type")

if __name__ == "__main__":
    
    # # Read in RGB Image
    # file = "../fixtures/mustache.jpg"
    # img = read(file)
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # sz = img.shape
    # full_dim = sz[0]*sz[1]
    # x = img.flatten()

    # Create Signal
    n = 4096
    t = np.linspace(0,1,n)
    f1 = 97
    f2 = 777
    x = np.cos(2*np.pi*f1*t) +np.cos(2*np.pi*f2*t)
    X = np.fft.fft(x)

    # Conduct Random sampling
    dim_y = int(n*0.05)
    measurement_probability = 1.0/n
    C = create_measurement_matrix("bayesian", dim_y,n, measurement_probability)
    y = C@x

    # Find sparse Vector
    s = optimize(y, C, 0, "dct", True)

    # Reconstruct original
    P = create_basis(len(s),"dct")
    # x_prime = P@s
    x_prime = idct(s)
    
    # Display Results
    fig, axes = plt.subplots(2,2)
    #fig.set_size_inches(10.5,8.5)
    fig.tight_layout(h_pad=3, w_pad =3)
    
    axes[0,0].set_title("x(t): True value")
    axes[0,0].set_ylabel("x")
    axes[0,0].set_xlabel("t (s)")
    axes[0,0].plot(t,x, "red")
    axes[0,0].set_xlim((0.24, 0.3))

    axes[0,1].set_title("X(w)")
    axes[0,1].set_ylabel("X")
    axes[0,1].set_xlabel("w (Hz)")
    axes[0,1].plot(np.linspace(-n/2, n/2,n),abs(np.fft.fftshift(X)), "black")
    axes[0,1].set_xlim((0, n/2))

    axes[1,0].set_title("~x(t): Recreated value")
    axes[1,0].set_ylabel("~x")
    axes[1,0].set_xlabel("t (s)")
    axes[1,0].plot(t,x_prime, "red")
    axes[1,0].set_xlim((0.24, 0.3))

    axes[1,1].set_title("~X(w)")
    axes[1,1].set_ylabel("~X")
    axes[1,1].set_xlabel("w (Hz)")
    axes[1,1].plot(np.linspace(-n/2, n/2,n),abs(np.fft.fftshift(np.fft.fft(x_prime))), "black")
    axes[1,1].set_xlim((0, n/2))

    plt.show()



