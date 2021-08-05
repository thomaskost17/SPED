'''
  File: compress.py
 
  Author: Thomas Kost
  
  Date: 02 August 2021
  
  @breif performing simple image compression using a fourier basis
 '''
import numpy as np
from utility import *
import matplotlib.pyplot as plt
import cv2 as cv

def compress(image : np.array, percentage: float, debug: bool  = False)->np.array:
    
    max_pixel_value = 256

    # Normalize by max value of an image pixel
    image = image/max_pixel_value

    # Take FFT of each pixel type
    rFimg = np.fft.fft2(image[:,:,0])
    gFimg = np.fft.fft2(image[:,:,1])
    bFimg = np.fft.fft2(image[:,:,2])

    # Plot Spectrums when debug requested
    if debug:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(np.log10(abs(np.fft.fftshift(rFimg))))
        plt.axis("off")
        plt.title("Red FFT")
        plt.subplot(1,3,2)
        plt.imshow(np.log10(abs(np.fft.fftshift(gFimg))))
        plt.axis("off")
        plt.title("Green FFT")
        plt.subplot(1,3,3)
        plt.imshow(np.log10(abs(np.fft.fftshift(bFimg))))
        plt.axis("off")
        plt.title("Blue FFT")

    # Sort FFT by magnitude
    rFimg_flat = np.sort(abs(rFimg.flatten()))
    gFimg_flat = np.sort(abs(gFimg.flatten()))
    bFimg_flat = np.sort(abs(bFimg.flatten()))

    # Find magnitude threshold for retaining Data
    max_val_red = rFimg_flat[int(len(rFimg_flat)*(1.0-percentage))]
    max_val_green = rFimg_flat[int(len(gFimg_flat)*(1.0-percentage))]
    max_val_blue = rFimg_flat[int(len(rFimg_flat)*(1.0-percentage))]

    # Create and apply data mask to keep top X percent of values
    r_mask = abs(rFimg) > max_val_red
    g_mask = abs(gFimg) > max_val_green
    b_mask = abs(bFimg) > max_val_blue
    rFimg = rFimg*r_mask
    gFimg = gFimg*g_mask
    bFimg = bFimg*b_mask

    # Perform IFFT 
    r_img = np.fft.ifft2(rFimg)
    g_img = np.fft.ifft2(gFimg)
    b_img = np.fft.ifft2(bFimg)

    # Recombine colors de-normalize and cast as integer
    comp_img = (np.dstack((abs(r_img),abs(g_img), abs(b_img)))*max_pixel_value).astype(np.uint8)
    return comp_img

if __name__ == "__main__":
    # Read in RGB Image
    file = "../fixtures/jelly.jpg"
    img = read(file)

    #compress image
    compressed = compress(img, 0.05, True)

    # Plot Original image
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    # Plot Compressed Image
    plt.subplot(1,2,2)
    plt.imshow(compressed)
    plt.title("Compressed Image")
    plt.axis("off")

    # Display plot
    plt.show()
