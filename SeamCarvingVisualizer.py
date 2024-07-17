#Import required libraries
import warnings

import numpy as np
from numba import njit
import argparse
import cv2

#suppress all warnings
warnings.filterwarnings('ignore')

#Function to calculate the energy of the image
##For more about how to calculate the energy of the image,
##please refer to the following link: https://en.wikipedia.org/wiki/Seam_carving | https://www.baeldung.com/cs/gradient-orientation-magnitude#:~:text=Gradient%20magnitude%20refers%20to%20the,directions.
@njit
def calculate_energy_map(image):
    #edge filter
    edge_filter = np.array([-1, 0, 1])

    #gradient in the x-dỉrection
    x_gradient = convolve(image, edge_filter, axis=1)

    #gradient in the y-direction
    y_gradient = convolve(image, edge_filter, axis=0)

    #calculate the energy map of the image
    x_gradient = x_gradient ** 2
    y_gradient = y_gradient ** 2
    energy_map = np.sqrt(np.sum(x_gradient, axis=2) + np.sum(y_gradient, axis=2))

    return energy_map

#Function to calculate convolution of a matrix with a filter
@njit
def convolve(image, edge_filter, axis):
    h, w, z = image.shape
    result = np.zeros_like(image, dtype=np.float64)
    if axis == 1:
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if (j - 1) < 0:
                        result[i, j, k] = (
                                image[i, j, k] * edge_filter[1] +
                                image[i, (j + 1), k] * edge_filter[2]
                        )
                    elif (j + 1) >= w:
                        result[i, j, k] = (
                                image[i, (j - 1), k] * edge_filter[0] +
                                image[i, j, k] * edge_filter[1]
                        )
                    else:
                        result[i, j, k] = (
                                image[i, (j - 1) % w, k] * edge_filter[0] +
                                image[i, j, k] * edge_filter[1] +
                                image[i, (j + 1) % w, k] * edge_filter[2]
                        )
    elif axis == 0:
        for j in range(w):
            for i in range(h):
                for k in range(z):
                    if (i - 1) < 0:
                        result[i, j, k] = (
                                image[i, j, k] * edge_filter[1] +
                                image[(i + 1), j, k] * edge_filter[2]
                        )
                    elif (i + 1) >= h:
                        result[i, j, k] = (
                                image[(i - 1), j, k] * edge_filter[0] +
                                image[i, j, k] * edge_filter[1]
                        )
                    else:
                        result[i, j, k] = (
                                image[(i - 1) % h, j, k] * edge_filter[0] +
                                image[i, j, k] * edge_filter[1] +
                                image[(i + 1) % h, j, k] * edge_filter[2]
                        )
    return result

