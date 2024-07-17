#Import required libraries
import warnings

import numpy as np
from numba import njit, jit
import argparse
import cv2

#suppress all warnings
warnings.filterwarnings('ignore')

#Function to calculate the energy of the image
#For more about how to calculate the energy of the image, please refer to the following link:
    # https://en.wikipedia.org/wiki/Seam_carving
    # https://www.baeldung.com/cs/gradient-orientation-magnitude#:~:text=Gradient%20magnitude%20refers%20to%20the,directions.
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

######################## SEAM HELPERS FUNCTIONS ########################

# This function is designed to find and return the minimum energy seam in an image, which is a crucial part of the seam carving algorithm used for content-aware image resizing.
# Input Parameters:
    # image: A 3D numpy array representing the image (height x width x number of color channels).
# Output:
    # seam_index: A list of column indices representing the minimum energy seam from top to bottom.
    # boolmask: A boolean mask of the same size as the image (height h by width w). The mask indicates which pixels are part of the seam (False) and which are not (True).
@jit
def get_minimum_seam(image):

    height, width, RGB = image.shape
    # 1. Energy Map Calculation
    energy_map = calculate_energy_map(image)
    # 2. Initialization: backtrack is a 2D array used to keep track of the path of the minimum energy seam.
    backtrack = np.zeros_like(energy_map, dtype=np.int64)
    # 3. Dynamic Programming: Calculate the minimum energy seam using dynamic programming.
        #For each pixel (i, j) from the second row to the last row, the code computes the minimum energy path to that pixel from the row above.
        # np.argmin finds the index of the minimum value in the specified range of the previous row.
        # min_energy is the minimum energy value from the previous row's range.
        # The backtrack array records the column index from which the minimum energy came.
        # The energy map M is updated by adding the minimum energy to the current pixel's energy.

    for i in range(1, height):
        j = 0
        idx = np.argmin(energy_map[i - 1, j:j + 2])
        min_energy = energy_map[i - 1, idx + j]
        backtrack[i, j] = idx + j
        energy_map[i, j] += min_energy
        for j in range(1, width):
            idx = np.argmin(energy_map[i - 1, j - 1:j + 2])
            min_energy = energy_map[i - 1, idx + j - 1]
            backtrack[i, j] = idx + j - 1

            energy_map[i, j] += min_energy

    # 4. Trace Back the Minimum Energy Seam:
        # boolmask is initialized to be a boolean mask of the same dimensions as the image, initially all set to True.
        # The column index of the minimum energy in the last row is found.
        # Starting from this column index, the code traces back to the first row using the backtrack array,
        # marking the seam pixels as False in the boolmask and storing their column indices in seam_index.
        # Finally, seam_index is reversed to start from the top row to the bottom row.
    boolmask = np.ones((height, width), dtype=np.bool)
    col = np.argmin(energy_map[-1])
    seam_index = []

    row = height
    while row > 0:
        row = row - 1
        boolmask[row, col] = False
        seam_index.append(col)
        col = backtrack[row, col]

    seam_index = seam_index[::-1]
    seam_index = np.array(seam_index)

    # Minh họa
    #Để minh họa hàm get_minimum_seam, hãy xem xét một ví dụ đơn giản với một hình ảnh nhỏ và các bước tính toán cụ thể.

        # Giả sử
        # Hãy xem xét một hình ảnh nhỏ với chiều cao h = 4 và chiều rộng w = 4. Bản đồ năng lượng ban đầu M của hình ảnh này được tính như sau:
        #
        # 4 2 3 1
        # 3 4 1 2
        # 2 3 2 4
        # 1 2 3 4
        #
        # Bước 1: Khởi tạo mảng backtrack
        # Khởi tạo backtrack là một mảng zero cùng kích thước với M:
        #
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        #
        # Bước 2: Cập nhật bản đồ năng lượng M và backtrack
        # Hàng 1 (i = 1)
        # Cột 0 (j = 0):
        #
        # Xem xét các pixel phía trên (M[0, 0:2] = [4, 2]):
        # idx = np.argmin([4, 2]) = 1
        # min_energy = M[0, 1] = 2
        # backtrack[1, 0] = 1
        # M[1, 0] += min_energy = 3 + 2 = 5
        # Cột 1 (j = 1):
        #
        # Xem xét các pixel phía trên (M[0, 0:3] = [4, 2, 3]):
        # idx = np.argmin([4, 2, 3]) = 1
        # min_energy = M[0, 1] = 2
        # backtrack[1, 1] = 1
        # M[1, 1] += min_energy = 4 + 2 = 6
        # Cột 2 (j = 2):
        #
        # Xem xét các pixel phía trên (M[0, 1:4] = [2, 3, 1]):
        # idx = np.argmin([2, 3, 1]) = 2
        # min_energy = M[0, 3] = 1
        # backtrack[1, 2] = 3
        # M[1, 2] += min_energy = 1 + 1 = 2
        # Cột 3 (j = 3):
        #
        # Xem xét các pixel phía trên (M[0, 2:4] = [3, 1]):
        # idx = np.argmin([3, 1]) = 1
        # min_energy = M[0, 3] = 1
        # backtrack[1, 3] = 3
        # M[1, 3] += min_energy = 2 + 1 = 3
        # Cập nhật M và backtrack sau hàng 1:
        #
        # 4 2 3 1
        # 5 6 2 3
        # 2 3 2 4
        # 1 2 3 4
        #
        # 0 0 0 0
        # 1 1 3 3
        # 0 0 0 0
        # 0 0 0 0
        #
        # [...]
        # Cập nhật M và backtrack sau hàng 3:
        #
        # 4 2 3 1
        # 5 6 2 3
        # 7 5 4 6
        # 6 6 7 8
        #
        # 0 0 0 0
        # 1 1 3 3
        # 0 2 2 2
        # 1 2 2 2
        #
        # Bước 3: Truy ngược lại đường seam có năng lượng thấp nhất
        # Khởi tạo mặt nạ boolean:
        #
        # Mặt nạ boolean khởi tạo tất cả các giá trị là True.
        #
        # Tìm chỉ số cột có năng lượng thấp nhất trong hàng cuối cùng:
        #
        # Hàng 3: row = 3, col = 0
        #
        # boolmask[3, 0] = False
        # seam_idx = [0]
        # col = backtrack[3, 0] = 1
        # Hàng 2: row = 2, col = 1
        #
        # boolmask[2, 1] = False
        # seam_idx = [0, 1]
        # col = backtrack[2, 1] = 2
        # Hàng 1: row = 1, col = 2
        #
        # boolmask[1, 2] = False
        # seam_idx = [0, 1, 2]
        # col = backtrack[1, 2] = 3
        # Hàng 0: row = 0, col = 3
        #
        # boolmask[0, 3] = False
        # seam_idx = [0, 1, 2, 3]
        # col = backtrack[0, 3] = 0 (mặc dù không còn sử dụng tiếp)
        # Đảo ngược và chuyển đổi danh sách các chỉ số cột:
        # seam_idx: [3, 2, 1, 0]
        #
        # boolmask:
        # 𝑇𝑟𝑢𝑒 𝑇𝑟𝑢𝑒 𝑇𝑟𝑢𝑒 𝐹𝑎𝑙𝑠𝑒
        # 𝑇𝑟𝑢𝑒 𝑇𝑟𝑢𝑒 𝐹𝑎𝑙𝑠𝑒 𝑇𝑟𝑢𝑒
        # 𝑇𝑟𝑢𝑒 𝐹𝑎𝑙𝑠𝑒 𝑇𝑟𝑢𝑒 𝑇𝑟𝑢𝑒
        # 𝐹𝑎𝑙𝑠𝑒 𝑇𝑟𝑢𝑒 𝑇𝑟𝑢𝑒 𝑇𝑟𝑢𝑒
        #
        # Kết luận
        # Đường seam có năng lượng thấp nhất đi qua các cột [3, 2, 1, 0] từ hàng đầu đến hàng cuối cùng. Mặt nạ boolean boolmask chỉ ra các pixel thuộc đường seam là False.


    return seam_index, boolmask

#This function inserts a new pixel at a specified position on a row of the image.
# Input Parameters:
    # output: 3D array (height x width x number of color channels) is the resulting image after adding solid lines.
    # im: The 3D array is the original image before adding solid lines.
    # row: Index of the row we are processing.
    # col: Index of the column into which we will insert the new pixel.
    # channel_num: Index of the current color channel (e.g. R, G, B).
    # p: New pixel value will be inserted into the image.
    # flag: Flag to determine where to insert a new pixel, which can be before or after the current pixel.
@jit(nopython=True)
def add_seam_util(output, im, row, col, channel_num, p, flag):

    # If flag is 0 (insert new pixel after current pixel):
    if flag == 0:
        # Keep the current pixel value at position(row, col, channel_num).
        output[row, col, channel_num] = im[row, col, channel_num]
        # Insert new pixel p at the position immediately after the current pixel.
        output[row, col + 1, channel_num] = p
        # Preserve all pixels before col in the row.
        output[row, :col, channel_num] = im[row, :col, channel_num]
        # Move all pixels after col + 1 to the right by one position.
        output[row, col + 2:, channel_num] = im[row, col + 1:, channel_num]

    # If flag is 1 (insert new pixel before current pixel):
    else:
        # Insert new pixel p at the position immediately before the current pixel.
        output[row, col, channel_num] = p
        # Move the current pixel one position to the right.
        output[row, col + 1, channel_num] = im[row, col, channel_num]
        # Move all pixels after col + 1 to the right by one position.
        output[row, col + 2:, channel_num] = im[row, col + 1:, channel_num]


# The add_seam function is used to add a vertical seam to an image.
#Input parameters:
    #image: A 3D numpy array representing the original image (height x width x number of color channels).
    #seam_index: A 1D array containing the column indices of the seamless line to add, with one index per row.
#Result:
    #output: A 3D numpy array representing the resulting image after adding the solid line.
@jit(nopython=True)
def add_seam(image, seam_index):
    # 1. Variable initialization:

    ## height, width, RGB: Get the height, width and number of color channels of the original image.
    height, width, RGB = image.shape
    ## Increase the width by one unit to accommodate more solid lines.
    width = width + 1
    ## dim = 3: Assume the number of color channels is 3 (eg RGB).
    #dim = 3
    ## size_tuple = (h, w, dim): Creates a tuple representing the size of the resulting image.
    size_tuple = (height, width, RGB)
    ## output = np.zeros(size_tuple): Initialize the resulting image with zero values.
    output = np.zeros(size_tuple)

    # 2. Process each row:

    ## Start from the last row and move up (row = height - 1 to row = 0).
    row = height - 1
    while row >= 0:
        ## For each row, get the column index where the seam will be added (col = seam_index[row]).
        col = seam_index[row]
        # 3. Add solid lines for each color channel:

        ## For each color channel (channel_num from 0 to RGB - 1):


        channel_num = 0
        while channel_num < RGB:
            ## If the solid line is not in the first column (col != 0):
            if col != 0:
                ## Calculate the new pixel value p as the average of the current pixel and the pixel to its left
                p = np.average([image[row, col - 1, channel_num], image[row, col, channel_num]])
                # Call add_seam_util to insert the new pixel p after the current pixel.
                add_seam_util(output, image, row, col, channel_num, p, 0)
            ## If the solid line is in the first column (col == 0):
            else:
                #Calculate the new pixel value p as the average of the current pixel and the pixel to its right
                p = np.average([image[row, col, channel_num], image[row, col + 1, channel_num]])
                #Call add_seam_util to insert the new pixel p before the current pixel.
                add_seam_util(output, image, row, col, channel_num, p, 1)
            # Increment channel_num to process the next color channel.
            channel_num += 1
        # Move to the next row (row = row - 1).
        row = row - 1

    return output