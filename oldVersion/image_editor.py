#################################################################
# FILE : image_editor.py
# WRITER : Noa Aharoni , noalee25 , 212392823
# EXERCISE : intro2cs ex6 2024
# DESCRIPTION: A program that edits photos
# STUDENTS I DISCUSSED THE EXERCISE WITH: -
#################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from ex6_helper import *
from typing import Optional
from math import *
import sys

##############################################################################
#                                  Functions                                 #
##############################################################################


def separate_channels(image: ColoredImage) -> List[SingleChannelImage]:
    image_lst = []
    for k in range(len(image[0][0])):
        temp_channle = []
        for i in range(len(image)):
            temp_line = []
            for j in range(len(image[i])):
                temp_line.append(image[i][j][k])
            temp_channle.append(temp_line)
        image_lst.append(temp_channle)
    return image_lst


def combine_channels(channels: List[SingleChannelImage]) -> ColoredImage:
    combined_l = []
    for i in range(len(channels[0])):
        line = []
        for j in range(len(channels[0][0])):
            temp_l = []
            for k in range(len(channels)):
                temp_l.append(channels[k][i][j])
            line.append(temp_l)
        combined_l.append(line)
    return combined_l


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    g_image = []
    for i in range(len(colored_image)):
        temp_line = []
        for j in range(len(colored_image[i])):
            num = colored_image[i][j][0]*0.299+colored_image[i][j][1]*0.587+colored_image[i][j][2]*0.114
            temp_line.append(round(num))
        g_image.append(temp_line)
    return g_image


def blur_kernel(size: int) -> Kernel:
    k_l = []
    for i in range(size):
        row_l = []
        for j in range(size):
            row_l.append(1/(size**2))
        k_l.append(row_l)
    return k_l


#a function that creates the matrix of the surrounding cells
def create_small_mat(image, cell, size):
    x = size//2
    rows = range(cell[0]-x, cell[0]+x+1)
    cols = range(cell[1]-x, cell[1]+x+1)
    mat = []
    for i in range(size):
        temp_line = []
        for j in range(size):
            if rows[i]<0 or rows[i]>=len(image) or cols[j]<0 or cols[j]>=len(image[0]):
                temp_line.append(image[cell[0]][cell[1]])
            else:
                temp_line.append(image[rows[i]][cols[j]])
        mat.append(temp_line)
    return mat


#a function that calculates the value of a cell
def calculate_kernel_value(mat, kernel):
    sum = 0
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            sum += mat[i][j]*kernel[i][j]
    sum = round(sum)
    if sum < 0:
        return 0
    elif sum > 255:
        return 255
    return sum




def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    new_image = []
    for i in range(len(image)):
        temp_l = []
        for j in range(len(image[i])):
            temp_m = create_small_mat(image, (i,j), len(kernel))
            temp_l.append(calculate_kernel_value(temp_m, kernel))
        new_image.append(temp_l)
    return new_image


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    dx = x - floor(x)
    dy = y - floor(y)
    a = image[floor(y)][floor(x)]
    b = image[ceil(y)][floor(x)]
    c = image[floor(y)][ceil(x)]
    d = image[ceil(y)][ceil(x)]
    return round(a*(1-dx)*(1-dy)+b*dy*(1-dx)+c*dx*(1-dy)+d*dx*dy)


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    new_img = []
    for i in range(new_height):
        temp_line = []
        for j in range(new_width):
            x = (j/(new_width-1))*(len(image[0])-1)
            y = (i/(new_height-1))*(len(image)-1)
            temp_line.append(bilinear_interpolation(image, y, x))
        new_img.append(temp_line)
    return new_img


#a function that rotate only right
def rotate_R(image):
    rotated_img = []
    for i in range(len(image[0])):
        temp_line = []
        for j in range(len(image)-1, -1, -1):
            temp_line.append(image[j][i])
        rotated_img.append(temp_line)
    return rotated_img

#a function that rotate only left
def rotate_L(image):
    rotated_img = []
    for i in range(len(image[0])-1, -1, -1):
        temp_line = []
        for j in range(len(image)):
            temp_line.append(image[j][i])
        rotated_img.append(temp_line)
    return rotated_img


def rotate_90(image: Image, direction: str) -> Image:
    if type(image[0][0]) == int:
        if direction == 'R':
            return rotate_R(image)
        return rotate_L(image)
    else:
        separated_img = separate_channels(image)
        new_img = []
        for i in separated_img:
            if direction == 'R':
                new_img.append(rotate_R(i))
            else:
                new_img.append(rotate_L(i))
        return combine_channels(new_img)


#a function that calculate the average value of a small matrix
def average_value(img, size, cell):
    temp_mat = create_small_mat(img, cell, size)
    sum_value = 0
    for i in temp_mat:
        sum_value += sum(i)
    return (sum_value-(img[cell[0]][cell[1]]))/((size*size)-1)


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int, c: float) -> SingleChannelImage:
    blur_k = blur_kernel(blur_size)
    blurred_img = apply_kernel(image, blur_k)
    new_img = []
    for i in range(len(image)):
        temp_line = []
        for j in range(len(image[0])):
            threshold = average_value(blurred_img, block_size, (i,j))-c
            if threshold > blurred_img[i][j]:
                temp_line.append(0)
            else:
                temp_line.append(255)
        new_img.append(temp_line)
    return new_img


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    new_img = []
    for i in range(len(image)):
        temp_line = []
        for j in range(len(image[0])):
            temp_line.append(round(floor(image[i][j]*(N/256))*(255/(N-1))))
        new_img.append(temp_line)
    return new_img


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    separated_img = separate_channels(image)
    q_img = []
    for i in separated_img:
        q_img.append(quantize(i, N))
    return combine_channels(q_img)

#checks if the image is already in grey scale
def check_img_greyscale(img):
    return type(img[0][0]) == int

#turns image to grey scale if its RGB
def option_1(image):
    if not check_img_greyscale(image):
        image = RGB2grayscale(image)
    else:
        print("already grey scale image")
    return image


#checks if the kernel is valid
def check_kernel_value(kernel_size):
    return round(float(kernel_size)) == int(kernel_size) and int(kernel_size) > 0 and int(kernel_size)%2 == 1

#blurring the image
def option_2(image):
    kernel_size = input("enter kernel size:")
    if check_kernel_value(kernel_size):
        if check_img_greyscale(image):
            image = apply_kernel(image, blur_kernel(int(kernel_size)))
        else:

            image = separate_channels(image)
            image = [apply_kernel(i, blur_kernel(int(kernel_size))) for i in image]
            image = combine_channels(image)
    else:
        print("invalid kernel value")
    return image

#resize image
def option_3(image):
    new_sizes = input("enter new height and width").split(",")
    if len(new_sizes) == 2 and round(float(new_sizes[0])) == int(new_sizes[0]) and round(float(new_sizes[1])) == int(new_sizes[1]) and int(new_sizes[0]) > 1 and int(new_sizes[1]) > 1:
        new_sizes = [int(i) for i in new_sizes]
        if check_img_greyscale(image):
            image = resize(image, new_sizes[0], new_sizes[1])
        else:
            image = separate_channels(image)
            image = [resize(i, new_sizes[0], new_sizes[1]) for i in image]
            image = combine_channels(image)
    else:
        print("invalit input")
    return image

#rotate image
def option_4(image):
    direction = input("enter wanted rotation direction")
    if direction == 'R' or direction == 'L':
        image = rotate_90(image, direction)
    else:
        print("invalid direction")
    return image

#create the outline of the image
def option_5(image):
    edges_input = input("enter blur size, block size and c").split(",")
    if len(edges_input) == 3 and round(float(edges_input[0])) == int(edges_input[0]) and round(float(edges_input[1])) == int(edges_input[1]) and int(edges_input[0]) > 0 and int(edges_input[0])%2 == 1 and int(edges_input[1]) > 0 and int(edges_input[1])%2 == 1 and int(edges_input[2]) > 0:
        edges_input = [int(i) for i in edges_input]
        if check_img_greyscale(image):
            image = get_edges(image, edges_input[0], edges_input[1], edges_input[2])
        else:
            image = RGB2grayscale(image)
            image = get_edges(image, edges_input[0], edges_input[1], edges_input[2])
    else:
        print("invalid edges values")
    return image

#quantize the image
def option_6(image):
    num_of_shades = input("enter number of wanted shades")
    if round(float(num_of_shades)) == int(num_of_shades) and int(num_of_shades) > 1:
        if check_img_greyscale(image):
            image = quantize(image, int(num_of_shades))
        else:
            image = quantize_colored_image(image, int(num_of_shades))
    else:
        print("invalid quantize value")
    return image


if __name__ == '__main__':
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
        image = load_image(image_path)
        user_input = input("enter number: 1-grey scale 2-blur image 3-resize 4-rotate 5-edges 6-quantize 7-show image 8-exit")
        continue_program = False if user_input == '8' else True
        while continue_program:
            if user_input == '1':
                image = option_1(image)
            if user_input == '2':
                image = option_2(image)
            if user_input == '3':
                image = option_3(image)
            if user_input == '4':
                image = option_4(image)
            if user_input == '5':
                image = option_5(image)
            if user_input == '6':
                image = option_6(image)
            if user_input == '7':
                show_image(image)
            if user_input == '8':
                break
            user_input = input("enter number: 1-grey scale 2-blur image 3-resize 4-rotate 5-edges 6-quantize 7-show image 8-exit")
        path_to_save = input("enter path to save new image:")
        save_image(image, path_to_save)
    else:
        print("invalid input")


