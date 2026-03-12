import numpy as np

def image_padding(input_img, pad_size,value):
    output_img = np.zeros([input_img.shape[0]+pad_size*2,input_img.shape[1]+pad_size*2]) + value
    output_img[pad_size:input_img.shape[0]+pad_size,pad_size:input_img.shape[1]+pad_size] = input_img
    return output_img