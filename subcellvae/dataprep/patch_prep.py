import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from skimage import transform
import czifile
import math

def image_padding(input_img, pad_size,value):
    output_img = np.zeros([input_img.shape[0]+pad_size*2,input_img.shape[1]+pad_size*2]) + value
    output_img[pad_size:input_img.shape[0]+pad_size,pad_size:input_img.shape[1]+pad_size] = input_img
    return output_img


def rotate_coor(x_i,y_i,x_c,y_c,rotate_angle):
 
    rotate_angle = rotate_angle*np.pi/180
 
    x_o = (x_i-x_c)*math.cos(rotate_angle) - (2*y_c-y_i-y_c)*math.sin(rotate_angle) +x_c
    y_o = -(x_i-x_c)*math.sin(rotate_angle) - (2*y_c-y_i-y_c)*math.cos(rotate_angle) +(2*y_c-y_c)

    return([x_o,y_o])


def list_czi_files(image_folder):
    return sorted([
        x for x in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, x)) and ('.czi' in x)
    ])

def load_and_pad(image_folder, cell_mask_folder, filename, major_ch,
                 pad_size=64, img_pad_val=None):
    """Load image + mask, normalize image, and pad both."""
    img = czifile.imread(os.path.join(image_folder, filename)).squeeze()[major_ch, :, :].astype(float) / (255 * 255)
    seg = tifffile.imread(os.path.join(cell_mask_folder, "cell_mask_" + filename + ".tif")).squeeze().astype(float)

    if img_pad_val is None:
        img_pad_val = float(np.mean(img))

    img = image_padding(img, pad_size, img_pad_val)
    seg = image_padding(seg, pad_size, 0)
    return img, seg

def init_debug_fig(train_img, train_seg, dpi=256):
    fig, ax = plt.subplots(1, 2, figsize=(15.6, 7.8), dpi=dpi, facecolor='w', edgecolor='k')
    ax[0].imshow(train_img, cmap=plt.cm.gray, vmax=1, vmin=0)
    ax[1].imshow(train_seg, cmap=plt.cm.gray, vmax=1, vmin=0)
    return fig, ax

def compute_grid(train_img_shape, patch_size, offset_frac_x=0.0, offset_frac_y=0.0):
    """Return x_num, y_num and starting x_0, y_0 for centered grid."""
    H, W = train_img_shape
    x_num = int(np.floor(W / patch_size))
    y_num = int(np.floor(H / patch_size))

    x_0 = int((W - x_num * patch_size) / 2 + offset_frac_x * patch_size)
    y_0 = int((H - y_num * patch_size) / 2 + offset_frac_y * patch_size)
    return x_num, y_num, x_0, y_0

def iter_grid_centers(x_num, y_num, x_0, y_0, patch_size):
    """Yield (x_i, y_i, x_c, y_c) grid centers."""
    for x_i in range(x_num):
        for y_i in range(y_num):  # NOTE: y_num (not x_num)
            y_c = int(y_0 + (y_i - 0.5) * patch_size)
            x_c = int(x_0 + (x_i - 0.5) * patch_size)
            yield x_i, y_i, x_c, y_c

def extract_big_patch(train_img, train_seg, x_c, y_c, double_ps):
    """Extract a big square patch centered at (x_c,y_c) with side 2*double_ps."""
    y_left = y_c - double_ps
    x_left = x_c - double_ps
    y_right = y_c + double_ps
    x_right = x_c + double_ps

    if y_left < 0 or x_left < 0 or y_right >= train_img.shape[0] or x_right >= train_img.shape[1]:
        return None

    patch_img = train_img[y_left:y_right, x_left:x_right]
    patch_seg = train_seg[y_left:y_right, x_left:x_right]
    return patch_img, patch_seg, x_left, y_left

def apply_optional_translation(rand_trans_flag, max_shift_px=0):
    """Return (rand_tx, rand_ty). With flag=0 -> (0,0)."""
    if rand_trans_flag:
        rand_tx = random.randint(-max_shift_px, max_shift_px)
        rand_ty = random.randint(-max_shift_px, max_shift_px)
    else:
        rand_tx, rand_ty = 0, 0
    return rand_tx, rand_ty

def first_crop_from_big(patch_img, patch_seg, patch_size, double_ps, rand_tx, rand_ty):
    """Crop a big_crop (still larger than final) from the big patch, with optional translation."""
    cx_left_1  = patch_size - rand_tx
    cx_right_1 = double_ps + patch_size - rand_tx
    cy_up_1    = patch_size - rand_ty
    cy_down_1  = double_ps + patch_size - rand_ty

    big_crop_img = patch_img[cy_up_1:cy_down_1, cx_left_1:cx_right_1]
    big_crop_seg = patch_seg[cy_up_1:cy_down_1, cx_left_1:cx_right_1]

    # corners of this crop in *original full-image coordinates* (still before rotation)
    first_crop_x = np.array([cx_left_1, cx_left_1, cx_right_1, cx_right_1, cx_left_1])
    first_crop_y = np.array([cy_up_1,  cy_down_1, cy_down_1,  cy_up_1,   cy_up_1])

    return big_crop_img, big_crop_seg, (cx_left_1, cy_up_1), first_crop_x, first_crop_y

def apply_optional_rotation(big_crop_img, big_crop_seg, rand_rota_flag, max_angle_deg=0.0):
    """Rotate both image and seg by same angle. With flag=0 -> no-op."""
    if rand_rota_flag:
        rand_angle = (random.random() * 2 - 1) * max_angle_deg
    else:
        rand_angle = 0.0

    if rand_angle == 0:
        return big_crop_img, big_crop_seg, rand_angle

    rot_img = transform.rotate(big_crop_img, rand_angle, resize=False, mode='constant', cval=0, clip=True)
    rot_seg = transform.rotate(big_crop_seg, rand_angle, resize=False, mode='constant', cval=0, clip=True)
    return rot_img, rot_seg, rand_angle

def center_crop(rot_img, rot_seg, patch_size, half_ps):
    """Take final patch_size crop from center region of rotated big crop."""
    cx_left_2  = half_ps
    cx_right_2 = patch_size + half_ps
    cy_up_2    = half_ps
    cy_down_2  = patch_size + half_ps
    crop_img = rot_img[cy_up_2:cy_down_2, cx_left_2:cx_right_2]
    crop_seg = rot_seg[cy_up_2:cy_down_2, cx_left_2:cx_right_2]
    return crop_img, crop_seg, (cx_left_2, cy_up_2)

def compute_final_polygon_in_full_image(patch_size, rand_angle,
                                       cx_left_2, cy_up_2,
                                       x_left, y_left,
                                       cx_left_1, cy_up_1):
    """Compute the polygon (in full-image coords) of the final crop after inverse-rotation."""
    X = np.array([cx_left_2, cx_left_2, cx_left_2 + patch_size, cx_left_2 + patch_size, cx_left_2])
    Y = np.array([cy_up_2,   cy_up_2 + patch_size, cy_up_2 + patch_size, cy_up_2,         cy_up_2])

    # rotate_coor expected signature: rotate_coor(X, Y, cx, cy, angle)
    X_inv, Y_inv = rotate_coor(X, Y, patch_size, patch_size, -rand_angle)

    # move from rotated-crop coords back to full-image coords
    X_full = X_inv + x_left + cx_left_1
    Y_full = Y_inv + y_left + cy_up_1
    return X_full, Y_full

def save_patch(movie_partitioned_data_dir, crop_img_filename, crop_patch_img):
    tifffile.imwrite(
        os.path.join(movie_partitioned_data_dir, crop_img_filename),
        crop_patch_img.astype(np.float32),
        imagej=True,
        metadata={'axes': 'YX'}
    )

def make_record_row(image_folder, filename, filenameID, x_c, y_c,
                    rand_angle, rand_tx, rand_ty,
                    X_full, Y_full,
                    movie_partitioned_data_dir, crop_img_filename,
                    movie_plot_dir, plot_filename):
    return pd.Series(
        [
            image_folder, filename, filenameID, x_c, y_c, rand_angle, rand_tx, rand_ty,
            X_full[0], X_full[1], X_full[2], X_full[3],
            Y_full[0], Y_full[1], Y_full[2], Y_full[3],
            movie_partitioned_data_dir, crop_img_filename, movie_plot_dir, plot_filename
        ],
        index=[
            'image_folder','filename','filenameID','x_c','y_c','rand_angle','rand_tx','rand_ty',
            'x_corner1','x_corner2','x_corner3','x_corner4',
            'y_corner1','y_corner2','y_corner3','y_corner4',
            'movie_partitioned_data_dir','crop_img_filename','movie_plot_dir','plot_filename'
        ]
    )