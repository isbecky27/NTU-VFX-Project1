# image alignment by using MTB algorithm
from cmath import inf
import numpy as np
import argparse
import cv2
import os

def BGR2GRAY(img):
    '''
    gray = (54 * red + 183 * green + 19 * blue) / 256
    '''
    # img_gray = (54 * img[:,:,2] + 183 * img[:,:,1] + 19 * img[:,:,0]) / 256
    # img_gray = np.array(img_gray, dtype='uint8')
    # img_gray = np.array([[(54*y[2] + 183*y[1] + 19*y[0]) / 256 for y in x] for x in img], dtype='uint8')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def exclusion_bitmap(img):
    '''
    get exclusion bitmap
    '''
    median = np.median(img.flatten())
    # eb = cv2.inRange(img, median - 4, median + 4) # wrong
    eb = np.array(img)
    eb[np.where(eb <= median - 10)] = 0
    eb[np.where(eb > median + 10)] = 255
    return eb

def threshold_bitmap(img):
    '''
    get threshold bitmap
    '''
    # median = statistics.median(img.flatten()) # wrong
    median = np.median(img.flatten())
    _, img_binary = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)
    return img_binary

def cal_diff(flag, tar):
    '''
    calculate the difference between two images
    '''
    eb1 = exclusion_bitmap(flag)
    eb2 = exclusion_bitmap(tar)
    # int type for bitwise operation
    flag = flag.astype(int) 
    tar = tar.astype(int)
    diff = np.bitwise_xor(flag, tar)
    diff = np.bitwise_and(diff, eb1)
    diff = np.bitwise_and(diff, eb2)
    return np.count_nonzero(diff)

def matrix(dx, dy):
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return M

def shift_img(flag, tar, dx, dy):
    h, w = flag.shape
    min_err = inf
    ret_dx, ret_dy = dx, dy
    for i in range(-1, 2):
        for j in range(-1, 2):
            M = matrix(dx + i, dy + j)
            tar_shift = cv2.warpAffine(tar, M, (w, h))
            diff = cal_diff(flag, tar_shift)
            if diff < min_err:
                min_err = diff
                ret_dx, ret_dy = dx + i, dy + j
    return ret_dx, ret_dy

def align(flag, tar, layer):
    if layer == 0:
        dx, dy = shift_img(flag, tar, 0, 0)
    else:
        h, w = flag.shape
        flag_shrink = cv2.resize(flag, (w//2, h//2))
        tar_shrink = cv2.resize(tar, (w//2, h//2))
        dx, dy = align(flag_shrink, tar_shrink, layer-1)
        dx *= 2
        dy *= 2
        dx, dy = shift_img(flag, tar, dx, dy)
    return dx, dy

def image_alignment(imgs):
    imgs_gray = [BGR2GRAY(img) for img in imgs]
    # generate threshold bitmap
    tb = np.array([threshold_bitmap(img) for img in imgs_gray])
    # flag (base) image, choose the middle one
    flagid = len(tb) // 2
    flag = tb[flagid]
    h, w = flag.shape
    
    layer = 2
    
    ret_imgs = []

    for i in range(0, len(tb)):
        final_dx, final_dy = align(flag, tb[i], layer)
        M = matrix(final_dx, final_dy)
        new_img = cv2.warpAffine(imgs[i], M, (w, h))
        ret_imgs.append(new_img)
    
    return ret_imgs

'''
from main import read_imgs_and_times

if __name__ == '__main__':

    ## add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './data/', help = 'Path to the directory that contains series of images.')
    parser.add_argument('--series_of_images', type = str, default = 'poster', help = 'The folder of a series of images that contains images and shutter time file.')
    parser.add_argument('--shutter_time_filename', type = str, default = 'shutter_times.txt', help = 'The name of the file where shutter time information is stored.')
    args = parser.parse_args()

    path = os.path.join(args.data_path, args.series_of_images, "")
    series = args.series_of_images
    filename = args.shutter_time_filename

    ## read images
    imgs, _ = read_imgs_and_times(path, filename)

    ## image alignment
    image_alignment(imgs)
'''
    