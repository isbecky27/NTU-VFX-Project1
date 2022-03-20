# image alignment by using MTB algorithm
from cmath import inf
from main import read_imgs_and_times
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
    eb[np.where(eb <= median - 4)] = 0
    eb[np.where(eb > median + 4)] = 255
    return eb

def threshold_bitmap(img):
    '''
    get threshold bitmap
    '''
    # median = statistics.median(img.flatten()) # wrong
    median = np.median(img.flatten())
    _, img_binary = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)
    return img_binary


def cal_diff(base_tb, base_eb, tar_tb, tar_eb):
    '''
    calculate the difference between two images
    '''
   
    # int type for bitwise operation
    base_tb = base_tb.astype(int) 
    tar_tb = tar_tb.astype(int)
    base_eb = base_eb.astype(int) 
    tar_eb = tar_eb.astype(int)
    diff = np.bitwise_xor(base_tb, tar_tb)
    diff = np.bitwise_and(diff, base_eb)
    # diff = np.bitwise_and(diff, tar_eb)
    return np.count_nonzero(diff)

def matrix(dx, dy):
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return M


def shift_img(base_tb, base_eb, tar_tb, tar_eb, dx, dy):
    h, w = base_tb.shape
    min_err = inf
    ret_dx, ret_dy = dx, dy
    for i in range(-1, 2):
        for j in range(-1, 2):
            M = matrix(dx + i, dy + j)
            tar_tb_shift, tar_eb_shift = cv2.warpAffine(tar_tb, M, (w, h)), cv2.warpAffine(tar_eb, M, (w, h))
            diff = cal_diff(base_tb, base_eb, tar_tb_shift, tar_eb_shift)
            if diff < min_err:
                min_err = diff
                ret_dx, ret_dy = dx + i, dy + j
    return ret_dx, ret_dy


def align(base_tb, base_eb, tar_tb, tar_eb, layer):
    if layer == 0:
        dx, dy = shift_img(base_tb, base_eb, tar_tb, tar_eb, 0, 0)
    else:
        h, w = base_tb.shape
        base_tb_shrink, base_eb_shrink = cv2.resize(base_tb, (w//2, h//2)), cv2.resize(base_eb, (w//2, h//2))
        tar_tb_shrink, tar_eb_shrink = cv2.resize(tar_tb, (w//2, h//2)), cv2.resize(tar_eb, (w//2, h//2))
        dx, dy = align(base_tb_shrink, base_eb_shrink, tar_tb_shrink, tar_eb_shrink, layer-1)
        dx *= 2
        dy *= 2
        dx, dy = shift_img(base_tb, base_eb, tar_tb, tar_eb, dx, dy)
    return dx, dy
    

def image_alignment(imgs):
    imgs_gray = [BGR2GRAY(img) for img in imgs]
    # generate threshold bitmap
    tb = np.array([threshold_bitmap(img) for img in imgs_gray])
    eb = np.array([exclusion_bitmap(img) for img in imgs_gray])
    # base_tb (base) image, choose the middle one
    base_id = len(tb) // 2
    base_tb = tb[base_id]
    base_eb = eb[base_id]
    h, w = base_tb.shape
    
    layer = 4
    
    ret_imgs = []

    # for i in range(len(tb)):
    #     cv2.imwrite("./myresult/tb_%d.PNG" % i, tb[i])
    #     cv2.imwrite("./myresult/eb_%d.PNG" % i, eb[i])


    for i in range(0, len(tb)):
        if i == base_id:
            ret_imgs.append(imgs[i])
            continue
        final_dx, final_dy = align(base_tb, base_eb, tb[i], eb[i], layer)
        M = matrix(final_dx, final_dy)
        new_img = cv2.warpAffine(imgs[i], M, (w, h))
        ret_imgs.append(new_img)
    
    return ret_imgs


'''
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
    imgs, _ = read_imgs_and_log_deltaT(path, filename)

    ## image alignment
    imgs = image_alignment(imgs)
    for i in range(len(imgs)):
        cv2.imwrite("./myresult/align_%d.PNG" % i, imgs[i])
'''    
    

    