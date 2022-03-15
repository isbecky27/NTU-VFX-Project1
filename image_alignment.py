# image alignment by using MTB algorithm
from main import read_imgs_and_log_deltaT
import numpy as np
import argparse
import random
import cv2
import os
import statistics

def BGR2GRAY(img):
    '''
    gray = (54 * red + 183 * green + 19 * blue) / 256
    '''
    img_gray = (54. * img[:,:,2] + 183 * img[:,:,1] + 19 * img[:,:,0]) / 256

    return img_gray

def threshold_bitmap(img):
    median = statistics.median(img.flatten())
    _, img_binary = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)

    return img_binary

def image_alignment(imgs):
    imgs_gray = [BGR2GRAY(img) for img in imgs]
    
    # threshold_bitmap(imgs_gray[4])
    '''
    TODO
    '''


if __name__ == '__main__':

    ## add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './data/', help = 'Path to the directory that contains series of images.')
    parser.add_argument('--series_of_images', type = str, default = 'memorial', help = 'The folder of a series of images that contains images and shutter time file.')
    parser.add_argument('--shutter_time_filename', type = str, default = 'shutter_times.txt', help = 'The name of the file where shutter time information is stored.')
    args = parser.parse_args()

    path = os.path.join(args.data_path, args.series_of_images, "")
    filename = args.shutter_time_filename

    ## read images
    imgs, _ = read_imgs_and_log_deltaT(path, filename)

    ## image alignment
    image_alignment(imgs)
    

    