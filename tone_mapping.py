from hdr import read_imgs_and_log_deltaT
import numpy as np
import argparse
import cv2
import os

def global_operator():
    '''
    TODO
    '''

    
def local_operator():
    '''
    TODO
    '''

def tone_mapping_using_package(hdr, method = '', save_path = ''):
    
    if method == 'Drago':
        '''
        Parameters:
            float gamma = 1.0f,
            float saturation = 1.0f,
            float bias = 0.85f
        '''
        ldrDrago = cv2.createTonemapDrago(1.0, 0.7).process(hdr) * 255 * 3
        cv2.imwrite(save_path + "tonemapping_Drago.png", ldrDrago)
    
    elif method == 'Mantiuk':
        '''
        Parameters:
            float gamma = 1.0f,
            float scale = 0.7f,
            float saturation = 1.0f 
        '''
        ldrMantiuk = cv2.createTonemapMantiuk(2, 0.85, 0.85).process(hdr) * 255 * 3
        cv2.imwrite(save_path + "tonemapping_Mantiuk.png", ldrMantiuk)
    
    elif method == 'Reinhard':
        '''
        Parameters:
            float gamma = 1.0f,
            float intensity = 0.0f,
            float light_adapt = 1.0f,
            float color_adapt = 0.0f
        '''
        ldrReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0).process(hdr) * 255 
        cv2.imwrite(save_path + "tonemapping_Reinhard.png", ldrReinhard)

    else:
        '''
        Parameters:
            float gamma = 1.0f
        '''
        ldr = cv2.createTonemap(3.5).process(hdr) * 255 * 3
        cv2.imwrite(save_path + "tonemapping_Tonemap.png", ldr)



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
    imgs, lnT = read_imgs_and_log_deltaT(path, filename)
    times = np.exp(lnT)

    ## HDR using opencv package
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(imgs, times)
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(imgs, times, response)

    tone_mapping_using_package(hdr, 'Drago')
    # tone_mapping_using_package(hdr, 'Mantiuk')
    # tone_mapping_using_package(hdr, 'Reinhard')
    # tone_mapping_using_package(hdr)
    