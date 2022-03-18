import numpy as np
import argparse
import cv2
import os

def global_operator(Lw, delta = 0.000001, a = 0.5, Lwhite = 200):
    '''
    Lw_bar = exp(1/N * Σlog(δ + Lw))
    Lm(x, y) = (a / Lw_bar) * Lw(x, y)  
    Ld(x, y) = [Lm(x, y) * (1 + Lm(x, y) / (Lwhite(x, y) ** 2))] / (1 + Lm(x, y)) 
    '''
    
    Lw_bar = np.exp(np.mean(np.log(delta + Lw)))
    Lm = (a / Lw_bar) * Lw
    Ld = (Lm * (1 + Lm / (Lwhite ** 2))) / (1 + Lm) 

    img = np.clip(np.array(Ld * 255), 0, 255).astype('uint8')

    return img

def local_operator(Lw, delta = 0.000001, a = 0.5):
    '''
    Ld = Lm / (1 + Lblur_smax)
    '''
    ## calculate Lm
    Lw_bar = np.exp(np.mean(np.log(delta + Lw)))
    Lm = (a / Lw_bar) * Lw
    
    ## calculate Lblur_smax
    Lblur_smax = np.zeros(Lw.shape)
    for i in range(3):
        Lblur_smax[:, :, i] = gaussian_blur(Lm[:, :, i]) 
    
    ## calculate Ld
    Ld = Lm / (1 + Lblur_smax)

    img = np.clip(np.array(Ld * 255), 0, 255).astype('uint8')

    return img

def gaussian_blur(Lm, s = 35, phi = 8, a = 0.5, e = 0.01):
    '''
    Lblur_s = Lm x Gs (convolution)
    Vs = (Lblur_s - Lblur_s+1) / ((2 ** Φ) * a / (s ** 2) + Lblur_s)
    Smax = |Vsmax| < ε
    '''

    h, w = Lm.shape
    Lblur_smax = np.zeros((h, w))

    Lblur_s_list = [cv2.GaussianBlur(Lm, (i, i), 0) for i in range(1, s+1, 2)]
    Lblur_s_list.insert(0, Lm) # add origin Lm
    Lblur_s_list = np.array(Lblur_s_list).transpose(1, 2, 0)

    Vs_list = np.array([np.abs(Lblur_s_list[:, :, idx] - Lblur_s_list[:, :, idx + 1]) / ((2 ** phi) * a / (ss ** 2) + Lblur_s_list[:, :, idx]) for idx, ss in enumerate(range(1, s, 2))])
    Vs_list_max = np.argmax(Vs_list > e, axis = 0)
    Vs_list_max -= 1 
    Vs_list_max[Vs_list_max < 0] = 0
    # print(Vs_list_max)

    i, j = np.ogrid[:h, :w]
    Lblur_smax = Lblur_s_list[i, j, Vs_list_max]
    
    ## too slow
    # for i in range(h):
    #     for j in range(w):
    #         Vs_list = [np.abs(Lblur_s_list[i, j, idx] - Lblur_s_list[i, j, idx + 1]) / ((2 ** phi) * a / (ss ** 2) + Lblur_s_list[i, j, idx]) for idx, ss in enumerate(range(1, s, 2))]
    #         Vs_idx = np.argmin(Vs_list)
    #         Lblur_smax[i, j] = Lblur_s_list[i, j, Vs_idx]
    
    return Lblur_smax

def gamma_correction(img, r):
	'''
		r > 1 : reduce brightness
		r < 1 : enhance brightness
	'''
	img = (img.clip(min = 0) / 255) ** r * 255

	return img

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

'''
from main import read_imgs_and_times

if __name__ == '__main__':

    ## add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './data/', help = 'Path to the directory that contains series of images.')
    parser.add_argument('--series_of_images', type = str, default = 'desk', help = 'The folder of a series of images that contains images and shutter time file.')
    parser.add_argument('--shutter_time_filename', type = str, default = 'shutter_times.txt', help = 'The name of the file where shutter time information is stored.')
    args = parser.parse_args()

    path = os.path.join(args.data_path, args.series_of_images, "")
    filename = args.shutter_time_filename

    ## read images
    imgs, times = read_imgs_and_times(path, filename)

    ## HDR using opencv package
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(imgs, times)
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(imgs, times, response)
    # hdr = cv2.imread('hdr.hdr',-1)
    # hdr = cv2.imread('./result/desk/radiance_debecvec.hdr', -1)
    
    ## tone mapping
    # tone_mapping_using_package(hdr, 'Drago')
    # tone_mapping_using_package(hdr, 'Mantiuk')
    # tone_mapping_using_package(hdr, 'Reinhard')
    # tone_mapping_using_package(hdr)
    ldr = global_operator(hdr)
    cv2.imwrite('desk_global.png', ldr)

    ldr = local_operator(hdr)
    cv2.imwrite('desk_local.png', ldr)
    
    cv2.imshow('Tone Mapping', ldr)
    cv2.waitKey(0)
'''