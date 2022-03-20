# HDR reconstruction using Robertson's method
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import random
import cv2
import os

def weight(z):
	z_min, z_max = 0, 255
	z_mid = (z_min + z_min) // 2

	return z_max - z if z > z_mid else z - z_min

def get_hdr_by_Robertson(imgs, times, iteration = 3, save_path = ''):
    '''
    imgs : p x h x w x c => p images
    times : p x 1 => p images' shutter times
    '''

    # init g
    g_BGR = [np.array([i / 128 for i in range(256)]) for _ in range(3)]
    Z_BGR = np.array(imgs).transpose(3, 0, 1, 2)
    E_BGR = [np.zeros((imgs[0].shape[0], imgs[0].shape[1]), dtype = 'float32') for _ in range(3)]
    
    for i in range(iteration):
        print('{}-th iteration'.format(i+1))
        E_BGR = [optimize_E(g, Z_BGR[idx], times) for idx, g in enumerate(g_BGR)]
        g_BGR = [optimize_g(g, Z_BGR[idx], E_BGR[idx], times) for idx, g in enumerate(g_BGR)]
    
        get_response_curve(np.log(g_BGR), i+1, save_path)
        get_radiance_map(E_BGR, i+1, save_path)

    E = np.array(E_BGR).transpose(1, 2, 0)
    return E

def optimize_E(g, Z, T):
    '''
    g : 256 x 1
    Z : p x h x w

    Ei = Σ w(z)g(z)ΔTj / Σ w(z)(ΔTj ** 2)
    '''
    p, h, w = Z.shape

    wgt_sum = np.zeros((h, w), dtype = 'float32')
    wt2_sum = np.zeros((h, w), dtype = 'float32')

    for j in range(p):
        Z_flatten = Z[j].flatten()
        
        # Σ w(z)g(z)ΔTj 
        weights = np.array([weight(z) for z in Z_flatten]).reshape(h, w)
        wgt_sum += weights * g[Z_flatten.astype('int')].reshape(h, w) * T[j]

        # Σ w(z)(ΔTj ** 2)
        wt2_sum += weights * (T[j] ** 2)

    wt2_sum[wt2_sum == 0] = 1.e-10

    return wgt_sum / wt2_sum
    
def optimize_g(g, Z, E, T):
    '''
    g(m) = (ΣEi * ΔTj) / |Em| 
    '''
    
    for m in range(256):
        j, i1, i2 = np.where(Z == m) #reshape(Z.shape[0], -1) 
        
        if len(i1) == 0:
            continue

        # ΣEi * ΔTj
        ET_sum = np.sum(E[i1, i2] * T[j])
        
        g[m] = ET_sum / len(j)

    # normalize g(128) = 1
    g /= g[128]

    return g

def get_response_curve(g_BGR, iteration, save_path):

	fig = plt.figure(figsize=(15, 5))
	bgr = ['Blue', 'Green', 'Red']

	for c in range(len(g_BGR)):
		plt.subplot(1, 3, c + 1)
		plt.plot(g_BGR[c], np.arange(256), color = bgr[c])
		plt.title(bgr[c])
		plt.xlabel('E: Log Exposure')
		plt.ylabel('Z: Pixel Value')

	plt.tight_layout()
	# plt.show()
	fig.savefig(save_path + str(iteration) + '_response_curve.png', bbox_inches = 'tight')

def get_radiance_map(E_BGR, iteration, save_path):
    
    E_BGR = np.array(E_BGR).transpose(1, 2, 0)
    # E_BGR[E_BGR == 0] = 1.e-10
    lnE_BGR = np.log(E_BGR)
    h, w, c = lnE_BGR.shape

    fig = plt.figure(figsize=(15, 5))
    bgr = ['Blue', 'Green', 'Red']

    for cc in range(c):
        plt.subplot(1, 3, cc + 1)
        img = plt.imshow(lnE_BGR[:, :, cc], cmap = 'jet')
        plt.colorbar(img, fraction = 0.046 * h / w, pad = 0.05, format = ticker.FuncFormatter(lambda x, _:'%.2f' % np.exp(x)))
        plt.title(bgr[cc])
        plt.axis('off')

    plt.tight_layout()
    # plt.show()

    fig.savefig(save_path + str(iteration) + '_radiance_Robertson.png', bbox_inches = 'tight')
    cv2.imwrite(save_path + str(iteration) + '_radiance_Robertson.hdr', E_BGR.astype('float32'))

'''
from main import read_imgs_and_times

if __name__ == '__main__':

    ## add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './data/', help = 'Path to the directory that contains series of images.')
    parser.add_argument('--series_of_images', type = str, default = 'window', help = 'The folder of a series of images that contains images and shutter time file.')
    parser.add_argument('--shutter_time_filename', type = str, default = 'shutter_times.txt', help = 'The name of the file where shutter time information is stored.')
    args = parser.parse_args()

    path = os.path.join(args.data_path, args.series_of_images, "")
    filename = args.shutter_time_filename

    ## read images
    imgs, times = read_imgs_and_times(path, filename)

    hdr = get_hdr_by_Robertson(imgs, times)

    ldrDrago = cv2.createTonemapDrago(1.0, 0.7).process(hdr) * 255 * 3
    cv2.imwrite("tonemapping_Drago.png", ldrDrago)
'''



