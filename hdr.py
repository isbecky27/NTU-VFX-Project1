from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import random
import cv2
import os

def select_sample_points(imgs, n):
	'''
	Output : 
		Z_BGR : list, 3 x p x n => Z_BGR[c][j][i] i-th pixel value in j-th image c-th color BGR
	'''
	
	h, w, c = imgs[0].shape
	index = [i for i in range(h * w)]
	random_idx = np.array(random.sample(index, n))

	i, j = random_idx // w, random_idx % w

	Z_BGR = [[imgs[p][i, j, cc] for p in range(len(imgs))] for cc in range(c)]

	# for j, img in enumerate(imgs):
	# 	img_flat = img.flatten()
	# 	pts = np.array([value for idx, value in enumerate(img_flat) if idx in random_idx])
	# 	Z[:, j] = pts

	return Z_BGR

def get_hdr_by_Paul_Debevec(imgs, Z_BGR, lnT, l, save_path = ''):
	
	g_BGR = [get_g_function(np.array(Z).T, lnT, l) for Z in Z_BGR]
	lnE_BGR = construct_radiance_map(imgs, g_BGR, lnT)

	get_radiance_map(lnE_BGR, save_path)
	get_response_curve(g_BGR, save_path)

	return np.exp(lnE_BGR)

def get_g_function(Z, B, l = 50):
	'''
	Input :
		Z : n x p => p images for each n pixels
		B : p x 1 => the log delta for p images j
		l : constant => lambda, the amount of smoothness	
	Output :
		g => the log exposure corresponding to pixel value z
		lnE => the log film irradiance 

	Ax = b :
		A : (n x p + 255) x (256 + n)
		x : (256 + n) x 1
		b : (n x p + 255) x 1
	'''

	n, p = np.array(Z).shape

	## construct A matrix
	A = np.zeros((n * p + 255, 256 + n), dtype = 'float32')
	
	k = 0
	for i in range(n):
		for j in range(p):	
			w = weight(Z[i][j])
			A[i * p + j][int(Z[i][j])] = w
			A[i * p + j][256 + i] = -w
			k += 1

	A[k][127] = 1 # g(128) = 1
	k += 1

	for i in range(1, 255): # pixel value 1 ~ 254
		w = weight(i)
		A[k, i - 1: i + 2] = np.array([1, -2, 1]) * l * w
		k += 1

	## construct b matrix
	b = np.zeros((n * p + 255, 1), dtype = 'float32')

	for i in range(n):
		for j in range(p):
			w = weight(Z[i][j])
			b[i * p + j] = w * B[j]

	## get least-square solution x
	A_inv = np.linalg.pinv(A)
	x = A_inv.dot(b)

	## get g function and lnE
	g, lnE = x[:256], x[256:]

	return g

def weight(z):
	z_min, z_max = 0, 255
	z_mid = (z_min + z_min) // 2

	return z_max - z if z > z_mid else z - z_min

def construct_radiance_map(imgs, g_BGR, B):
	'''
	lnE = Σ w(z)(g(z)-lnT)/ Σ w(z)
	'''

	imgs = np.array(imgs)
	p, h, w, c = imgs.shape

	lnE_BGR = np.zeros((h, w, c), dtype = 'float32')

	for cc in range(c):
		w_sum = np.zeros((h, w), dtype = 'float32')
		lnE_sum = np.zeros((h, w), dtype = 'float32')

		for pp in range(p):

			img = imgs[pp][:, :, cc].flatten()
	
			# w * (g(z)-lnT)
			weights = np.array([weight(z) for z in img]).reshape(h, w)
			w_lnE = weights * np.array(g_BGR[cc][img] - B[pp]).reshape(h, w)

			# Σ w * (g(z)-lnT)
			lnE_sum += w_lnE

			# Σ w
			w_sum += weights

		# Σ w * (g(z)-lnT) / Σ w
		w_sum[w_sum == 0] = 0.000000001
		lnE_BGR[:, :, cc] = (lnE_sum / w_sum)

	return lnE_BGR
	
def get_response_curve(g_BGR, save_path):

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
	fig.savefig(save_path + 'response_curve.png', bbox_inches = 'tight')

def get_radiance_map(lnE_BGR, save_path):

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

	fig.savefig(save_path + 'radiance_Debevec.png', bbox_inches = 'tight')
	cv2.imwrite(save_path + 'radiance_Debevec.hdr', np.exp(lnE_BGR).astype('float32'))

'''
from main import read_imgs_and_log_deltaT

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
	imgs, lnT = read_imgs_and_log_deltaT(path, filename)
	times = np.exp(lnT)

	## HDR using opencv package
	calibrate = cv2.createCalibrateDebevec()
	response = calibrate.process(imgs, times)
	merge_debevec = cv2.createMergeDebevec()
	# hdr = merge_debevec.process(imgs, times, response)
	# hdr = np.log(hdr)

	display_response_curve(response, '')
'''

