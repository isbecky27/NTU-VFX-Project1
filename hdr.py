from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from fractions import Fraction
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

	display_radiance_map(lnE_BGR, save_path)
	display_response_curve(g_BGR, save_path)

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

#-------- Need to be modified - BEGIN --------#
def fmt(x, pos):
    return '%.3f' % np.exp(x)

def display_response_curve(g_BGR, save_path):
    bgr_string = ['blue', 'green', 'red']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c in range(3):
        ax = axes[c]
        ax.plot(g_BGR[c], np.arange(256), c=bgr_string[c])
        ax.set_title(bgr_string[c])
        ax.set_xlabel('E: Log Exposure')
        ax.set_ylabel('Z: Pixel Value')
        ax.grid(linestyle=':', linewidth=1)
    fig.savefig(save_path + 'response_curve.png', bbox_inches='tight', dpi=256)

def display_radiance_map(lnE_BGR, save_path):

	plt.clf()
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	bgr_string = ['blue', 'green', 'red']

	for cc in range(3):
		ax = axes[cc]
		im = ax.imshow(lnE_BGR[:, :, cc], cmap='jet')
		ax.set_title(bgr_string[cc])
		ax.set_axis_off()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))

	radiance_bgr = np.exp(lnE_BGR)
	print(np.max(radiance_bgr) / np.min(radiance_bgr))
    
	fig.savefig(save_path + 'radiance_debevec.png', bbox_inches='tight', dpi=256)
	cv2.imwrite(save_path + 'radiance_debecvec.hdr', radiance_bgr.astype(np.float32))

#-------- Need to be modified - END --------#
	

	