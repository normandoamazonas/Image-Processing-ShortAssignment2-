'''Normando de Campos Amazonas Filho, 11561949
Image Enhancement, SCC0251_Turma01_1Sem_2020_ET
Short Assignment 2: Image Restoration
https://github.com/normandoamazonas/ShortAssignment2'''

import numpy as np
import imageio
from scipy.fftpack import fftn, ifftn, fftshift

'''
case_b1_n.png
3
0.6
0.001
'''
#reading the inputs
filename = str(input()).rstrip()
input_img = imageio.imread(filename)
k = int(input())#3
sigma =float(input())# 0.6
gamma =  float(input()) #0.001

#laplacian kernel
laplacianKernel = np.array([[0,-1,0],
                     [-1, 4,-1],
                     [0,-1,0]])


#code provided by the runcodes (gaussian_filter)
def gaussian_filter(k=3, sigma=1.0):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma))
    return filt/np.sum(filt)

#Scaling function to normalize the images
def Scaling(I,max_input):
    Min = np.min(I)
    Max = np.max(I)
    #n,m =I.shape
    im= (I-Min)*(max_input/(Max-Min))
    return im#.astype(int)

#Gaussian filter
filtro = gaussian_filter(k,sigma)

# computing the number of padding on one side
a = int(input_img.shape[0]//2 - filtro.shape[0]//2)
h_pad = np.pad(filtro, (a,a-1), 'constant', constant_values=(0))

# computing the Fourier transforms for the input image and gaussian filter
F = fftn(input_img)
H = fftn(h_pad)

#Computing G
G = np.multiply(F,H)
#G = Scaling(G,np.max(input_img))

#padding the laplacian filter, code based on padding gaussian filter
b = int(input_img.shape[0]//2 - laplacianKernel.shape[0]//2)
p_pad = np.pad(laplacianKernel,(b,b-1), 'constant', constant_values=(0))
P =fftn(p_pad) # fft of laplacian filter

#computing the F_hat
F_hat = (np.conj(H)/(np.abs(H)**2+gamma*np.abs(P)**2))*G

#restoration
f_hat = ifftn(F_hat).real

#Finally, computing the scaling
f_hat = Scaling(f_hat,np.max(input_img))

#print the standart deviation
print("%.1f"%np.std(f_hat))
