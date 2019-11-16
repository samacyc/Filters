from PIL import Image
import sys 
from math import exp,pi ,sqrt
from scipy import misc
from skimage import data,io
from skimage import img_as_float
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.viewer import ImageViewer
import scipy.misc



def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col= image.shape[:2]
      mean = float(input("Entrer la moyenne "))
      var = float(input("Entrer la var"))
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = float(input("saisir s_vs_p coeff"))
      amount = float(int("amount"))
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy


def function(tabpixel,sigma,x,y) : 
   p0 =somme = 0
   coeff = exp(-1/(2*sigma*sigma))/((2*pi*sigma*sigma))
   helper = exp(1/(2*sigma*sigma))
   filtre = [[coeff*helper,1,coeff*helper],[1,coeff*helper,1],[coeff*helper,1,coeff*helper]]
   for i in range (-1,2) : 
       for j in range (-1,2) : 
              somme += filtre[1+i][1+j]
              p0 += filtre[1+i][1+j]*tabpixel[y+i,x+j]

   p0= float(p0/somme)
   return p0 

img = img_as_float(data.coins())
img_out = img.copy()  

blur = noisy('gauss',img)
scipy.misc.imsave('bluredbygauss.png', blur)


row , line = blur.shape 
tabpixel = np.asarray(blur)
sigma = float(input("Entrer Sigma :  ")   )       
for x in range(-1 , line-1):
    for y in range (-1,row-1) : 
        p = function(tabpixel,sigma,x,y)
        img_out.itemset((y,x),p)

scipy.misc.imsave('gaussexpo.png', img_out)
cv2.imshow('blur',blur)
cv2.imshow('gauss',img_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
