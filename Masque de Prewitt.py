# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:00:01 2019

@author: Samacyc
"""

from PIL import Image
from scipy import misc
from skimage import data
#import cv2
import numpy as np
import scipy.misc
def produit_mat(Matrice_1,Matrice_2):
    n = np.shape(Matrice_1)[0]
    Produit = np.zeros((n,n))

    for i in range(n) : 
        for j in range (n) : 
              Produit[i][j] = Matrice_1[i][j]*Matrice_2[i][j]
    
    return Produit
    
def somme (matrice) : 
    somme = 0
    n = matrice.shape[0]
    for i in range(n) : 
        for j in range (n) : 
            somme += matrice[i][j]
    
    return somme

def reshaping(matrice) : 
    line , row = np.shape(matrice)
    new_arr = np.zeros((line+2,row+2))

    line1 , row1 = np.shape(new_arr)
    for i in range (1,line1-1):
        for j in range(1,row1-1):
            new_arr[i][j]=matrice[i-1][j-1]
    for j in range(1,row1-1):
      new_arr[0][j] = matrice[1][j-1]
      new_arr[line1-1][j] = new_arr[line1-3][j]
    
    for i in range(line1):
      new_arr[i][0]=new_arr[i][2]
      new_arr[i][row1-1] = new_arr[i][row1-3]
    return new_arr
    
def decrease (matrice) : 
    line , row = np.shape(matrice)
    new_arr = np.zeros((line-2,row-2))
    line1 , row1 = np.shape(new_arr)
    for i in range(line1):
        for j in range(row1) : 
            new_arr[i][j]=matrice[i+1][j+1]
            
    return new_arr

def copy(matrice,x,y):
    new_arr = np.zeros((3,3))
    for i in range(3) :
        for j in range(3) : 
            new_arr[i][j] = matrice[x-1+i][y-1+j]
    
    return new_arr

def final_func(matrice,matrice1,x,y):
    helper = np.zeros((3,3))
    helper = copy(matrice,x,y)
            
    helper = produit_mat(helper,matrice1)
    n = somme(helper)
    
    return float(n)
    


    
hx = np.array([[-1/3,0,1/3],[-1/3,0,1/3],[-1/3,0,1/3]])
hy = np.array([[-1/3,-1/3,-1/3],[0,0,0],[1/3,1/3,1/3]])
img = (data.coins())
img_outx = img.copy() 
img_outy = img.copy()  
line , row = img.shape
img_outx = reshaping(img_outx)
img_outy = reshaping(img_outy)
print(img_outx)
line1 , row1 = np.shape(img_outx)
helper = np.zeros((3,3))
for x in range(1,line1-1):
     for y in range (1,row-1) : 
         img_outx[x][y] = final_func(img_outx,hx,x,y)
         img_outy[x][y]  = final_func(img_outy,hy,x,y)
img_outx = decrease(img_outx)
img_outy = decrease(img_outy)
#print(img_outx)
img_outx = Image.fromarray(img_outx)
img_outx = img_outx.convert('RGB')
scipy.misc.imsave('outx.png', img_outx)
img_outy = Image.fromarray(img_outy)
img_outy = img_outy.convert('RGB')
scipy.misc.imsave('outy.png', img_outy)
img_outx.show()
#img_outy.show()
