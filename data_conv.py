
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import os
from PIL import Image
from os import listdir
from os.path import isfile, join


# In[2]:


counter = 0
mypath = ''           # give the path of the folder where you want to store the processed images
root = ''         # give the path where the images from flickr27 dataset has been saved
i = []
pi = []
onlyfiles = [f for f in glob.glob(root + "/*jpg") if isfile(join(mypath, f))] # or png if png files are also there in root folder 
print(onlyfiles[0][56:])
images = [cv2.imread(file) for file in glob.glob(root + "/*jpg")] # or png if png files are also there in root folder 
img_list = list(images)


# In[3]:


for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,117,240,cv2.THRESH_BINARY)
    #print('****************************')
    pi.append(gray)
    i.append(img)
    #print(onlyfiles[counter][56:])
    cv2.imwrite(os.path.join(mypath , (onlyfiles[counter][56:])), thresh)
    #print(counter)
    counter=int(counter)+1

