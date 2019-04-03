
# coding: utf-8

# In[ ]:


import cv2
import numpy
import glob
import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd


# In[ ]:


counter = 1000
root = '' #path from where training data logo images need to collected
dest = 'training_data/' # folder where images are organized  according to brand they represent
onlyfiles = [f for f in listdir(root) if isfile(join(root, f))]
images = [cv2.imread(file) for file in glob.glob(root + '/*jpg')]


# In[ ]:


with open('flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt', 'r') as f:
    lines = f.readlines()

# remove spaces

#lines = [line.replace(' ', '') for line in lines]

# finally, write lines in the file
#with open('flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt', 'w') as f:
    #f.writelines(lines)

data = pd.read_csv('flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt', sep="\t", header=None)


# In[ ]:


for f in onlyfiles:
    for ID in data[0]: 
        if f==ID:
            loc_txt_file = list(data[0]).index(f)
            loc_root = list(onlyfiles).index(f)
            img_name = data.iloc[b][1]+'/'+f
           
            k = os.path.join(dest,img_name)
            print(k)
            cv2.imwrite(k, images[loc_root])
            try:
                os.remove(root + '/' + f)
            except:
                pass
            

