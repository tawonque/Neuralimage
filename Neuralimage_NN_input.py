# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''This script will prepare the data from teh raster and output the requested 
training and testing datasets. In this case, the datasets will be just a small sample
of the total dataset available.
by tawonque 28/12/2016'''

#%% Import modules

import os
import numpy as np

import rasterio
import matplotlib.pyplot as plt

%matplotlib inline

#%% If necessary change directory

working_directory = './Neuralimage'

print('Initial directory --> ', os.getcwd())
os.chdir(working_directory)
print('Final working directory --> ', os.getcwd())

#%% Files to use

train_raster = './train_raster.tif'
train_feature = './train_polygon.tif'

test_raster = './test_raster.tif'
test_feature = './test_polygon.tif'

#%% fix random seed for reproducibility

seed = 7
np.random.seed(seed)

#%% Let's read the rasters in

with rasterio.open(train_raster) as X_train:
    X_train_array = X_train.read()
    
#%%
with rasterio.open(train_feature) as y_train:
    y_train_array = y_train.read()

#%%
with rasterio.open(test_raster) as X_test:
    X_test_array = X_test.read()

#%%
with rasterio.open(test_feature) as y_test:
    y_test_array = y_test.read()
    
#%% Let's calculate the max for all bands and save them for future reference

b_max = np.max(X_test_array[0,...] & X_test_array[0,...])
g_max = np.max(X_test_array[1,...] & X_test_array[1,...])
r_max = np.max(X_test_array[2,...] & X_test_array[2,...])
n_max = np.max(X_test_array[3,...] & X_test_array[3,...])
bgrn_max = [b_max, g_max, r_max, n_max]
np.save('./bgrn_max.npy', bgrn_max)

#%% Let's start thinking about the size of the subset windows. THese frames should be big enough.
# if we consider that the pixels are 30 cm, cars are 4 to 5 meters long.
# Say, 4 cars -at least- plus some separation between them. 50 x 50 pixels?
# I want an odd number so I can centre my window in the categorical pixel.
# to make a 49 x 49 window, select l = 24 (24 samples in each direction from pixel)

l = 24

#%% #%% ####### Train dataset #######

height = X_train_array.shape[1] - 1 # represents the height of the test image less one
width = X_train_array.shape[2] - 1 # represents the width of the test image less one

# select k indexes corresponding to 'inside-polygon' locations
# First generate all the indexes of pixels in inside-polygon locations
train_index_1 = np.transpose(np.where(y_train_array == 1))
train_index_1 = train_index_1[(train_index_1[:,1] <= (height-(l+1))) & (train_index_1[:,1] >= l)] # filter along height to avoid edges
train_index_1 = train_index_1[(train_index_1[:,2] <= (width-(l+1))) & (train_index_1[:,2] >= l)] # filter along width to avoid edges

train_index_0 = np.transpose(np.where(y_train_array == 0))                            
train_index_0 = train_index_0[(train_index_0[:,1] <= (height-(l+1))) & (train_index_0[:,1] >= l)] # filter along height to avoid edges
train_index_0 = train_index_0[(train_index_0[:,2] <= (width-(l+1))) & (train_index_0[:,2] >= l)] # filter along width to avoid edges

# and then let's do a small subset, of only 5 locations
# number of positive and negative locations on which to base the model
k = 200

subset_train_index_1 = train_index_1[np.random.randint(0, train_index_1.shape[0], k)]
subset_train_index_0 = train_index_0[np.random.randint(0, train_index_0.shape[0], k)]
                              
#%% Let's design the window for each inside-polygon location. 
#i (centre)
#i-24
#i+24+1

# positive frames...

X_train_set = []

s = k
for N in range(0,k):
    n = subset_train_index_1[N,1]
    m = subset_train_index_1[N,2]
    X_train_frames_N = X_train_array[:, (n-l):(n+l+1), (m-l):(m+l+1)]
    X_train_set = np.append(X_train_set, X_train_frames_N) 
    
print(s)

# negative frames...

t = k
for N in range(k):
    n = subset_train_index_0[N,1]
    m = subset_train_index_0[N,2]
    X_train_frames_N = X_train_array[:, (n-l):(n+l+1), (m-l):(m+l+1)]
    X_train_set = np.append(X_train_set, X_train_frames_N) 
        
print(t)

# reshape to be [samples][bands][rows frame][columns frame]
X_train_set = X_train_set.reshape(t+s, 4, (l*2)+1, (l*2)+1).astype('float32')
    
#%% Now that we have the X train dataset, let's get the y train dataset using the same indexes
# positives first, negatives later to be consistent with the array 'X_train_set'

y_train_set = np.append([1]*s,[0]*t)


#%% ####### Test dataset #######

#We have to extract the test set. Let's extract a similar number of positive and negative frames.

# select k indexes corresponding to 'inside-polygon' locations
# First generate all the indexes of pixels in inside-polygon locations
height = X_test_array.shape[1] - 1 # represents the height of the test image less one
width = X_test_array.shape[2] - 1 # represents the width of the test image less one

test_index_1 = np.transpose(np.where(y_test_array == 1))
test_index_1 = test_index_1[(test_index_1[:,1] <= (height-(l+1))) & (test_index_1[:,1] >= l)] # filter along height to avoid edges
test_index_1 = test_index_1[(test_index_1[:,2] <= (width-(l+1))) & (test_index_1[:,2] >= l)] # filter along width to avoid edges

test_index_0 = np.transpose(np.where(y_test_array == 0))
test_index_0 = test_index_0[(test_index_0[:,1] <= (height-(l+1))) & (test_index_0[:,1] >= l)] # filter along height to avoid edges
test_index_0 = test_index_0[(test_index_0[:,2] <= (width-(l+1))) & (test_index_0[:,2] >= l)] # filter along width to avoid edges

# and then let's do a small subset, of only 5 locations
# number of positive and negative locations on which to base the model
k = 200

subset_test_index_1 = test_index_1[np.random.randint(0, test_index_1.shape[0], k)]
subset_test_index_0 = test_index_0[np.random.randint(0, test_index_0.shape[0], k)]
                              
#%% Same windows as before
#i (centre)
#i-24
#i+24+1

# positive frames...

X_test_set = []

u = k
for N in range(0,k):
    n = subset_test_index_1[N,1]
    m = subset_test_index_1[N,2]
    X_test_frames_N = X_test_array[:, (n-l):(n+l+1), (m-l):(m+l+1)]
    X_test_set = np.append(X_test_set, X_test_frames_N) 
    
    #if X_test_frames_N.shape[2] != (l*2+1): # Quick trick to get rid of frame sat the edges of the training image
    #    u -= 1                               # Same trick might be applied on the other dimension
print(u)

# negative frames...

v = k
for N in range(k):
    n = subset_test_index_0[N,1]
    m = subset_test_index_0[N,2]
    X_test_frames_N = X_test_array[:, (n-l):(n+l+1), (m-l):(m+l+1)]
    X_test_set = np.append(X_test_set, X_test_frames_N) 
    
    #if X_test_frames_N.shape[2] != (l*2+1): # Quick trick to get rid of frame sat the edges of the training image
    #    v -= 1                               # Same trick might be applied on the other dimension
print(v)

# reshape to be [samples][bands][rows frame][columns frame]
X_test_set = X_test_set.reshape(u+v, 4, (l*2)+1, (l*2)+1).astype('float32')
    
#%% Now that we have the X test dataset, let's get the y test dataset using the same indexes
# positives first, negatives later to be consistent with the array 'X_test_set'

y_test_set = np.append([1]*u,[0]*v)


#%% Let's save the four files, which we will use as the input for our Neural Network

# Training X
np.save('./X_train_set.npy', X_train_set)

# Training y
np.save('./y_train_set.npy', y_train_set)

# Testing X
np.save('./X_test_set.npy', X_test_set)

# Testing y
np.save('./y_test_set.npy', y_test_set)

# Indexes to be able to restore the pixel to their correct coordinates
np.save('./X_train_indexes_1', subset_train_index_1)
np.save('./X_train_indexes_0', subset_train_index_0)
np.save('./X_test_indexes_1', subset_test_index_1)
np.save('./X_test_indexes_0', subset_test_index_0)

# Shape of the y_test_array (it will help in restoring the pixel to their original locations)
np.save('./y_test_array_shape.npy', y_test_array.shape)
                            
    





                             
             
             




    