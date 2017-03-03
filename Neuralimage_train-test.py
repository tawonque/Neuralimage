#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:37:01 2016  

@author: Tavo
"""


'''This script will create training and testing datasets from two rasters
by tawonque 20/12/2016'''

#%% Import modules

import os
import numpy as np

import rasterio
import rasterio.tools.mask

#%% If necessary change directory

working_directory = './Data_science/Neuralimage'

print('Initial directory --> ', os.getcwd())
os.chdir(working_directory)
print('Final working directory --> ', os.getcwd())

#%% input rasters

# Satellite image
main_raster = './satellite_raster.tif'

# Classification binary image: inside-polygon (1) / outside-polygon (0)
polygon_raster = './masked_raster_final.tif'

# Set the desired dataset to use in this script ---- main or polygon rasters?
dataset = main_raster
#dataset = polygon_raster

# X-axis percentage that we want as a training image. % from left to right
xslice = 0.6

#%% Let's open the raster to split and explore some of the characteristics

if dataset == main_raster:
    with rasterio.open(dataset) as src:
        #array = src.read()
        b, g, r, n = (src.read(k) for k in (1, 2, 3, 4))    
        print(src.profile)
        print('--o--o--o--o--o--o--o--o--o--o--o--o--')
        print(src.affine)    
        
if dataset == polygon_raster:
    with rasterio.open(dataset) as src:
        array = src.read(1)
        print(src.profile)
        print('--o--o--o--o--o--o--o--o--o--o--o--o--')
        print(src.affine)
    

#%% Let's explore and design the cuting off dimensions
    
        pixel_w = src.affine[0] # Extract pixel width and height
        pixel_h = src.affine[4]

        train_w_pix = int(np.round((src.width * xslice), decimals=0))
        train_w = train_w_pix * pixel_w 
        ur_train_x = src.affine[2] + train_w # upper-right corner x
        ur_train_y = src.affine[5] # upper-right corner y [unchanged in this case]

        test_w_pix = int(src.width - np.round((src.width * xslice), decimals=0))
        test_w = test_w_pix * pixel_w
        ul_test_x = ur_train_x # upper-left corner x
        ul_test_y = ur_train_y # upper-left corner y

#%% Now let's extract the 'train' and 'test' datasets from the main array
# In this case we are extracting a subset of the x dimension, but it could
# be adapted to extract a tile too (limitng x and y coordinates)
# Notice rotation is zero, which makes the transformations much simpler

#%%### Let's prepare the training set
        
        train_meta = src.meta.copy() # This can be made more generic, but for now let's only substitute what we need to change  
        del train_meta['transform']
        train_meta.update({'affine': rasterio.Affine(0.3, 0.0, 335394.6,0.0, -0.3, 7399004.399999999),
                       'width': train_w_pix})
        if dataset == main_raster:
            train_b = b[:src.height, :train_w_pix]
            train_g = g[:src.height, :train_w_pix]
            train_r = r[:src.height, :train_w_pix]
            train_n = n[:src.height, :train_w_pix]

            train_meta.update({'count': 4})        
        
        if dataset == polygon_raster:
            train_array = array[:src.height, :train_w_pix]

            train_meta.update({'count': 1})
            
        #train_window = ((0, src.height), (0, train_w_pix))

#%%### and now the test set
    
        test_meta = src.meta.copy() # This can be made more generic, but for now let's only substitute what we need to change  
        del test_meta['transform']
        test_meta.update({'affine': rasterio.Affine(0.3, 0.0, ul_test_x, 0.0, -0.3, ul_test_y),
                          'width': test_w_pix})
        
        if dataset == main_raster:
            test_b = b[:, train_w_pix:]
            test_g = g[:, train_w_pix:]
            test_r = r[:, train_w_pix:]
            test_n = n[:, train_w_pix:]
            
            test_meta.update({'count': 4})
        
        if dataset == polygon_raster:
            test_array = array[:, train_w_pix:]
            
            test_meta.update({'count': 1})    

            #test_window = ((0, src.height), (train_w_pix, src.width))

#%% Let's create the two new files 'train' and 'test' with the right parameters
        
        with rasterio.open('./train_raster.tif', 'w', **train_meta) as train:
            for k, arr in [(1, train_b), (2, train_g), (3, train_r), (4, train_n)]:
                train.write_band(k, arr)
        with rasterio.open('./test_raster.tif', 'w', **test_meta) as test:
            for k, arr in [(1, test_b), (2, test_g), (3, test_r), (4, test_n)]:
                test.write_band(k, arr)

#%%
        with rasterio.open('./train_polygon.tif', 'w', **train_meta) as train_polygon:
            train_polygon.write_band(1, train_array)
        with rasterio.open('./test_polygon.tif', 'w', **test_meta) as test_polygon:
            test_polygon.write_band(1, test_array)
            
        
    
