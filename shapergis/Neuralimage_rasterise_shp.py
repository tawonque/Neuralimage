#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:18:28 2016

@author: Tavo
"""

'''This script will create a raster from a polygon shapefile
by tawonque 20/12/2016'''

#%% Import modules

import os
import sys
import numpy as np

from osgeo import gdal

import fiona
import rasterio
import rasterio.tools.mask

#%% If necessary change directory

working_directory = './Data_science/Neuralimage'

print('Initial directory --> ', os.getcwd())
os.chdir(working_directory)
print('Final working directory --> ', os.getcwd())

#%% define input paths

inshapefile = './polygon_feature-EPSG32723.shp'
main_raster = './satellite_raster.tif'
oneband_raster = './satellite_raster-oneband.tif'
masked_raster = './masked_raster.tif'
final_raster = './masked_raster_final.tif'

#%% Copy the info from only one band into a new raster
'''
This is another option using gdal, bu it is not too pythonic!:
    
os.system('gdal_translate -b 1 -mask none ' + main_raster + ' ' + oneband_raster)

print(gdal.Info(oneband_raster)) #Some basic info on the output raster
'''

with rasterio.open(main_raster) as src:
    array = src.read(1)
    out_meta = src.meta.copy()
    out_meta.update({'count': 1})

with rasterio.open(oneband_raster, 'w', **out_meta) as dst:
    dst.write(array, 1) # '1' is to fix the indexes of the output file
    
#%% Now let's clip the oneband_raster using rasterio and fiona

with fiona.open(inshapefile, "r") as shapefile:
    features = [feature["geometry"] for feature in shapefile]

with rasterio.open(oneband_raster) as src:
    out_image, out_transform = rasterio.tools.mask.mask(src, features, crop=False)
    out_meta = src.meta.copy()
    
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(masked_raster, "w", **out_meta) as dest:
    dest.write(out_image)

#%% Open the new one-band raster and make all the values equal to 1

with rasterio.open(masked_raster) as src:
    array = src.read()
    array[array != 0] = 1
    profile = src.profile

with rasterio.open(final_raster, 'w', **profile) as dst:
    dst.write(array)
    dst.close()
# Now we have a raster with 1 for inside-polygon locations and 0 for outside-polygon locations    
