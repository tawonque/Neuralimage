#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:14:00 2016

@author: Tavo
"""

'''This script will take a polygon shapefile and project it to UTM 23S using ogr/gdal
by tawonque 19/12/2016'''

#%% Import modules

import os
import sys

from osgeo import ogr
from osgeo import osr

#%% If necessary, change directory

working_directory = './Data_science/Neuralimage'

print('Initial directory --> ', os.getcwd())
os.chdir(working_directory)
print('Final working directory --> ', os.getcwd())

#%% define input path and other embedded params

infile = './polygon_feature.shp'
outfile = './polygon_feature-EPSG32723.shp'


driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open(infile, 0)
layer = dataSource.GetLayer()
daLayer = dataSource.GetLayer(0)
layerDefinition = daLayer.GetLayerDefn()

#Show fields in the layer   

for i in range(layerDefinition.GetFieldCount()):
    print(layerDefinition.GetFieldDefn(i).GetName())
    
#%% Check the spatial reference, if none, define one and its transform to UTM

spatialRef = layer.GetSpatialRef()
if spatialRef == None:
    print('Undefined spatial reference')
 
#get path and filename seperately 
(outfilepath, outfilename) = os.path.split(outfile) 
#get file name without extension            
(outfileshortname, extension) = os.path.splitext(outfilename) 
 
#%%
# Spatial Reference of the input file 
# Access the Spatial Reference and assign the input projection 
Wkt4326 = ('''GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]''')

inSpatialRef = osr.SpatialReference() 
#inSpatialRef.ImportFromProj4(4326) # unprojected WGS84 
inSpatialRef.ImportFromWkt(Wkt4326)

#%%
# Spatial Reference of the output file 
# Access the Spatial Reference and assign the output projection 
# UTM 33N which we use for all Spitsbergen does not work since area too far  
# outside UTM 33 
Wkt32723 = str('''PROJCS["WGS 84 / UTM zone 23S",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-45],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",10000000],
    AUTHORITY["EPSG","32723"],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]''')

outSpatialRef = osr.SpatialReference() 
#outSpatialRef.ImportFromEPSG(32722) #(20823) #UTM zone 23S !!!Issues with the projection libraries
outSpatialRef.ImportFromWkt(Wkt32723)

# create Coordinate Transformation 
coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef) 
 
#%% Open the input shapefile and get the layer 
driver = driver
indataset = dataSource
if indataset is None: 
    print('Could not open file')
    sys.exit(1) 
inlayer = layer

#%% Create the output shapefile but check first if file exists 
if os.path.exists(outfile): 
    driver.DeleteDataSource(outfile) 
 
outdataset = driver.CreateDataSource(outfile) 
if outfile is None: 
    print('Could not create file')
    sys.exit(1) 
outlayer = outdataset.CreateLayer(outfileshortname, geom_type=ogr.wkbPolygon) 
 
#%% Get the FieldDefn for attributes and add to output shapefile 
# (which i know are in my file) 
feature = inlayer.GetFeature(0) 
fieldDefn1 = feature.GetFieldDefnRef('rid') 
 
outlayer.CreateField(fieldDefn1) 
 
# get the FeatureDefn for the output shapefile 
featureDefn = outlayer.GetLayerDefn() 
 
#%% Loop through input features and write to output file 
infeature = inlayer.GetNextFeature() 
while infeature: 
  
    #get the input geometry 
    geometry = infeature.GetGeometryRef() 
  
    #reproject the geometry, each one has to be projected separately 
    geometry.Transform(coordTransform)
    #ogr._ogr.Geometry_Transform(geometry, coordTransform)
    
    #create a new output feature 
    outfeature = ogr.Feature(featureDefn) 
  
    #set the geometry and attribute 
    outfeature.SetGeometry(geometry) 
    outfeature.SetField('rid', infeature.GetField('rid')) 
  
    #add the feature to the output shapefile 
    outlayer.CreateFeature(outfeature) 
  
    #destroy the features and get the next input features 
    outfeature.Destroy 
    infeature.Destroy 
    infeature = inlayer.GetNextFeature() 
  
#%% Close the shapefiles 
indataset = None 
outdataset = None
 
#%% Create the prj projection file 
outSpatialRef.MorphToESRI() 
file = open(outfilepath + '\\'+ outfileshortname + '.prj', 'w') 
file.write(outSpatialRef.ExportToWkt()) 
file.close()
