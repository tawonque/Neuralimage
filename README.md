# Neuralimage
Make predictions using polygon shapefile and a satellite raster

These scripts:

[1] take a shapefile (polygon), projects it, converts it into a raster
[2] splits the input raster and the polygon-raster into test and train datasets
[3] creates a training set of labelled image frames, both inside (1) or outside (0) the polygons
[4] runs a 2D convolutional neural network to predict the parts of the image that correspond to a feature defined by the polygon.

The features to identifiy could be parks, parking lots, certain types of roofs, etc.
