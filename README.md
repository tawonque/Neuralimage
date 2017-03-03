# Neuralimage<br>
Make predictions using polygon shapefile and a satellite raster<br><br>

These scripts:<br><br>

[1] take a shapefile (polygon), projects it, converts it into a raster<br>
[2] splits the input raster and the polygon-raster into test and train datasets<br>
[3] creates a training set of labelled image frames, both inside (1) or outside (0) the polygons<br>
[4] runs a 2D convolutional neural network to predict the parts of the image that correspond to a feature defined by the polygon.<br><br>

The features to identifiy could be parks, parking lots, certain types of roofs, etc.
