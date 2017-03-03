#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:09:18 2016

@author: Tavo
"""

'''This script will read numpy arrays (pickled objects .npy) and use them as input 
in a 2D convolutional neural network to predict inside-polygon locations
by tawonque 29/12/2016'''

#%%
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, floatX=float32"
  
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline

#%% If necessary change directory

working_directory = './Neuralimage'

print('Initial directory --> ', os.getcwd())
os.chdir(working_directory)
print('Final working directory --> ', os.getcwd())

#%% Open the files and set some prelim params

# Training X
X_train = np.load('./X_train_set.npy')

# Training y
y_train = np.load('./y_train_set.npy')
y_train = np_utils.to_categorical(y_train)

# Testing X
X_test = np.load('./X_test_set.npy')

# Training y
y_test = np.load('./y_test_set.npy')
y_test = np_utils.to_categorical(y_test)

# Testing raster (for some operations at the end)
test_raster = './test_raster.tif'

# Indexes
subset_train_index_1 = np.load('./X_train_indexes_1.npy')
subset_train_index_0 = np.load('./X_train_indexes_0.npy')
subset_test_index_1 = np.load('./X_test_indexes_1.npy')
subset_test_index_0 = np.load('./X_test_indexes_0.npy')

# Shape of y_test (or of X_test_array with one band)
y_test_array_shape = np.load('./y_test_array_shape.npy')

# Classes
num_classes = y_train.shape[1]

# Max of the satellite raster. useful for normalisation later on
bgrn_max = np.load('./bgrn_max.npy')

# Dimension of the frames
frame_shape = X_train.shape[1:4]

# Prediction raster
polygon_predict_raster = './polygon_predictions.tif'

#%% plot 4 images as gray scale

# Choose band
band = 3

# Choose positive or negative frames
frame = X_train # positive
#frame = X_test # negative

# Let's plot

plt.subplot(331)
plt.imshow(frame[0][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(frame[1][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(frame[2][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.imshow(frame[3][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(frame[4][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(frame[5][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(337)
plt.imshow(frame[6][band,...], cmap=plt.get_cmap('gray'))
plt.subplot(338)
plt.imshow(frame[7][band,...], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()

#%% fix random seed for reproducibility

seed = 7
np.random.seed(seed)

#%% normalize inputs. I decided to normalise it using the max of all the bands together.

for i in range(len(bgrn_max)):
    X_train[:,i,:,:] = X_train[:,i,:,:] / bgrn_max[1]
    X_test[:,i,:,:] = X_test[:,i,:,:] / bgrn_max[1]

#%% Define the network

def convo_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 10, 10, border_mode='same', input_shape=frame_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(30, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#%% Build, fit and evaluate the model

# Build the model
model = convo_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=200, batch_size=400, verbose=2)

# Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#%% Output an array with the predictions

y_predict = model.predict_classes(X_test)

# And output the confusion matrix too...

metrics.confusion_matrix(y_test[:,1], y_predict)

#%% We can reconstruct a projected raster file with the predicted values

# Create an empty array (NaNs or -999s) with the dimensions of the Test raster

polygon_locations_NN = np.full(y_test_array_shape, -999, dtype=int) #, order='C')

#%% Assign the values of the y_predict array to their correct index of the test raste
# In our case, we had a test y available, which is not always the case. However, the 
# extraction of the indexes can be done from X_test instead of y_test and this would not alter
# the general workflow. In our case, the indexes are...

y_predict_indexes = np.append(subset_test_index_1, subset_test_index_0, axis=0)

for i in range(y_predict.shape[0]):
    band = 0
    h = y_predict_indexes[i,1]
    w = y_predict_indexes[i,2]

    polygon_locations_NN[(band, h, w)] = y_predict[i]

polygon_locations_NN = polygon_locations_NN.astype(np.int32)

#%% Open the test raster (on which the predictions where made)
# Copy the profile parameters

with rasterio.open(test_raster) as src:
    out_meta = src.meta.copy()
    out_meta.update({'count': 1,
                     'dtype': 'int32'})

#%% Save a new raster with the predictions
    
with rasterio.open(polygon_predict_raster, 'w', **out_meta) as dst:
    dst.write(polygon_locations_NN) # '1' is to fix the indexes of the output file



