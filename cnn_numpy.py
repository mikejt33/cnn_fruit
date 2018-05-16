#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:31:43 2018

@author: Michael
"""

import pickle #saving and loading our serialized model 
import numpy as np #matrix math
from app.model.preprocessor import Preprocessor as img_prep #image preprocessing


def read_file(filename):
    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label

#class for loading our saved model and classifying new images
class LiteOCR:
    
	def __init__(self, fn="alpha_weights.pkl", pool_size=2):
        #load the weights from the pickle file and the meta data
		[weights, meta] = pickle.load(open(fn, 'rb'), encoding='latin1') #currently, this class MUST be initialized from a pickle file
	#list to store labels
        self.vocab = meta["vocab"]
        
        #how many rows and columns in an image
		self.img_rows = meta["img_side"] ; self.img_cols = meta["img_side"]
        
        #load our CNN
		self.CNN = LiteCNN()
        #with our saved weights
		self.CNN.load_weights(weights)
        #define the pooling layers size
		self.CNN.pool_size=int(pool_size)
    
    #classify new image
	def predict(self, image):
		print(image.shape)
        #vectorize the image into the right shape for our network
		X = np.reshape(image, (1, 1, self.img_rows, self.img_cols))
		X = X.astype("float32")
        
        #make the prediction
		predicted_i = self.CNN.predict(X)
        #return the predicted label
		return self.vocab[predicted_i]

class LiteCNN:
	def __init__(self):
        # a place to store the layers
		self.layers = [] 
        # size of pooling area for max pooling
		self.pool_size = None 

	def load_weights(self, weights):
		assert not self.layers, "Weights can only be loaded once!"
        #add the saved matrix values to the convolutional network
		for k in range(len(weights.keys())):
			self.layers.append(weights['layer_{}'.format(k)])

	def predict(self, X):        
        #here is where the network magic happens at a high level
        h = self.cnn_layer(X, layer_i=0, border_mode="full"); X= h
        h = self.relu_layer(X); X = h;
        h = self.cnn_layer(X, layer_i=2, border_mode="valid"); X = h
        h = self.relu_layer(X); X = h;
        h = self.maxpooling_layer(X); X = h
        h = self.dropout_layer(X, .25); X = h
        h = self.flatten_layer(X, layer_i=7); X = h;
        h = self.dense_layer(X, fully, layer_i=10); x = H
        h = self.softmax_layer2D(X); x = h
        max_i = self.classify(X)
        return max_i[0]
    
    #given our feature map we've learned from convolving around the image
    #lets make it more dense by performing pooling, specifically max pooling
    #we'll select the max values from the image matrix and use that as our new feature map
	def maxpooling_layer(self, convolved_features):
        #given our learned features and images
		nb_features = convolved_features.shape[0]
		nb_images = convolved_features.shape[1]
		conv_dim = convolved_features.shape[2]
		res_dim = int(conv_dim / self.pool_size)       #assumed square shape

        #initialize our more dense feature list as empty
		pooled_features = np.zeros((nb_features, nb_images, res_dim, res_dim))
        #for each image
		for image_i in range(nb_images):
            #and each feature map
			for feature_i in range(nb_features):
                #begin by the row
				for pool_row in range(res_dim):
                    #define start and end points
					row_start = pool_row * self.pool_size
					row_end   = row_start + self.pool_size

                    #for each column (so its a 2D iteration)
					for pool_col in range(res_dim):
                        #define start and end points
						col_start = pool_col * self.pool_size
						col_end   = col_start + self.pool_size
                        
                        #define a patch given our defined starting ending points
						patch = convolved_features[feature_i, image_i, row_start : row_end,col_start : col_end]
                        #then take the max value from that patch
                        #store it. this is our new learned feature/filter
						pooled_features[feature_i, image_i, pool_row, pool_col] = np.max(patch)
		return pooled_features

    #convolution is the most important of the matrix operations here
    #well define our input, lauyer number, and a border mode (explained below)
	def cnn_layer(self, X, layer_i=0, border_mode = "full"):
        #we'll store our feature maps and bias value in these 2 vars
		features = self.layers[layer_i]["param_0"]
		bias = self.layers[layer_i]["param_1"]
        #how big is our filter/patch?
		patch_dim = features[0].shape[-1]
        #how many features do we have?
		nb_features = features.shape[0]
        #How big is our image?
		image_dim = X.shape[2] #assume image square
        #R G B values
		image_channels = X.shape[1]
        #how many images do we have?
		nb_images = X.shape[0]
        
        #With border mode "full" you get an output that is the "full" size as the input. 
        #That means that the filter has to go outside the bounds of the input by "filter size / 2" - 
        #the area outside of the input is normally padded with zeros.
		if border_mode == "full":
			conv_dim = image_dim + patch_dim - 1
        #With border mode "valid" you get an output that is smaller than the input because 
        #the convolution is only computed where the input and the filter fully overlap.
		elif border_mode == "valid":
			conv_dim = image_dim - patch_dim + 1
        
        #we'll initialize our feature matrix
		convolved_features = np.zeros((nb_images, nb_features, conv_dim, conv_dim));
        #then we'll iterate through each image that we have
		for image_i in range(nb_images):
            #for each feature 
			for feature_i in range(nb_features):
                #lets initialize a convolved image as empty
				convolved_image = np.zeros((conv_dim, conv_dim))
                #then for each channel (r g b )
				for channel in range(image_channels):
                    #lets extract a feature from our feature map
					feature = features[feature_i, channel, :, :]
                    #then define a channel specific part of our image
					image   = X[image_i, channel, :, :]
                    #perform convolution on our image, using a given feature filter
					convolved_image += self.convolve2d(image, feature, border_mode);

                #add a bias to our convoved image
				convolved_image = convolved_image + bias[feature_i]
                #add it to our list of convolved features (learnings)
				convolved_features[image_i, feature_i, :, :] = convolved_image
		return convolved_features

    #In a dense layer, every node in the layer is connected to every node in the preceding layer.
	def dense_layer(self, X, layer_i=0):
        #so we'll initialize our weight and bias for this layer
		W = self.layers[layer_i]["param_0"]
		b = self.layers[layer_i]["param_1"]
        #and multiply it by our input (dot product)
		output = np.dot(X, W) + b
		return output
    
    

	@staticmethod
    
    #so what does the convolution operation look like?, given an image and a feature map (filter)
	def convolve2d(image, feature, border_mode="full"):
        #we'll define the tensor dimensions of the image and the feature
		image_dim = np.array(image.shape)
		feature_dim = np.array(feature.shape)
        #as well as a target dimension
		target_dim = image_dim + feature_dim - 1
        #then we'll perform a fast fourier transform on both the input and the filter
        #performing a convolution can be written as a for loop but for many convolutions
        #this approach is too comp. expensive/slow. it can be performed orders of magnitude
        #faster using a fast fourier transform. 
		fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
        #and set the result to our target 
		target = np.fft.ifft2(fft_result).real

		if border_mode == "valid":
			# To compute a valid shape, either np.all(x_shape >= y_shape) or
			# np.all(y_shape >= x_shape).
            #decide a target dimension to convolve around
			valid_dim = image_dim - feature_dim + 1
			if np.any(valid_dim < 1):
				valid_dim = feature_dim - image_dim + 1
			start_i = (target_dim - valid_dim) // 2
			end_i = start_i + valid_dim
			target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
		return target

	def relu_layer(x):
        #turn all negative values in a matrix into zeros
		z = np.zeros_like(x)
		return np.where(x>z,x,z)

	def softmax_layer2D(w):
        #this function will calculate the probabilities of each
        #target class over all possible target classes. 
		maxes = np.amax(w, axis=1)
		maxes = maxes.reshape(maxes.shape[0], 1)
		e = np.exp(w - maxes)
		dist = e / np.sum(e, axis=1, keepdims=True)
		return dist

    #affect the probability a node will be turned off by multiplying it
    #by a p values (.25 we define)
	def dropout_layer(X, p):
		retain_prob = 1. - p
		X *= retain_prob
		return X

    #get the largest probabililty value from the list
	def classify(X):
		return X.argmax(axis=-1)

    #tensor transformation, less dimensions
	def flatten_layer(X):
		flatX = np.zeros((X.shape[0],np.prod(X.shape[1:])))
		for i in range(X.shape[0]):
			flatX[i,:] = X[i].flatten(order='C')
		return flatX