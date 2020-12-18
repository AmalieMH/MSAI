#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:03:31 2020

@author: Amalie Monberg Hindsholm and Hongwei Li et al.

Inspiration from:
Fully Convolutional Network Ensembles for White Matter Hyperintensities Segmentation in MR Images
Article, by Hongwei Li et al 2018.

"""



from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Dropout, Activation
from keras.optimizers import Adam
from evaluation_single import getLesionDetectionNum, getImages 
from keras import backend as K

cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))
import glob
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### ----define loss function for U-net ------------



smooth = 1.
def dice_coef_for_training(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
	return -dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
		# width, the 3rd dimension
		cw = (target.get_shape()[2] - refer.get_shape()[2]).value
		assert (cw >= 0)
		if cw % 2 != 0:
			cw1, cw2 = int(cw/2), int(cw/2) + 1
		else:
			cw1, cw2 = int(cw/2), int(cw/2)
		# height, the 2nd dimension
		ch = (target.get_shape()[1] - refer.get_shape()[1]).value
		assert (ch >= 0)
		if ch % 2 != 0:
			ch1, ch2 = int(ch/2), int(ch/2) + 1
		else:
			ch1, ch2 = int(ch/2), int(ch/2)

		return (ch1, ch2), (cw1, cw2)




### ----define U-net architecture--------------
def get_unet(img_shape = None, weights_tf=None, custom_load_func = False):

		dim_ordering = 'tf'
		inputs = Input(shape = img_shape)
		concat_axis = -1
		### the size of convolutional kernels is defined here    
		conv1 = Convolution2D(64, 5, 5, border_mode='same', dim_ordering=dim_ordering, name='conv1_1')(inputs)
		ac = Activation('relu')(conv1)
		do = Dropout(0.2)(ac)
		conv1 = Convolution2D(64, 5, 5, border_mode='same', dim_ordering=dim_ordering)(do)
		ac = Activation('relu')(conv1)
		do = Dropout(0.2)(ac)
		pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(do)
		conv2 = Convolution2D(96, 3, 3, border_mode='same', dim_ordering=dim_ordering)(pool1)
		ac2 = Activation('relu')(conv2)
		do2 = Dropout(0.2)(ac2)
		conv2 = Convolution2D(96, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do2)
		ac2 = Activation('relu')(conv2)
		do2 = Dropout(0.2)(ac2)
		pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(do2)

		conv3 = Convolution2D(128, 3, 3, border_mode='same', dim_ordering=dim_ordering)(pool2)
		ac3 = Activation('relu')(conv3)
		do3 = Dropout(0.2)(ac3)
		conv3 = Convolution2D(128, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do3)
		ac3 = Activation('relu')(conv3)
		do3 = Dropout(0.2)(ac3)
		pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(do3)

		conv4 = Convolution2D(256, 3, 3, border_mode='same', dim_ordering=dim_ordering)(pool3)
		ac4 = Activation('relu')(conv4)
		do4 = Dropout(0.2)(ac4)
		conv4 = Convolution2D(256, 4, 4, border_mode='same', dim_ordering=dim_ordering)(do4)
		ac4 = Activation('relu')(conv4)
		do4 = Dropout(0.2)(ac4)
		pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(do4)
		
		
		conv5 = Convolution2D(512, 3, 3, border_mode='same', dim_ordering=dim_ordering)(pool4)
		ac5 = Activation('relu')(conv5)
		do5 = Dropout(0.2)(ac5)
		conv5 = Convolution2D(512, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do5)
		ac5 = Activation('relu')(conv5)
		do5 = Dropout(0.2)(ac5)

		up_conv5 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(do5)
		ch, cw = get_crop_shape(conv4, up_conv5)
		crop_conv4 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv4)
		up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)    
		conv6 = Convolution2D(256, 3, 3, border_mode='same', dim_ordering=dim_ordering)(up6)
		ac6 = Activation('relu')(conv6)
		do6 = Dropout(0.2)(ac6)
		conv6 = Convolution2D(256, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do6)
		ac6 = Activation('relu')(conv6)
		do6 = Dropout(0.2)(ac6)

		up_conv6 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(do6)
		ch, cw = get_crop_shape(conv3, up_conv6)
		crop_conv3 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv3)
		up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
		conv7 = Convolution2D(128, 3, 3, border_mode='same', dim_ordering=dim_ordering)(up7)
		ac7 = Activation('relu')(conv7)
		do7 = Dropout(0.2)(ac7)
		conv7 = Convolution2D(128, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do7)
		ac7 = Activation('relu')(conv7)
		do7 = Dropout(0.2)(ac7)

		up_conv7 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(do7)
		ch, cw = get_crop_shape(conv2, up_conv7)
		crop_conv2 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv2)
		up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
		conv8 = Convolution2D(96, 3, 3, border_mode='same', dim_ordering=dim_ordering)(up8)
		ac8 = Activation('relu')(conv8)
		do8 = Dropout(0.2)(ac8)
		conv8 = Convolution2D(96, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do8)
		ac8 = Activation('relu')(conv8)
		do8 = Dropout(0.2)(ac8)

		up_conv8 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(do8)
		ch, cw = get_crop_shape(conv1, up_conv8)
		crop_conv1 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv1)
		up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
		conv9 = Convolution2D(64, 3, 3, border_mode='same', dim_ordering=dim_ordering)(up9)
		ac9 = Activation('relu')(conv9)
		do9 = Dropout(0.2)(ac9)
		conv9 = Convolution2D(64, 3, 3, border_mode='same', dim_ordering=dim_ordering)(do9)
		ac9 = Activation('relu')(conv9)
		do9 = Dropout(0.2)(ac9)

		ch, cw = get_crop_shape(inputs, do9)
		conv9 = ZeroPadding2D(padding=(ch, cw), dim_ordering=dim_ordering)(conv9)
		conv10 = Convolution2D(1, 1, 1, activation='sigmoid', dim_ordering=dim_ordering)(conv9)
		model = Model(input=inputs, output=conv10)
		
		if not weights_tf == None:
			
			model.load_weights(weights_tf)
		
		model.compile(optimizer=Adam(lr=(1e-5)), loss=dice_coef_loss, metrics=[dice_coef_for_training])
		

		return model

###----define prepocessing methods/tricks for different datasets------------------------
def preprocessing(FLAIR_image):
    
	# Preprocessing: converting input to a size of 384x384, Gaussian Normalisation
	num_selected_slice = np.shape(FLAIR_image)[0]
	image_rows_Dataset = np.shape(FLAIR_image)[1]
	image_cols_Dataset = np.shape(FLAIR_image)[2]

	brain_mask_FLAIR = np.zeros((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
	FLAIR_image_suitable = np.zeros((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
	Brain_mask_flair_suitable = np.zeros((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

	# FLAIR --------------------------------------------
	if FLAIR_image.shape[0] == 44:
		thresh_FLAIR = thresh_FLAIR_2
	else:
		thresh_FLAIR = thresh_FLAIR_1
	
	brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
	brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
	if FLAIR_image.shape[2] == 512:
		img = brain_mask_FLAIR
		plt.imshow(img[20, :, :], interpolation='nearest')
		plt.savefig('mask'+str(FLAIR_image.shape[2])+str(thresh_FLAIR))
	
    
	brain_mask_FLAIR=brain_mask_FLAIR.astype(np.int16)
	FLAIR_image=FLAIR_image.astype(np.int16)
	
	if np.shape(FLAIR_image)[1] < 384:
		FLAIR_image_suitable[:, (int(cols_standard/2)-int(image_cols_Dataset/2)):(int(cols_standard/2)+int(image_cols_Dataset/2)), (int(cols_standard/2)-int(image_cols_Dataset/2)):(int(cols_standard/2)+int(image_cols_Dataset/2))] = FLAIR_image
		Brain_mask_flair_suitable[:, (int(cols_standard/2)-int(image_cols_Dataset/2)):(int(cols_standard/2)+int(image_cols_Dataset/2)), (int(cols_standard/2)-int(image_cols_Dataset/2)):(int(cols_standard/2)+int(image_cols_Dataset/2))] = brain_mask_FLAIR
		FLAIR_image = FLAIR_image_suitable
		brain_mask_FLAIR = Brain_mask_flair_suitable
	
	elif np.shape(FLAIR_image)[1] >= 384:
		FLAIR_image = FLAIR_image[:, (int(image_rows_Dataset/2)-int(rows_standard/2)):(int(image_rows_Dataset/2)+int(rows_standard/2)), (int(image_cols_Dataset/2)-int(cols_standard/2)):(int(image_cols_Dataset/2)+int(cols_standard/2))]
		brain_mask_FLAIR = brain_mask_FLAIR[:, (int(image_rows_Dataset/2)-int(rows_standard/2)):(int(image_rows_Dataset/2)+int(rows_standard/2)), (int(image_cols_Dataset/2)-int(cols_standard/2)):(int(image_cols_Dataset/2)+int(cols_standard/2))]
		
	brain_mask_FLAIR=brain_mask_FLAIR.astype(np.int16)
	FLAIR_image=FLAIR_image.astype(np.int16)
	FLAIR_image = FLAIR_image-np.mean(FLAIR_image[brain_mask_FLAIR == 1]) 
	FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
	#
	
	FLAIR_image  = FLAIR_image[..., np.newaxis]
	
	
	return FLAIR_image

def postprocessing(FLAIR_array, pred):
	num_selected_slice = np.shape(FLAIR_array)[0]
	image_rows_Dataset = np.shape(FLAIR_array)[1]
	image_cols_Dataset = np.shape(FLAIR_array)[2]
	original_pred = np.zeros((num_selected_slice,image_rows_Dataset,image_cols_Dataset), np.float32) # Converting to original image-size
	original_pred[:,int((int(image_rows_Dataset)-int(rows_standard))/2):int((int(image_rows_Dataset)+int(rows_standard))/2),int((int(image_cols_Dataset)-int(cols_standard))/2):int((int(image_cols_Dataset)+int(cols_standard))/2)] = pred[:,:,:,0]
	
	return original_pred



###---Here comes the main funtion--------------------------------------------
###---Leave one patient out validation--------------------------------------------


rows_standard = 384
cols_standard = 384
thresh_FLAIR = 70      #simple brain mask, can be changed according to input images
thresh_FLAIR_1 = 70     #to mask the brain
thresh_FLAIR_2 = 1000     #to mask the brain
	
#read the dirs of test data 
inputDir = '/your_path/input_folder/' #Insert your own path

###---dir to save results---------
outputDir = '/your_path/Mask_output/' #Insert your own path

#-------------------------------------------


# Read FLAIR-image#
for dirs in os.listdir(inputDir):
    print('processing images from ',inputDir)
    
    mid_dir = os.path.join(outputDir,str(dirs))  #directory for images
    FLAIR_img = sitk.ReadImage(glob.glob(inputDir+str(dirs)+'/*FLAIR.nii.gz')[0]) # Insert your own FLAIR file name
    img_shape=(rows_standard, cols_standard, 3)
                                                        
    # save meta-data																
    FLAIR_arr = sitk.GetArrayFromImage(FLAIR_img)
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_img)
    FLAIR_spacing = FLAIR_img.GetSpacing()
    FLAIR_origin = FLAIR_img.GetOrigin()
    FLAIR_direction = FLAIR_img.GetDirection()
                        
    # Pre-process test-image
    imgs_test = preprocessing(FLAIR_array)
    print('image size', FLAIR_array.shape)										
                                            
    # Save previous and next slice for a stack-of-slices package
    imgs_trip_test = np.empty((imgs_test.shape[0], imgs_test.shape[1], imgs_test.shape[2], 3))
    for test_slice in np.arange(imgs_test.shape[0]):
        if test_slice == 0:
            imgs_trip_test[test_slice, :, :, 0] = imgs_test[test_slice,...,0]
        else:
            imgs_trip_test[test_slice, :, :, 0] = imgs_test[test_slice-1,...,0]
        imgs_trip_test[test_slice, :, :, 1] = imgs_test[test_slice,...,0]
        if test_slice == imgs_test.shape[0]-1:
            imgs_trip_test[test_slice, :, :, 2] = imgs_test[test_slice,...,0]
        else:
            imgs_trip_test[test_slice, :, :, 2] = imgs_test[test_slice+1,...,0]
    
    # Model definition
    weights_tf = 'weights/tf.h5' #<- file for transfer learning from miccai-model
    imgs_test = imgs_trip_test
    model = get_unet(img_shape,weights_tf,custom_load_func=False)
    model.load_weights(os.path.join('weights/model.h5'))
    
    # Prediction
    pred = model.predict(imgs_test, batch_size=1,verbose=1)        
    
    thres = 0.5
    pred[pred[...,0] > thres] = 1      #thresholding 
    pred[pred[...,0] <= thres] = 0
        
    # Postprocessing						
    original_pred = postprocessing(FLAIR_array, pred)
                        
    # Create ouput directory
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    mid_dir = os.path.join(outputDir,str(dirs))  #directory for images
    if not os.path.exists(mid_dir):
        os.mkdir(mid_dir)
        
    # Save postprocessed segmentation map
    filename_resultImage_nii = os.path.join(mid_dir, 'mask.nii') # create filename
    nii_img = sitk.GetImageFromArray(original_pred)
    nii_img.SetSpacing(FLAIR_spacing)
    nii_img.SetDirection(FLAIR_direction)
    nii_img.SetOrigin(FLAIR_origin)
    sitk.WriteImage(nii_img, filename_resultImage_nii)
                    
    #### removing all lesions with a 3D size less than 8 voxels		####				
    resultImage_vac = getImages(filename_resultImage_nii)
    arrRes = getLesionDetectionNum(resultImage_vac)             # Get array of lesion-voxels
    resUn, resCounts = np.unique(arrRes, return_counts=True)    # Get individual lesions and their sizes
    
    # Remove those lesions with less than 8 voxels
    for les in resUn:
        if les == 0:
            continue
        les_size = resCounts[np.where((resUn==les))] * nii_img.GetSpacing()[0]* nii_img.GetSpacing()[1]*nii_img.GetSpacing()[2] 
        if les_size < 8:
            arrRes[np.where((arrRes==les))] = 0
    resUn, resCounts = np.unique(arrRes, return_counts=True)
                                            
    for vox in original_pred:
        original_pred[np.where((arrRes == 0))] = 0
                                                
    # Save "vacuumed" image																
    filename_vac_resultImage_nii = os.path.join(mid_dir, 'mask_vac.nii')
    nii_img = sitk.GetImageFromArray(original_pred)
    nii_img.SetSpacing(FLAIR_spacing)
    nii_img.SetDirection(FLAIR_direction)
    nii_img.SetOrigin(FLAIR_origin)
    sitk.WriteImage(nii_img, filename_vac_resultImage_nii)
        



