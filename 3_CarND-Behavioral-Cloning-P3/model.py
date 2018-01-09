import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import json


# steering adjustment angle with trail and error
STEERING_ADJUSTMENT=0.28
#filter angle to separate from center images
FILTER_ANGLE = 0.15
EPOCHS=20
BATCH_SIZE=128


model_json_filename = 'model.json'
model_weights_filename = 'model.h5'

def load_split_drivinglog(showbarchart=False):
	col_names = ['center', 'left','right','steering','throttle','brake','speed']
	data = pd.read_csv('./data/driving_log.csv', skiprows=[0], names=col_names)
	print("total data points:" , len(data))

	#split train and validation data using shuffle and train_test_split. 10% as a validation set
	temp_center, temp_steering = shuffle(data.center.tolist(), data.steering.tolist())
	center_img_path, X_valid_img_path, steering, y_valid = train_test_split(temp_center, temp_steering, test_size=0.1, random_state=42)
	if showbarchart:
		visulaize_hist_steering_angles(np.float32(temp_steering))
	#separate left and right from center images and append to left and right with steering adjustment
	straight_img_path, left_img_path, right_img_path = [], [], []
	straight_angle, left_angle, right_angle = [],  [], []
	for  i in steering:
		j = steering.index(i)
		#angles picked randomly with trail and error
		if i > FILTER_ANGLE:
			right_img_path.append(center_img_path[j])
			right_angle.append(i)
		elif i < -FILTER_ANGLE :
			left_img_path.append(center_img_path[j])
			left_angle.append(i)
		else :
			straight_img_path.append(center_img_path[j])
			straight_angle.append(i)

	temp_center_img_path = data.center.tolist()
	temp_steering_angle = data.steering.tolist()
	temp_right_img_path = data.right.tolist()
	temp_left_img_path = data.left.tolist()

	index_left_list = random.sample(range(len(temp_center_img_path)), (len(straight_img_path) - len(left_img_path)))
	index_right_list = random.sample(range(len(temp_center_img_path)), (len(straight_img_path) - len(right_img_path)))

	for i in index_left_list :
	 	if temp_steering_angle[i] < -FILTER_ANGLE:
	 		left_img_path.append(temp_right_img_path[i])
	 		left_angle.append(temp_steering_angle[i] - STEERING_ADJUSTMENT)

	for i in index_right_list :
	 	if temp_steering_angle[i] > FILTER_ANGLE:
	 		right_img_path.append(temp_left_img_path[i])
	 		right_angle.append(temp_steering_angle[i] + STEERING_ADJUSTMENT)

	X_train_img_path = straight_img_path + left_img_path +right_img_path
	y_train = np.float32(straight_angle + left_angle + right_angle)

	return X_train_img_path, y_train, X_valid_img_path, y_valid

def visulaize_hist_steering_angles(angles):
	n_bins = 25
	avg_samples_per_bin = len(angles)/n_bins
	hist, bins = np.histogram(angles, n_bins)
	center = (bins[:-1]+bins[1:])/2
	plt.bar(center, hist, align='center')
	plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')

#random brightness adjustment
def hsv_adjustment(img) :
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	rand = random.uniform(0.3, 1)
	img[:,:,2] = rand*img[:,:,2]
	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	return img

def flip_image_vertical(img, angle):
	img = cv2.flip(img, 1)
	angle = angle*(-1)
	return img, angle

def crop_resize(img):
	img = cv2.resize(img[60:140,:], (64,64))
	return img

def generate_train_data(data, angle, batch_size):
	X_train_batch = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
	y_train_batch = np.zeros((batch_size,), dtype = np.float32)
	#y_train_temp = [];
	#print("type of angle:",angle.dtype)
	while True:
		data, angle = shuffle(data, angle)
		for i in range(batch_size) :
			index = int(np.random.choice(len(data), 1))
			img = plt.imread('./data/' + data[index].strip())
			img = hsv_adjustment(img)
			X_train_batch[i] = crop_resize(img)
			temp = angle[index]*(np.float32(1.0)+np.float32(np.random.uniform(-0.10, 0.10)))
			#print ("index:", i, "angleindex:", angle[index], "temp:", temp, ", angle type:", type(angle[index]), ", temp type:", type(temp))
			#print(type(np.float32(temp, dtype=np.float32)), ", batch type :", type(y_train_batch[i]))
			#y_train_temp.append(temp)
			y_train_batch[i] = temp
			flip_coin = random.randint(0,1)
			if flip_coin == 1:
				#X_train_batch[i], y_train_temp[i] = flip_image_vertical(X_train_batch[i], y_train_temp[i])
				X_train_batch[i], y_train_batch[i] = flip_image_vertical(X_train_batch[i], y_train_batch[i])

		#print("X train length:", len(X_train_batch), "Y train length:", len(y_train_temp))
		#print("X train length:", len(X_train_batch), "Y train length:", len(y_train_batch))
		#yield X_train_batch, np.asarray(y_train_temp, dtype=np.float32)
		yield X_train_batch, y_train_batch

def generate_validation_data(data, angle, batch_size):
	X_valid_batch = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
	y_valid_batch = np.zeros((batch_size,), dtype = np.float32)
	#y_valid_temp = []

	while True:
		data, angle = shuffle(data, angle)
		#print("X valid length:", len(data), "Y valid length:", len(angle))
		for i in range(batch_size):
			index = int(np.random.choice(len(data), 1))
			img = plt.imread('./data/' + data[index].strip())
			X_valid_batch[i] = crop_resize(img)
			#y_valid_temp.append(angle[index])
			y_valid_batch[i] = angle[index]

		#print("X valid length:", len(X_valid_batch), "Y valid length:", len(y_valid_temp))
		#print("X valid length:", len(X_valid_batch), "Y valid length:", len(y_valid_batch))
		#yield X_valid_batch, np.asarray(y_valid_temp, dtype=np.float32)
		yield X_valid_batch, y_valid_batch

def visualize_data(data, angle):
	plt.figure(figsize = (10,8))
	gs1 = gridspec.GridSpec(3,3)
	gs1.update(wspace=0.01, hspace=0.19)
	plt_count = 0

	for i in range(3):
		index = random.randint(1, len(data))
		img = plt.imread('./data/' + data[index].strip())
		ax1 = plt.subplot(gs1[plt_count])
		plt_count += 1

		plt.axis('off')
		ax1.set_aspect('equal')
		#ax1[i].axis('off')
		plt.title("Original image")
		plt.imshow(img)

		brightness_image = hsv_adjustment(img)
		ax2 = plt.subplot(gs1[plt_count])
		plt_count += 1

		#ax2[i+1].axis('off')
		plt.imshow(brightness_image)
		plt.title("brightness changed image")
		plt.axis('off')

		cropped_image = crop_resize(brightness_image)
		ax3 = plt.subplot(gs1[plt_count])
		plt_count += 1

		#ax2[i+1].axis('off')
		plt.imshow(cropped_image)
		plt.title("crop & resize image")
		plt.axis('off')

	plt.show()


def train_and_save_model(X_train, y_train, X_valid, y_valid):
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(64, 64, 3)))
	model.add(Convolution2D(24, 5, 5, border_mode = 'valid', subsample=(2, 2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5, border_mode = 'valid', subsample=(2, 2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5, border_mode = 'valid', subsample=(2, 2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, border_mode = 'valid', W_regularizer = l2(0.001)))
	#model.add(Convolution2D(64, 3, 3, border_mode = 'same', W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode = 'valid', W_regularizer = l2(0.001)))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(80, W_regularizer = l2(0.001)))
	#model.add(Dense(80))
	model.add(Dropout(0.5))
	model.add(Dense(40, W_regularizer = l2(0.001)))
	#model.add(Dense(40))
	model.add(Dropout(0.5))
	model.add(Dense(20, W_regularizer = l2(0.001)))
	#model.add(Dense(20))
	model.add(Dropout(0.5))
	model.add(Dense(10, W_regularizer = l2(0.001)))
	#model.add(Dense(10))
	model.add(Dropout(0.5))
	model.add(Dense(1, W_regularizer = l2(0.001)))
	#model.add(Dense(10))
	model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
	model.summary()

	train_data = generate_train_data(X_train, y_train, BATCH_SIZE)
	validation_data = generate_validation_data(X_valid, y_valid, BATCH_SIZE)
	#X_valid, y_valid = generate_validation_data(X_valid, y_valid, len(X_valid))

	model.fit_generator(train_data, samples_per_epoch=len(X_train), nb_epoch=EPOCHS, validation_data = validation_data, nb_val_samples=len(X_valid), verbose=2)
	#model.fit_generator(train_data, samples_per_epoch=len(X_train), nb_epoch=EPOCHS, validation_data = (X_valid, y_valid), verbose=2)

	print('Training finished')

	with open(model_json_filename, "w") as json_file :
		json_file.write(model.to_json())
	model.save_weights(model_weights_filename)

	print("Model generation successful")

X_train_img_paths, y_train, X_valid_img_paths, y_valid = load_split_drivinglog(True)

#visulaize_hist_steering_angles(y_train)

print("train images :", len(X_train_img_paths))
print("train steering angles:", len(y_train))
print("validation images:", len(X_valid_img_paths))
print("validation steering angles:", len(y_valid))

#train_and_save_model(X_train_img_paths, y_train, X_valid_img_paths, y_valid)

visualize_data(X_train_img_paths, y_train)
