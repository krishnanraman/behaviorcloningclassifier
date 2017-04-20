import cv2

import matplotlib.image as mpimg
import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda

import h5py

import glob
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

WIDTH = 64
HEIGHT = 64
CHANNELS = 3
VALIDATION_SET_SIZE = 0.1 #10% of data for validation

classes = [[-1.001,-0.8],
[-0.8,-0.6],
[-0.6,-0.4],
[-0.4,-0.2],
[-0.2,-0.1],
[-0.1,-0.08],
[-0.08,-0.06],
[-0.06,-0.04],
[-0.04,-0.02],
[-0.02,-0.001],
[-0.001,0.001],
[0.001,0.02],
[0.02,0.04],
[0.04,0.06],
[0.06,0.08],
[0.08,0.1],
[0.1,0.2],
[0.2,0.4],
[0.4,0.6],
[0.6,0.8],
[0.8,1.0001]]
CLASSLABELS = len(classes)

#convert a camera angle, ie. a float between -1 to 1, to a class label c
# camera angles between -1 & +1, we split this range into 40 discrete buckets
def cameraToClassLabel(x):
	ans = 0
	for i in range(0,CLASSLABELS):
		boundary = classes[i]
		if (boundary[0] <= x) and (x < boundary[1]):
			ans = i
	return ans

# convert the class label back to camera angle
def classLabelToCamera(i):
	boundary = classes[i]
	return (boundary[0] + boundary[1])/2.0

def mkHistogram(y):
	ylabel = []
	for yi in y:
		ylabel.append(cameraToClassLabel(yi))

	# now count how many labels of each type
	hist = np.histogram(ylabel, np.arange(CLASSLABELS+1))
	counts = hist[0]
	for i in range(0,len(counts)):
		print("|", classes[i], "| = ", counts[i])

def makeModel():
	input_shape = (WIDTH, HEIGHT, 3)

	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
	model.add(Convolution2D(16, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(48, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(CLASSLABELS))
	model.add(Activation('sigmoid'))
	model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
	model.summary()

	return model

def saveModel(model):
	model.save("model.h5")

def preprocessImage(img):
	return cv2.resize(img[60:140,:], (WIDTH, HEIGHT))

def adjust(angle, adjustment):
	a = angle + adjustment
	if a > 1.0:
		a = 1.0
	elif a < -1.0:
		a = -1.0
	else:
		pass
	return a

def makeImageMap():
	# make a map of image name vs class label
	image_camera = {}

	# open the driving log & read all lines
	myfile = open('driving_log.csv') # This is the datafile I generated from my simulations
	lines = myfile.read().split("\n")

	# read line by line & populate the map
	for line in lines:
	    cols = line.split(",")	
	    camera = float(cols[3])     #extract the camera angle
	    image_camera["IMG"+ cols[0].split('IMG')[1]] = camera

	myfile = open('data/driving_log.csv') # This is the datafile provided by udacity
	lines = myfile.read().split("\n")

	# read line by line & populate the map
	for line in lines:
	    cols = line.split(",")
	    camera = float(cols[3]) #extract the camera angle
    
	    # get the image name along with IMG dir
	    image_camera["data/"+ cols[0].strip()]  = camera
	    image_camera["data/"+ cols[1].strip()]  = camera
	    image_camera[ "data/"+ cols[2].strip()] = camera

	print("Read ", len(image_camera), "lines")
	return image_camera

def makeDataset(image_camera):

	# read images from my training set & from udacity's training set
	images = glob.glob("data/IMG/*.jpg")
	images.extend(glob.glob("IMG/*.jpg"))
	numImages = len(images)
	print(numImages, "number of images")

	#populate the images & class labels
	X_sim = []
	y_sim = []
	adjustment = 0.27

	i = numImages - 1
	for name in images:
		if name.find("center") != -1:  # this is a center image
			
			img = preprocessImage(mpimg.imread(name))
			myangle = image_camera[name]
	
			# look if you haved left & right pieces to augment data further
			leftName = name.replace("center", "left")
			rightName = name.replace("center", "right")

			X_sim.append( img )
			y_sim.append( myangle)

			if image_camera.get(leftName) != None:
				# have left & right images

				if myangle > 0.15: # RIGHT TURN
	
					leftImg = preprocessImage(mpimg.imread(leftName))
					leftangle = adjust( myangle, adjustment)  # HARDER RIGHT TURN
			
					X_sim.append( leftImg )
					y_sim.append( leftangle)

				elif myangle < -0.15: #LEFT TURN

					rightImg = preprocessImage(mpimg.imread(rightName))
					rightangle = adjust(myangle, -adjustment) # HARDER LEFT TURN
				
					X_sim.append( rightImg )
					y_sim.append( rightangle)

	return X_sim,y_sim

def main(_):

	# make keras model, train & save
	model = makeModel() # model built based on the number of class labels

	train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
	test_datagen = ImageDataGenerator()

	#first obtain a map of image name mapped to camera angle's class label
	image_camera = makeImageMap()

	# gather the features & labels
	X, y = makeDataset(image_camera)

	#shuffle the training set, split into train & validation using sklearn api
	X_shuff, y_shuff = shuffle(X,y)
	y_encoded = []
	for yi in y_shuff:
		label = cameraToClassLabel(yi)
		arr = np.zeros(CLASSLABELS)
		arr[label] = 1
		y_encoded.append(arr)

	X_train, X_validation, y_train, y_validation = train_test_split(np.array(X_shuff),np.array(y_encoded), test_size = VALIDATION_SET_SIZE, random_state = 100)
	print("Training Set size: ", len(X_train), "Validation set size: ", len(X_validation))

	train_generator = train_datagen.flow(X_train,y_train)
	validation_generator = test_datagen.flow(X_validation,y_validation)

	model.fit_generator(train_generator, nb_epoch=35, samples_per_epoch = len(X_train), validation_data = validation_generator,  nb_val_samples = len(X_validation))
	print('Done Training')
	saveModel(model)

	# make sure model has been saved correctly, by reading back from it & using it for prediction
	model2 = load_model("model.h5")
	print("predictions from model2")
	print(model2.predict(X_validation,batch_size=len(X_validation)))

if __name__ == '__main__':
  tf.app.run()