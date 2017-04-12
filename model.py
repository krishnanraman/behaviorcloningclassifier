import cv2

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import np_utils
import h5py
import math

import glob
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCHSIZE = 128
WIDTH = 80
HEIGHT = 30

CLASSLABELS = 36 # 360/36 =  10 degrees in one class 
CHANNELS = 3


#convert a camera angle, ie. a float between -1 to 1, to a class label c
# camera angles between -1 & +1, we split this range into 40 discrete buckets
def cameraToClassLabel(x):
    return int((x+1)*(CLASSLABELS-1)/2)

# convert the class label back to camera angle
def classLabelToCamera(x):
	return float(2*x/(CLASSLABELS-1) - 1)

def makeModel():
	model = Sequential()
	model.add(Convolution2D(HEIGHT, 3, 3, input_shape=(HEIGHT, WIDTH, CHANNELS)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Activation('relu'))
	model.add(Flatten(input_shape=(HEIGHT,WIDTH, CHANNELS)))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(CLASSLABELS))
	model.add(Activation('softmax'))
	model.summary()
	return model

def saveModel(model):
	model.save("model.h5")

def preprocessImage(img):
	# input is a BGR image from cv2.imrrad
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
	crop_img = img[60:140:, :] # order of params y1:y2, x1:x2
	return cv2.resize(crop_img, (WIDTH, HEIGHT))

def flip(img):
	return cv2.flip( img, 1 ) # 1 means flip around y axis

def randomBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = np.random.uniform(0.2,0.8)
    hsv[:,:,2] = rand*hsv[:,:,2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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
	adjustment = 0.25

	i = numImages - 1
	while (i >= 0):

		name = images[i]
		if name.find("center") != -1:  # this is a center image
			
			img = preprocessImage(cv2.imread(name))
			myangle = image_camera[name]
	
			# look if you haved left & right pieces to augment data further
			leftName = name.replace("center", "left")
			rightName = name.replace("center", "right")

			X_sim.append( img )
			y_sim.append( cameraToClassLabel(myangle))

			if image_camera.get(leftName) != None:
				# have left & right images

				if myangle > 0.15: # RIGHT TURN
	
					leftImg = preprocessImage(cv2.imread(leftName))
					leftangle = adjust( myangle, adjustment)  # HARDER RIGHT TURN
			
					X_sim.append( leftImg )
					y_sim.append( cameraToClassLabel(leftangle))

				elif myangle < -0.15: #LEFT TURN

					rightImg = preprocessImage(cv2.imread(rightName))
					rightangle = adjust(myangle, -adjustment) # HARDER LEFT TURN
				
					X_sim.append( rightImg )
					y_sim.append( cameraToClassLabel(rightangle))


		i -= 1
		
	return np.array(X_sim), np.array(y_sim)


def generator_training(x,y):
	batch_train = np.zeros((BATCHSIZE, HEIGHT, WIDTH, CHANNELS), dtype = np.float32)
	batch_angle = np.zeros((BATCHSIZE), dtype = np.uint8)
	while True:
		data, angle = shuffle(x,y)
		for i in range(BATCHSIZE):
			rand = int(np.random.choice(len(data),1))
			img = x[rand]
			label = angle[rand]
			
			batch_train[i] = img
			batch_angle[i] = label

			# flip the image with 50% probability
			if  np.random.random(1)[0] < 0.5: 	       	
				imgflipped = flip(img)	
				# mirror reflection = flip around y axis, steering angle inverts
				flippedlabel = cameraToClassLabel(-classLabelToCamera(label)) #clever way to invert 
				
				batch_train[i] = imgflipped
				batch_angle[i] = flippedlabel
				#print(label, "=>", flippedlabel)

			# alter brightness of the image with 50% probability
			if  np.random.random(1)[0] < 0.5: 	       	
				batch_train[i] = randomBrightness(batch_train[i])

		yield batch_train, np_utils.to_categorical(batch_angle, CLASSLABELS)  # NOTE we have to convert 128x1 tro 128x36 so labels havea cardinality

# Validation generator: pick random samples. Apply resizing and cropping on chosen samples        
def generator_validation(x,y):
	batch_train = np.zeros((BATCHSIZE, HEIGHT,WIDTH , CHANNELS), dtype = np.float32)
	batch_angle = np.zeros((BATCHSIZE), dtype = np.uint8)
	while True:
		data, angle = shuffle(x,y)
		for i in range(BATCHSIZE):
			rand = int(np.random.choice(len(data),1))
			batch_train[i] = x[rand]
			batch_angle[i] = angle[rand]
		yield batch_train,  np_utils.to_categorical(batch_angle, CLASSLABELS)

def main(_):
	#first obtain a map of image name mapped to camera angle's class label
	image_camera = makeImageMap()

	# gather the features & labels
	X, y = makeDataset(image_camera)
	print(len(np.unique(y)), " unique class labels")

	#shuffle the training set, split into train & validation using sklearn api
	X, y = shuffle(X,y)
	X_train, X_validation, y_train, y_validation = train_test_split(X,y, test_size = 0.10, random_state = 100)
	print("Training Set size: ", len(X_train), "Validation set size: ", len(X_validation))

	trainingGen = generator_training(X_train, y_train)
	validationGen = generator_validation(X_validation, y_validation)

	# make keras model, train & save
	model = makeModel() # model built based on the number of class labels
	model.compile('adam', 'categorical_crossentropy', ['accuracy'])
	model.fit_generator(trainingGen, samples_per_epoch = math.ceil(len(X_train)), nb_epoch=50, validation_data = validationGen, nb_val_samples = len(X_validation))
	print('Done Training')
	saveModel(model)

	# make sure model has been saved correctly, by reading back from it & using it for prediction
	model2 = load_model("model.h5")
	print("predictions from model2")
	print(model2.predict(X_validation[0:10],BATCHSIZE=10))

if __name__ == '__main__':
  tf.app.run()