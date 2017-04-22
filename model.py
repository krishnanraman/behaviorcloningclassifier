import sys 
import cv2
import matplotlib.image as mpimg
import tensorflow as tf
import h5py
import glob
import numpy as np

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

WIDTH = 64 #image width, height
HEIGHT = 64
CHANNELS = 3 #number of color channels
VALIDATION_SET_SIZE = 0.1 #10% of data for validation
BATCH_SIZE = 32 # SGD samples per batch

classes = [
[-1.001,-0.4],
[-0.4,-0.2],
[-0.2,-0.1],
[-0.1,-0.001],
[-0.001,0.001],
[0.001,0.1],
[0.1,0.2],
[0.2,0.4],
[0.4,1.001],
]

CLASSLABELS = len(classes)

#convert a camera angle to a class label
def cameraToClassLabel(x):
  label = 0
  for i in range(0,CLASSLABELS):
    boundary = classes[i]
    if (boundary[0] <= x) and (x < boundary[1]):
      label = i

  arr = np.zeros(CLASSLABELS) # encode label as 1-hot numpy array
  arr[label] = 1
  return arr

# convert the class label back to camera angle
def classLabelToCamera(arr):
  label = np.argmax(arr) # index of 1-hot array with the max value
  boundary = classes[label]
  midpoint = (boundary[0] + boundary[1])/2.0
  return midpoint

def makeModel():
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (WIDTH, HEIGHT, 3)))

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(CLASSLABELS, activation='sigmoid'))
  model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
  model.summary()

  return model

def preprocessImage(img):
  return cv2.resize(img[60:140,:], (WIDTH, HEIGHT))

def adjust(angle, adjustment):
  a = angle + adjustment
  if a > 1.0:
    a = 1.0
  elif a < -1.0:
    a = -1.0

  return a

def makeImageMap():
  image_camera = {}  # a dictionary of image name to steering angle

  myfile = open('udacitydataset/driving_log.csv') # This is the datafile provided by udacity
  lines = myfile.read().split("\n")

  for line in lines:
      cols = line.split(",")
      camera = float(cols[3]) #extract the camera angle
    
      # get the image name along with IMG dir
      image_camera["udacitydataset/"+ cols[0].strip()]  = camera
      image_camera["udacitydataset/"+ cols[1].strip()]  = camera
      image_camera[ "udacitydataset/"+ cols[2].strip()] = camera

  return image_camera

def makeDataset(image_camera):

  # read images from my training set & from udacity's training set
  images = glob.glob("udacitydataset/IMG/*.jpg")
  numImages = len(images)

  #populate the images & class labels
  # you want to store the zeros & nonzeros separately
  # the zeros are over-represented, so a majority can be discarded
  X_zero = []
  y_zero = []
  X_nonzero = []
  y_nonzero= []
  adjustment = 0.27

  for name in images:
    if name.find("center") != -1:  # this is a center image
      
      img = preprocessImage(mpimg.imread(name))
      myangle = image_camera[name]
  
      # look if you haved left & right pieces to augment data further
      leftName = name.replace("center", "left")
      rightName = name.replace("center", "right")

      if myangle == 0.0:
        X_zero.append( img )
        y_zero.append(cameraToClassLabel(myangle))

      else:
        # for every non-zero steering angle
        # create a mirror-image by vertical-flipping the image
        # the steering angle will invert as well
        X_nonzero.append( img )
        y_nonzero.append(cameraToClassLabel(myangle))

        X_nonzero.append(cv2.flip(img,1))
        y_nonzero.append(cameraToClassLabel(-myangle))

      if image_camera.get(leftName) != None: # have left & right images

        if myangle > 0.15: # RIGHT TURN
  
          leftImg = preprocessImage(mpimg.imread(leftName))
          leftangle = adjust( myangle, adjustment)  
      
          X_nonzero.append( leftImg )
          y_nonzero.append(cameraToClassLabel(leftangle))

        elif myangle < -0.15: #LEFT TURN

          rightImg = preprocessImage(mpimg.imread(rightName))
          rightangle = adjust(myangle, -adjustment) 
        
          X_nonzero.append( rightImg )
          y_nonzero.append(cameraToClassLabel(rightangle))

  #shuffle the training set, split into train & validation using sklearn api
  X_shuffz, y_shuffz = shuffle(X_zero,y_zero)
  X_shuffnz, y_shuffnz = shuffle(X_nonzero,y_nonzero)

  # zeros are over-represented. Discard most of the zeros. Keep only 500 of the 4000+ zeros
  n = 500
  X = X_shuffz[0:n] + X_shuffnz
  y = y_shuffz[0:n] + y_shuffnz
  return np.array(X),np.array(y)

def main(_):
  # obtain a map of image name mapped to camera angle's class label
  image_camera = makeImageMap()

  # gather the features & labels
  X,y = makeDataset(image_camera)

  # split into train & validation
  X_train, X_validation, y_train, y_validation = train_test_split(X,y, test_size = VALIDATION_SET_SIZE, random_state = 100)
  print("Training Set size: ", len(X_train), "Validation set size: ", len(X_validation))

  model = makeModel() # make keras model
  
  # make image generators to train on randomly generated images
  train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')
  test_datagen = ImageDataGenerator()

  # attach the generators to the sources
  train_generator = train_datagen.flow(X_train,y_train)
  validation_generator = test_datagen.flow(X_validation,y_validation)

  # cmd-line epoch control
  if len(sys.argv) > 1:
    E = int(sys.argv[1])
  else:
    E = 25

  # train the model
  model.fit_generator(train_generator, 
    steps_per_epoch = int(len(X_train)/BATCH_SIZE), 
    epochs=E, 
    verbose=1,
    validation_data =validation_generator, 
    validation_steps= int(len(X_validation)/BATCH_SIZE))

  model.save("model.h5")
  print('Done Training, Model saved to model.h5')

if __name__ == '__main__':
  tf.app.run()
