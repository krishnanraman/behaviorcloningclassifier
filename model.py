import cv2
import sys
import matplotlib.image as mpimg
import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
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

classes = [
[-1.001,-0.8],
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
[0.8,1.001],
]
CLASSLABELS = len(classes)

#convert a camera angle to a class label
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

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
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

def flip(image, angle):
  new_image = cv2.flip(image,1)
  new_angle = angle*(-1)
  return new_image, new_angle

def makeImageMap():
  # make a map of image name vs class label
  image_camera = {}

  myfile = open('udacitydataset/driving_log.csv') # This is the datafile provided by udacity
  lines = myfile.read().split("\n")

  # read line by line & populate the map
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
        y_zero.append( myangle)

      else:
        X_nonzero.append( img )
        y_nonzero.append( myangle)
        i2,a2 = flip(img,myangle)
        X_nonzero.append( i2 )
        y_nonzero.append( a2 )

      if image_camera.get(leftName) != None:
        # have left & right images

        if myangle > 0.15: # RIGHT TURN
  
          leftImg = preprocessImage(mpimg.imread(leftName))
          leftangle = adjust( myangle, adjustment)  # HARDER RIGHT TURN
      
          X_nonzero.append( leftImg )
          y_nonzero.append( leftangle)

        elif myangle < -0.15: #LEFT TURN

          rightImg = preprocessImage(mpimg.imread(rightName))
          rightangle = adjust(myangle, -adjustment) # HARDER LEFT TURN
        
          X_nonzero.append( rightImg )
          y_nonzero.append( rightangle)

    #shuffle the training set, split into train & validation using sklearn api
  X_shuffz, y_shuffz = shuffle(X_zero,y_zero)
  X_shuffnz, y_shuffnz = shuffle(X_nonzero,y_nonzero)

  y_encodedz = []
  for yi in y_shuffz:
    label = cameraToClassLabel(yi)
    arr = np.zeros(CLASSLABELS)
    arr[label] = 1
    y_encodedz.append(arr)

  y_encodednz = []
  for yi in y_shuffnz:
    label = cameraToClassLabel(yi)
    arr = np.zeros(CLASSLABELS)
    arr[label] = 1
    y_encodednz.append(arr)


  return np.array(X_shuffz),np.array(y_encodedz), np.array(X_shuffnz),np.array(y_encodednz)

def main(_):

  # make keras model, train & save
  model = makeModel() # model built based on the number of class labels

  train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')
  test_datagen = ImageDataGenerator()

  #first obtain a map of image name mapped to camera angle's class label
  image_camera = makeImageMap()

  # gather the features & labels
  Xz, yz, Xnz, ynz = makeDataset(image_camera)
  X = []
  y = []

  # zeros are over-represented. Throw most of zeros out. Keep only 500 of the 4000+ zeros
  for i in range(0,500):
    X.append(Xz[i])
    y.append(yz[i])
    
  # keep all the nonzeros
  for i in range(0,len(Xnz)):
    X.append(Xnz[i])
    y.append(ynz[i])

  X,y = shuffle(np.array(X), np.array(y))

  X_train, X_validation, y_train, y_validation = train_test_split(X,y, test_size = VALIDATION_SET_SIZE, random_state = 100)
  print("Training Set size: ", len(X_train), "Validation set size: ", len(X_validation))

  train_generator = train_datagen.flow(X_train,y_train)
  validation_generator = test_datagen.flow(X_validation,y_validation)

  if len(sys.argv) > 1:
    E = int(sys.argv[1])
  else:
    E = 25

  model.fit_generator(train_generator, steps_per_epoch = int(len(X_train)/32), epochs=E, verbose=1,validation_data =validation_generator, validation_steps= int(len(X_validation)/32))
  print('Done Training')
  saveModel(model)

  # make sure model has been saved correctly, by reading back from it & using it for prediction
  model2 = load_model("model.h5")
  print("predictions from model2")
  print(model2.predict(X_validation,batch_size=int(len(X_validation)/128)))

if __name__ == '__main__':
  tf.app.run()
