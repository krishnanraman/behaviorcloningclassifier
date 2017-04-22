PROCESS
========

My primary goals in this preoject were two-fold
1. Use a classifier, NOT a regressor !
2. Use the fewest possible classes

Reasons:
All the writeups on Behavior Cloning indicated this was a regression problem, to be solved using a powerful DNN regression setup,
such as NVIDIA, comma.ai, VGG etc. 

However, I am convinced that Behavior Cloning can be solved using a very simple (mnist) classifier with very few classes! 
Driving around a circular track successfully involves mostly driving straight (steering angle = 0), occasionally making small adjustments at the turns ( steering angle == nonzero ). The non-zero steering angles are finite. The set of non-zero steering angles a human uses in practice is quite small, such as +5, -5, +15, -15, +25, -25. 
Thus, I figured an autonomous driver would also need to learn a very small set of classes. 

From careful inspection of the udacity dataset (driving_log.csv), it is clear that 

*  majority of the steering angles are 0, followed by a large proportion close to 0, and a very small number close to 1.
*  the 0 degree steering sangle is vastly over-represented ( over 4K samples, in an 8K dataset, ie. > 50% )
*  it is important to distinguish between small turns i.e. -0.1 degree and -0.3 degree must be in separate buckets
*  is is however not crucial to distinguish between large angles i.e. a single bucket will suffice for [-0.4, -1]

Based on these observations, I created 9 distinct buckets/classes -
```
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
```

Bijections
==========

I defined 2 simple bijections, mapping the steering angle <-> class label

 ```
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
```

To convert a steering angle to class label, we simply read off the index of the bucket the angle belongs to.
To convert from the label to an angle, we obtain the midpoint of the bucket. 
We encode the labels as one-hot numpy arrays for purposes of classification.
The bijections will round-trip -
ie. we assert that classLabelToCamera(cameraToclassLabel(x)) == x

Step 1. Data processing => Dictionary
=====================================
We build a dictionary, associating the image name to its class label.
Mostly this involves line by line parsing, splitting by delimiter comma, converting the strings to floats & then moving from camera angle to class label.

Step 2. Dictionary => Training Set
==================================
Given a dictionary of image names & class labels, we treat this exactly like the MNIST problem
We read each image using and examine the label.
The labrels are converted to one-hot numpy arrays using the bijections created above.

Image preprocessing is a single step cropping operation - we remove the sky & the floor portion of the image,
followed by re-scaling the rest of the image into a compact 64by64 size to speed up learning.

Image Augmentation
========================
* The crucial insight is to examine the angle - a nonzero angle can be negated corresponding to a flipped image.
The intuition here is - if a person is steering at 5 degrees to the left, that is equivalent to a -5 degree in the mirror-image.
The mirror image is obtained by flipping the original. 
Negation of camera angle must be following by re-applying the bijection, in order to find the new class label for the negated angle.

* If the steering angle exceeds a threshold, say 0.15, we can use the left image to give us an additional (img,angle) tuple.
The left image is adjusted by an empirical 0.27 degrees, so it simulates a harder left.
Similarly if the steering angle exceeds - 0.15, we can use the right image to give us an additional (img,angle) tuple.

* We store zero-steering and the non-zeros separately. 
After shuffling the dataset, a majority of the zeros are discarded.
This is because over 50% of the dataset consists of zero-steering. 
To normalize the distribution, we only retain 500 tuples ie. one-eighth of the original.

Step 3. Keras Image Generator
=============================
While the training set is quite small (8K images), an image generator vastly increases the number of images available for training.
The Keras image generator produces random images similar to the source images, parametrized by shear, zoom & shift ranges.

Step 4. Defining the Keras Classifier 
=====================================
Very similar to the Keras Traffic Classifier Model - 2 convolution layers with 32 & 64 filters using 3by3 kernels and relu activation. A 2x2 max pooling after the convolution, followed by the standard multi-layer feedforward neural network, and an additional dropout layer.
Because this is multi-category classification, the loss function is 'categorical_crossentropy'.

Step 5. Training Process
========================
We train for 25 epochs with a SGD batchsize of 32. The last few epochs are shown below. 
We achieve a high  > 45% accuracy on the test set.

```
$ $ KERAS_BACKEND=tensorflow  python mymodel.py 

Training Set size:  8410 Validation set size:  935
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 62, 62, 32)        896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 29, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 12544)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                802880    
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 9)                 585       
=================================================================
Total params: 822,857.0
Trainable params: 822,857.0
Non-trainable params: 0.0
_________________________________________________________________
....

Epoch 21/25
262/262 [==============================] - 43s - loss: 1.5004 - acc: 0.4284 - val_loss: 1.3798 - val_acc: 0.4795
Epoch 22/25
262/262 [==============================] - 43s - loss: 1.5090 - acc: 0.4307 - val_loss: 1.4121 - val_acc: 0.4784
Epoch 23/25
262/262 [==============================] - 43s - loss: 1.5006 - acc: 0.4392 - val_loss: 1.4087 - val_acc: 0.4718
Epoch 24/25
262/262 [==============================] - 43s - loss: 1.5150 - acc: 0.4247 - val_loss: 1.3724 - val_acc: 0.4950
Epoch 25/25
262/262 [==============================] - 43s - loss: 1.4896 - acc: 0.4345 - val_loss: 1.4120 - val_acc: 0.4463
Done Training, Model saved to model.h5
```

Step 6. Drive.py
=================
We use the SimplePIController with a fixed speed of 9. The images are read off from input data after base64 decoding. These images must be pre-processed ( ie. cropped & rescaled ) similar to our training set before we use the trained model for prediction.
The model will predict a 9-class numpy array. This one-hot array must then be converted to a steering angle using the bijection. This is a straightforward process of looking up the index of the numpy array with the max value (np.argmax) to obtain the bucket number aka label. Once the bucket/label is known, we find the midpoint of the bucket & that's the steering angle.

Step 7. Results
================
Video: "https://www.youtube.com/watch?v=OSPpME79v2A?autoplay=1"

From the video, it is clear that while the vehicle is able to successfully complete several laps around the circular track, the drive is not "smooth" - this is because the model has learnt exactly 9 steering angles, so the turns are pretty sharp. The car appears to weave quite a bit because of these sharp turns. 

However, if the number of classes is increased to say 20, the ride is much smoother - but training the model requires a much deeper network with more epochs. Ofcourse, in the limit i.e. if the number of classes were increaed to say 1000, it becomes a regression problem!

Takeaway
========
The major takeaway for me was that a difficult problem like behavior cloning, which seems to require a DNN Regressor, can non-intuitively be treated as a very simple classification problem, much like mnist !

The hard part is correctly identifying the class boundaries, picking the fewest number of classes required to successfully navigate the circular terrain, and using a very simple classifier network to successfully solve the problem with this minimal setup.
It is not necessary to use the heavy machinery ( VGG, NVidia architecture, comma.ai ) at all.

