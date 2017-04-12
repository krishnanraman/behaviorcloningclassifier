RESULTS
=========
$ python mymodel.py 
Shape of Y-hot: (9688, 50)
Using TensorFlow backend.
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 158, 318, 160) 4480        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 79, 159, 160)  0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 79, 159, 160)  0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2009760)       0           activation_1[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           257249408   flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 128)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            6450        activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 50)            0           dense_2[0][0]                    
====================================================================================================
Total params: 257,260,338
Trainable params: 257,260,338
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 7750 samples, validate on 1938 samples
Epoch 1/2
7750/7750 [==============================] - 351s - loss: 8.0569 - acc: 0.4879 - val_loss: 7.9925 - val_acc: 0.5041
Epoch 2/2
7750/7750 [==============================] - 347s - loss: 8.1381 - acc: 0.4951 - val_loss: 7.9925 - val_acc: 0.5041
Validation Acc 0.504127966915
Epoch 1/2
9688/9688 [==============================] - 532s - loss: 8.1090 - acc: 0.4969     
Epoch 2/2
5824/9688 [=================>............] - ETA: 211s - loss: 8.2251 - acc: 0.4897

PROCESS
========

Step 1. Drove around the track several times, staying mostly in the center but also swerving right & left, finally obtained 4800 images and their camera angles.

Step 2. From inspection, it is clear the camera angles lie between -1 & +1. A lot of the angles are also identical, due to mostly heading straight.

Step 3. The camera angle range [-1,1] can be discretized into 100 buckets. This makes it a classification problem with 100 labels, which is easier to solve than trying to predict a floating point angle.

I defined 2 simple bijections, mapping the camera angle <-> class label
def cameraToClassLabel(x):
    return int((x+1)*(100-1)/2)

# convert the class label back to camera angle
def classLabelToCamera(x):
	return float(2*x/99 - 1)

I wrote tests to ensure the bijections are sound.
The bijections must round-trip -
ie. we assert that classLabelToCamera(cameraToclassLabel(x)) == x

Step 4. Data processing => Dictionary
We build a dictionary, associating the image name to its class label.
Mostly this involves line by line parsing, splitting by delimiter comma, converting the strings to floats & then moving from camera angle to class label.
Key Step: cameraClass = cameraToClassLabel(float(camera))

Step 5. Dictionary => Training Set
Given a dictionary of image names & class labels, we treat this like the MNIST problem albeit with many more labels.
We read each image using plt and examine the label.
The crucial insight is to examine the angle - a nonzero angle can be negated corresponding to a flipped image.
The intuition here is - if a person is steering at 5 degrees to the left, that is equivalent to a -5 degree in the mirror-image.
The mirror image is obtained by flipping the original. Negation of camera angle must be following by re-applying the bijection, in order to find the new class label for the negated angle.
While we expect 100 classes in theory, we get fewer because most of the angles are simply 0 ( driving straight )

Step 6. Preprocess
We shuffle the training data & obtain a one-hot representation of our labels.

Step 7. Defining the model
Am mostly sticking with the Keras Traffic Classifier Model. 
The difference is chiefly in the number of classes & input shapes (160x320 images)
A convolution layer with 160 filters, 3x3 kernel & valid padding has been used.
A 2x2 max pooling after the convolution, followed by the standard multi-layer feedforward neural network, with the number of classes dynamically obtained from the shape of the training set. The shape tells us how many class labels there were.

We obtain 49% accuracy quite easily, within 2 epocs & using fewer than 10K images.
Accuracy can be increased much further with larger corpus, more epochs & changing the topology of the model.
Due to running on a CPU, we restrict ourselves to the current results.




# behaviorcloningclassifier
