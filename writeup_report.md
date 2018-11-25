# **Behavioral Cloning** 

## Writeup Report (Template from Udacity)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images_for_Visualization/train_log.jpg "training log"
[image2]: ./Images_for_Visualization/model_plot.png "Model Architeture"
[image3]: ./Images_for_Visualization/example_center_image.png "Center Image"
[image4]: ./Images_for_Visualization/example_left_image.png "Left Image"
[image5]: ./Images_for_Visualization/example_right_image.png "Right Image"
[image6]: ./Images_for_Visualization/example_normal_image.png "Normal Image"
[image7]: ./Images_for_Visualization/example_flipped_image.png "Flipped Image"
[image8]: ./Images_for_Visualization/example_VShift_image.png "Vertical Shift Image"
[image9]: ./Images_for_Visualization/example_Brightness_image.png "Random Brightness Image"
[image10]: ./Images_for_Visualization/example_Shadow_image.png "Random Shadow Image"
[image11]: ./Images_for_Visualization/example_resized_image.png "Resized Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results
* video.mp4  generated from autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model (model.py lines 170-199) consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64.  

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 172). 

The model includes a fine-tuning mode (code line 141-146), which allows fine-tuning a trained model when new datasets come in

The model permits early training termination when the validation loss stopped improving for two epochs. The best model with smallest validation loss is saved.

#### 2. Attempts to reduce overfitting in the model

The model contains maxpooling layers in order to reduce overfitting (model.py lines 178 & 182). 

Some other approaches like generating more data were also employed to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 217-219). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 205).

#### 4. Appropriate training data

I tried the data from Udacity. I also collected data from the simulator for two laps. A combination of center images, left and right side images, and their flipped images from these two datasets were used for training and validation.


For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a simple model and then move to complex models

My first step was to use a convolution neural network model similar to the [Lenet model](http://yann.lecun.com/exdb/lenet/).  I have used this model for traffic-sign-classification and it worked well. But the model did not work well for the steering angle prediction. The trained model failed at a sharp left turn. 

I then tried a convolution neural network model similar to the [NVADIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). I thought this model might be appropriate because the NVADIA group has used it for their autonomous cars. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I used a python generator to generate data for training and validation instead of storing them in memory.

For the first-time training, I just used the center images to see how well the model could perform. I found that my trained model still failed at a sharp left turn. 

I then added left and right side images to my image datasets. For the side images, I calculated the steering angles 

```
correction = 0.25
left_angle = center_angle+ correction
right_angle = center_angle-correction

```

and appended them to the steering angle datasets. This not just increased the number of training data, but also helped train the car how to recover from side. I added flipped center images and center angle to the dataset as well, which helped with the left-turn bias.

After this setting, I tried the model on the data from Udacity and my collected data. They both worked well (the car was able to safely drive over one lap in the autonomous mode).

I also tried image augmentation.  I downsized the images for fast training. I then applied random shift(vertically), random brightness and randam shadow sucessively to the training set, not to the validation set. As a result, I got a training loss a little bit higher than validation loss (see below). 

![alt text][image1]

I think if I increased the number of epochs, the training loss would eventually get lower than validation loss,  but the training early terminated because the validation loss stopped improving for two successive epochs. 

The trained model worked well, too. The vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 170-199) consisted of a convolution neural network with the following layers and layer sizes 
_________________________________________________________________
Layer (type)                 Output Shape              Param #

=================================================================
lambda_1 (Lambda)            (None, 70, 224, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 66, 220, 24)       1824
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 33, 110, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 106, 36)       21636
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 53, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 49, 48)        43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 45, 64)         76864
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 43, 64)         36928
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 2, 41, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 5248)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               524900
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11

=================================================================

Total params: 747,899

Trainable params: 747,899

Non-trainable params: 0

Here is a visualization of the architecture 

![alt text][image2]

#### 3. Creation of the Training Set & Training Process


I used the data provided by Udacity and my own collected data (two laps of center lane driving). I randomly shuffled the data set and put 20% of the data into a validation set. 

Each data row includes three images from center, left and right cameras. Here are examples of center, left and right images.

![alt text][image3]
![alt text][image4]
![alt text][image5]

The side images were used for training vehicle how to recover to center, so I did not record the vehicle recovering from the left side and right sides of the road back to center.

To augment the dataset, I also flipped images and angles thinking that this would help with left-turn bias.  For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I tried some other approachs to augment the dataset like random shift (vertically), random brightness and random shadow.  Here are some visualizations.

Random Shift(vertically)

![alt text][image6]
![alt text][image8]

Random Brightness

![alt text][image6]
![alt text][image9]

Random Shadow

![alt text][image6]
![alt text][image10]

These approaches help generalize the model. 

In addition, these approaches were applied only to the training set, not to the validation set. The purpose was to make the training harder.

I downsized all the images in the dataset from  160x320x3 to 70x224x3 for fast training. Here is an example of resized image:

![alt text][image6]
![alt text][image11]


All the resized images were put in a python generator for memory saving. 

I trained the model with input images from the python generator.  I used an adam optimizer so that manually tuning the learning rate wasn't necessary.  The training would terminate early if the validation loss stopped improving for two successive epochs. The best model was saved when the training finished or early terminated. 

Here is a plot of the training history (same with the first image in the markdown)

![alt text][image1]

The validation loss stops decreasing after the 4-th epoch. The training loss keeps decreasing until the training meets the early termination condition. 


