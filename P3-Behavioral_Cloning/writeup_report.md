#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./examples/center_curve_recovery1.jpg "Recovery Image 1"
[image4]: ./examples/center_curve_recovery2.jpg "Recovery Image 2"
[image5]: ./examples/center_curve_recovery3.jpg "Recovery Image 3"
[image6]: ./examples/center-2laps-init.jpg      "Normal Image"
[image7]: ./examples/center-2laps-flipped.jpg	"Flipped Image"
[image8]: ./examples/center-2laps-gray.jpg      "Grayscale Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video_out.mp4 recorded sim output doing couple of laps on track 1

####2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I've implemented model guided with Nvidia paper "End to End Learning for Self-Driving Cars" - https://arxiv.org/abs/1604.07316
Code line numbers are for model.py file.
Model consists of image preprocessing:
 - normalization        - code lines 79
 - cropping             - code lines 80
 - convert to grayscale - code lines 81
 
Convolution layers are based on nvida model and are from line 83 to line 87. The model includes RELU layers to introduce nonlinearity. 


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 91, 93, 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100-103). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track as visible from included video_out.mp4 file.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step was to use a convolution neural network model seen in udacity lectures. I thought this model might be appropriate to quickly get results.

From there on I did couple of iterations playing with different layer configurations, but in the end settled on model solution from nvidia paper, as one giving me best results.

To combat the overfitting, model is using dropout layers on lines 91, 93, 95 of model.py file.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots(curves) where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded nice and smooth turn in curves vehicle had trouble with, and also recovery action to get vehicle to center of road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road on track 1, seen in video_out.mp4 file.

####2. Final Model Architecture

As mentioned before final model architecture is guided with nvidia paper (model.py lines 77-96) consisted of a convolution neural network with the following layers and layer sizes:

Keras model summary:

====================================================================================================                                    [0/112]
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 65, 320, 1)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   624         lambda_2[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]                  
====================================================================================================

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 2 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I trained network was eager to see the results, and was amazed how with small amount of data, there is already significant result. I've noticed that vehicle has problems going through some curvers and then have recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to center of lane. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles providing more 'different' data, and resulting in better generalization. Here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

All images as part of model are converted to grayscale:
![alt text][image6]
![alt_text][image8]

After the collection and augmentation process, I had 15042 number of data points.
Data has been normalized, cropped and converted to grayscale -- those operations are part of keras model in order to better use cpu/gpu resources.

Data points have been randomly shuffled and I put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The 'ideal' number of epochs was 20 as more show overfitting on training data. I used an adam optimizer so that manually training the learning rate wasn't necessary.

##Conclusion:
This was rather fun project. Behavioral cloning is powerfull technique, which can even with relatively small data set can produce astounding results.
Even though I was/am short on time with completing projects till end of Term1, I tried to train and run on both tracks but without satisfactory results, and time constraint prevented me to code/test further. I learned from nvidia paper that one really has to carefully select and balance training data( equal amount of straight, right and left), so that would be my next step in getting model to run on both tracks.
