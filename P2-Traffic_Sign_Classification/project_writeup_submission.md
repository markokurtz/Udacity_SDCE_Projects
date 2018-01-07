#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cells 3-8 of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
Cell 3 defines functions to visualise dataset with 12 random images per signclass.
Cell 4 uses above menioned functions to visualise training set.
Cell 5 uses above menioned functions to visualise validation set.
Cell 6 uses above menioned functions to visualise test set.

Idea in Cell 7 is to show sample frequency per sign class for all 3 given datasets; train, valid and test.
Data shows imballance of frequency which should be recitified with data augmentation later.

In Cell 8 I am exploring ratio of samples available in training set against samples in validation and test sets.
There is such great imbalance as there is with sample frequency per sign class as seen in Cell 7



###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in code cells 12-15 of the IPython notebook.

As a first step, I decided to convert the images to grayscale because I wanted to reduce complexity, and speed up training.

Example of original and preprocesed images can be seen in code cell 28 of the IPython notebook

As a last step, I normalized the image data because it facilitates numerical stability of model.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Since provided data traffic-sign-data.zip file had 3 pickle files, for training, validation and test datasets respectively, there was no need to split data into sets (or at least presence of those 3 files led me to bealive that it's so)

Since there is a disbalanced number of samples per signclasses (as seen on plots of code cell 7), I've used data augmentation techinques to increase and equalize number of samples across all sign classes. Data augmentation was implemented in code cell 9 of the IPython notebook. Three methods were used to augment datasets; rotation, translation and shear. Parameters used with tranformative methods are in 'lower' non-aggressive range in order to keep space oriented information (i.e. too much rotation can change meaning of sign). They were applied randomly on each image of processing dataset, for such number of iterations per sign class that produces double of samples of largest signclass for each signclass. I have applied augmentation to training and validation sets.

Results of augmentation can be seen on plots in code cell 11. Random 10 augmented samples can be seen in code cell 10.

My final number of images in train dataset: 172860 and in validation dataset: 20640

The difference between the original data set and the augmented data set is almost 5x (4,9 and 4,6) more data in augmented training and validation datasets. Additional augmentation method would be, to determine which signclasses are safe for vertical and/or horizontal mirroring, and apply to them respecitvely.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 19th cell of the ipython notebook. 

My final model is Lenet used in class, modified to take in 32,32,1 inputs and with dropouts on FC layers. 

Here is model overview by layers:
- input (32x32x1 - grayscale preprocessed)
- conv layer (5x5x1, stride 1 to 28x28x6)
- relu
- maxpool (2x2 kernel, 2x2 stride) 
- conv layer ()
- relu
- maxpool (2x2 kernel, 2x2 stride)
- flatten (400 nodes)
- fully connected layer (400 -> 200 nodes)
- relu
- dropout
- fully connected layer (120 -> 84 nodes)
- relu
- dropout
- fully connected layer/Output (84 -> 43)
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 24th cell of the ipython notebook. 

To train the model, I used slightly modified lenet arch. mentioned above with adam optimizer and dropout keeping probabilities of 0.5. Batch size was 128 with 1000 Epochs. Learning rate was 0.001. mu = 0  and sigma = 0.1

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 23rd and 24th cell of the Ipython notebook.

My final model results were:

* training set accuracy of 0.989
* validation set accuracy of 0.927 
* test set accuracy of 0.943


If a well known architecture was chosen:
* What architecture was chosen?
  Lenet.
* Why did you believe it would be relevant to the traffic sign application?
  No particular belif, it was fastest to start with. Given more time to implement, test and train. I would look at custom nets with i.e. 3convs one FC and softmax layer. Also would like to implement Vivek Yadav's net (https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad#.dcizjnopo) and Mrinal Haloi's Spatial Transformer net (https://arxiv.org/pdf/1511.02992.pdf) 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

22 German traffic signs that my wife took pics of while we drove up to local shopping center can be found in code cell 28.
###Updated on resub request
In Code Cell 36 of zipped ipython notebook Traffic_Sign_Classifier_updated.ipynb, I've added brief description of how image quality may effect model performance.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.


The model was able to correctly guess with an accuracy of 91%. This compares somewhat favorably to the accuracy on the test set of 94%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31st cell of the Ipython notebook.

Instead of going image through image I will make general conclusion and will adress single interesting case.
For majority of signclasses model expressed strong belif, biggest uncertainty it has is with speed limit signs.
It seems model always recognizes it's speed limit type of sign, however extracting proper numeral speed value is problematic.

Most interesting for me was last sign in test. Sign class is priority road, however there are some round shadows/structures casted from sign behind it, that seem to represent sign to model as one of round type. Since model strongly corelates round signs with borders, to be speed limit or no vehicles, those turned out to be top 3 predicted sign classes.

This was interesting excersise, however due to lack of time I wasn't able to try out too many things, for sure I will revisit this later on.