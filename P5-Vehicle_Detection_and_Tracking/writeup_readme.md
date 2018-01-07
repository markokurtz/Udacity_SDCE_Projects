##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (IPython notebook refers to P5_Vehicle_Detection.ipynb file for rest of this doc) using class HogSVMClassifier implementied in file hog_classify.py 

In dunder init function I load all the `cars` and `notcars` images. Code in IPython notebook block 1 prints number of samples for both classes, and display example images from both datasets. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Examples for both datasets can be found in respective blocks 3 and 4 of IPython notebook.

Examples are using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`:

####2. Explain how you settled on your final choice of HOG parameters.

I tried some combination of paramaeters but in the end decided to go with values from udacity lectures.
TODO: upisi parametre

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in using method train_svm() in class HogSVMClassifier in file hog_classify.py, algorithm/code for training is taken from udacity lectures and was adapted to OO. Training statistics can be seen as evulation of IPython notebook's code block 5.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Code for windows search is implemented in windows_search.py file in function find_cars(), it's predominantly code from udacity class - using Hog Sub-sampling Window Search  I chose parameters through experimentation process.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched YCrCb 3-channel HOG features in the feature vector, which provided a nice result.  Result image can be seen in ipython notebook code cell

As for optimization of detection pipeline, after some experimentation that did not provide noticable improvements, I investigated Yolo v2 NN (https://pjreddie.com/darknet/yolo/), Tiny yolo to be exact. Python implementation of yolotiny was downloaded from (https://github.com/allanzelener/YAD2K) I wrote some glue code in yolotiny.py file. Video pipeline performance with yolotiny is really good as can be seen from ipython cell number 9, where it processed at speed 35.11it/s.


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented tracking solution in tracker.py file Tracker keeps list of Cars, activate/show them on new frame that reach certain lifetime.
To reduce number of detected boxes, I used overlapping detection, merging similar boxes. In order to display bounding boxes, for cars who's detected boxes are intermitently flashing, I used TTL-ish concept. To remove false positives, I limited search to reasonable image area, and used `car` lifecycle vitality to display only detections that are more 'stable' in recent detections.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Yolo Tiny is really performing good, however my tracker/drawing pipeline really need more work to make it more clean and robust. I implemented line polynomial fitting on centers of detected/accepted boxes per each car. Idea behind it is to deduce speed of vehicle movement across detected polynomial line, and to predict center for next bounding box in case vehicle for some reason was not detected properly in current frame. Further improvement work would be on improving code structure/style and utilizing above mentioned 'next box prediction'.



