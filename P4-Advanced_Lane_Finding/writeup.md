##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in camera class implemented in camera.py file. Camera object instantiation is executed in ipython notebook "P4 Advanced Lane Lines" code block 1.

Calibration code is taken from classes and fitted into camera class and is executed with its calibrate method. So bellow default operation explanation is applicable.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

Example of undistort can be found in ipython notebook "P4 Advanced Lane Lines" as a result of execution of block 1.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
In python notebook code block 2 I've loaded single frame from video, to perform pipeline operations, unmodified image from video is shown as result of code block 2 evaluation.
In code block 3 I've run undistort on captured frame, and displayed result
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

I used a combination of color and gradient thresholds to generate a binary image, code was adapted from lecture. Function colgra_pipeline() has been implemented in colgra.py file. In python notebook code block 4 input image is undistorted image if fed as input to colgra pipeline and resulting binary image is displayed.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in class Transformer implemented in file transformer.py. Dunder init method takes src and destination coordinates for transformation. With experimentation I've found coordinates that seem to work well but might need some further tuning. Init function further creates both perspective transform and inverse. I chose the hardcode the source and destination points in the following manner:
```src_coordinates = [[700, 458], [1130, 720], [200, 720], [590, 458]]
   dst_coordinates = [[915, 0], [915, 700], [395, 700], [395, 0]]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Comparison between images can be seen in evaluated code block 5 of ipython notebook.

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane detection is handled via detect method od Lane class which is implemented in lane.py. Detect method heavily reuses code from class, first detecting via windows search, to speed up subsequent detection algorithm is searching within margin of previously detected lines. To maintain detection 'stabillity' and prevent detected lines from 'jumping', I am comparing 'a' constant in polynomial with previously detected, and if it's greater then 0.00025, detected line is discarded. Additionaly I keep recent 5 fits from which I take mean and use it as 'best_fit' to plot line polynomial and in the end lane. 

Method plot() visualize detected lane lines with fitted 2nd order polynomials. In output of of ipython notebook code block 6, we see detected lane lines with 

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lane.py, class lane, method compute_curvature().
I used lecture suggested pixel->m conversion ratios and code.
####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in file lane.py, class Lane, method zel_traka().
Result image can be seen from evaluated code block 7 of iptyhon notebook.

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Due to prolonged illness I am facing this project rather late, and unfortunatelly do not have a lot of time to implement all ideas I have. I reused majority of code from lectures and wrapped all the code snippets with classes/methods. I added mean averaging for last 5 detected lines, to smooth polynomial output. Pipeline will surely fail with lot of sudden changes in light conditions. There are also issues with ambiguity of lane lines during road repairs. Code performance is also not good, I need to do analysis to detect bottlenecks, perhaps perform line detection in batches of images over multiple cpus.
