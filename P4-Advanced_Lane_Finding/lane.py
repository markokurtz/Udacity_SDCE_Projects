import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.recent_5_fits = deque(maxlen=5)
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #maximum allowed 'a' diff
        self.max_a_diff = 0.00025

    def update(self, current_fit, allx, ally):
        self.detected = True
        self.current_fit = current_fit
        if self.best_fit is None:
            self.best_fit = current_fit
        else:
            self.diffs = np.absolute(self.best_fit - self.current_fit)
            a, __, __ = self.diffs
            if a < self.max_a_diff:
                self.recent_5_fits.append(current_fit)
                r5mean = np.mean(self.recent_5_fits, axis=0)
                # self.best_fit = current_fit
                self.best_fit = r5mean
        self.allx = allx
        self.ally = ally


class Lane(object):
    def __init__(self):
        self.lineL = Line()
        self.lineR = Line()


    def detect(self, image):

        binary_warped = image

        if (self.lineR.detected is False) or (self.lineL.detected is False):
            print("doing windows search")
            # to satisfy this copy pasted block
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            if left_fit.any():
                self.lineL.update(left_fit, leftx, lefty)
            else:
                self.lineL.detected = False

            if right_fit.any():
                self.lineR.update(right_fit, rightx, righty)
            else:
                self.lineR.detected = False

        else:
            left_fit = self.lineL.best_fit
            right_fit = self.lineR.best_fit
            print("Doing faster margin search")
            ### after initial consisting line/lane fitting
            # Assume you now have a new warped binary image
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            if left_fit.any():
                self.lineL.update(left_fit, leftx, lefty)
            else:
                self.lineL.detected = False

            if right_fit.any():
                self.lineR.update(right_fit, rightx, righty)
            else:
                self.lineR.detected = False


    def plot(self, warped):
        binary_warped = warped
        left_fit = self.lineL.best_fit
        right_fit = self.lineR.best_fit
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        ## print polynomials
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[self.lineL.ally, self.lineL.allx] = [255, 0, 0]
        out_img[self.lineR.ally, self.lineR.allx] = [0, 0, 255]

        # return out_img

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    def compute_curvature(self, image):
        # Define conversions in x and y from pixels space to meters
        # TODO: parametrize
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        y_eval_l = 720 #np.max(self.lineL.ally)
        y_eval_r = 720 #np.max(self.lineR.ally)
        y_eval = 720

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lineL.ally*ym_per_pix, self.lineL.allx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.lineR.ally*ym_per_pix, self.lineR.allx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_l*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_r*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self.lineL.radius_of_curvature = left_curverad
        self.lineR.radius_of_curvature = right_curverad

        self.offcenter = 0
        left_fitx = self.lineL.best_fit[0]*y_eval**2 + self.lineL.best_fit[1]*y_eval + self.lineL.best_fit[2]
        right_fitx = self.lineR.best_fit[0]*y_eval**2 + self.lineR.best_fit[1]*y_eval + self.lineR.best_fit[2]

        camera_center = image.shape[1] / 2.
        lane_center = (right_fitx + left_fitx) / 2.

        vehicle_lane_offset = camera_center - lane_center
        self.offcenter = vehicle_lane_offset * xm_per_pix


    def zel_traka(self, warped, undist, Mparam):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # TODO: compute this in single place
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = self.lineL.best_fit[0]*ploty**2 + self.lineL.best_fit[1]*ploty + self.lineL.best_fit[2]
        right_fitx = self.lineR.best_fit[0]*ploty**2 + self.lineR.best_fit[1]*ploty + self.lineR.best_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Mparam, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # compute and display lane curvature and lane center vehicle offset
        self.compute_curvature(result)
        result = cv2.putText(result, 'Left curvature:  {:8.2f}m'.format(self.lineL.radius_of_curvature),
                             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 1)
        result = cv2.putText(result, 'Right curvature: {:8.2f}m'.format(self.lineR.radius_of_curvature),
                             (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 1)
        result = cv2.putText(result, 'Lane center off: {:8.2f}m'.format(self.offcenter),
                             (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 1)


        return result
