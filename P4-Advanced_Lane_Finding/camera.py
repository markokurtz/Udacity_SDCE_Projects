import numpy as np
import cv2
import glob


class Camera(object):

    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        self.calibarate()

    def calibarate(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Make a list of calibration images
        # TODO: parametrize path to calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

        # get calibration parameters
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = (
            cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None))

    def undistort(self, img):
        # convert to gray
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # undistort image
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist
