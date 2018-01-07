import numpy as np
import cv2
from skimage.feature import hog

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def show_hog_features( image, colormap, orientations = 9, pixels_per_cell = 8, cells_per_block = 2 ):

    image = mpimg.imread(hogsvmclassifier.cars[0])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    YCrCb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    fig = plt.figure()
    for i, channel in enumerate(range(YCrCb_image.shape[2])):
        __, hog_image = hogsvmclassifier.get_hog_features( img = YCrCb_image[:,:,channel], orient = orientations,
                                    pix_per_cell = pixels_per_cell, cell_per_block = cells_per_block,
                                    vis = True)
        # YCrCb channels as grayscale
        fig.add_subplot(2,YCrCb_image.shape[2],i + 1)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        plt.axis('off')
        ax1.set_title('Channel: {}'.format(i), fontsize=50)
        plt.imshow(YCrCb_image[:,:,channel], cmap = 'gray')

        # Hog features per YCrCb channel
        fig.add_subplot(1,YCrCb_image.shape[2],i + 1)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        plt.axis('off')
        ax1.set_title('Channel: {}'.format(i), fontsize=50)
        plt.imshow(hog_image)

    plt.show()

def getOverlap(a, b):
    # get range 'size' for each 'dimension'
    range_a = a[1] - a[0]
    range_b = b[1] - b[0]
    minimum_range = min(range_a, range_b)

    # calculate overlap betwen them
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))

    normalized_overlap = overlap / minimum_range

    return normalized_overlap

def getBoxOverlap(box1, box2):
    car_last_box = box1
    box = box2

    xOverlap = getOverlap((car_last_box[0][0], car_last_box[1][0]),
                            (box[0][0], box[1][0]))
    yOverlap = getOverlap((car_last_box[0][1], car_last_box[1][1]),
                            (box[0][1], box[1][1]))

    return xOverlap, yOverlap


def reject_outliers(data, m=2):

    arr = np.int32(data)
    elements = np.array(arr)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    final_list = [x for x in arr if (x > mean - 2 * sd)]
    final_list = [x for x in final_list if (x < mean + 2 * sd)]

    fl_median = [np.median(final_list, axis=0)]

    if any(final_list):
        if len(final_list) == len(data):
            return final_list
        else:
            # add some fl_medians to have symetry
            diff = len(data) - len(final_list)
            return final_list + (diff * fl_median)
    else:
        return data
