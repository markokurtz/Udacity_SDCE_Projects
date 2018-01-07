import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label


# fix imports

from utils import *
from car import Car
# end fix imports

### TODO: review if you use these methods
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        # heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 2

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
### TODO: end


from collections import deque

class Tracker(object):

    def __init__(self, min_overlap = 0.8, l5boxes_maxlen = 5):
        self.cars = []

        # minimum box overlap to be own box
        self.min_overlap = min_overlap

        self.l5boxes_maxlen = l5boxes_maxlen

    def izracunaj_pravac(self):
        for car in self.cars:
            if car.frames_alive >= car.l5boxes.maxlen:
                tocke_x = []
                tocke_y = []
                for box in car.ploted_boxes:
                    center_x = (box[0][0] + box[1][0]) / 2
                    center_y = (box[0][1] + box[1][1]) / 2

                    tocke_x.append(center_x)
                    tocke_y.append(center_y)

                #print(tocke_y, tocke_x)
                # throw out outliers
                tockey = reject_outliers(tocke_y)
                tockex = reject_outliers(tocke_x)

                lesser_set = min(len(tockey), len(tockex))

                fit_line = np.polyfit(tockey[:lesser_set], tockex[:lesser_set], 1)
                #print(fit_line)

                # TODO: box trajectory line
                if fit_line[0] > 0:
                    car.pravac.append(fit_line)

                np.apply_along_axis(reject_outliers, 0, car.pravac)

                car.pravac_mean_fit = np.median(car.pravac, axis=0)



    def active_boxes(self):
        # list of active boxes to be returned
        boxes = []

        for car in self.cars:
            if car.frames_alive >= car.l5boxes.maxlen:
                mean_avg_box = np.median(car.l5boxes, axis=0)
                car.ploted_boxes.append(mean_avg_box)
                boxes.append(mean_avg_box)

        self.izracunaj_pravac()
        return boxes


    def process_boxes(self, boxes_list):

        for box in boxes_list:

            box_processed = False

            # ako nema auta napravi jedan i ubaci prvu kutiju
            if not any(self.cars):
                #print('Empty cars list: creating one and inserting {} box'.format(box))
                newcar = Car()
                newcar.l5boxes.append(box)
                self.cars.append(newcar)

            # vec imamo nesto auta
            else:
                # iterate over cars and look for overlap
                for carnumber, car in enumerate(self.cars):

                    #print('Working on car number:', carnumber)
                    #print(car.l5boxes)

                    #initial overlaps
                    xOverlap = 0
                    yOverlap = 0

                    #last in queue of 5 boxes for this car
                    car_last_box = car.l5boxes[-1]

                    xOverlap, yOverlap = getBoxOverlap(car_last_box, box)

                    #print('x:{} ,y:{}'.format(xOverlap, yOverlap))

                    # ako imamo overlap veci od mjere, to je vjerojatno nova kutija od tog auta
                    if xOverlap > self.min_overlap and yOverlap > self.min_overlap:
                    #if xOverlap == 0 and yOverlap == 0:
                        #print('Overlap > 0.8 - seems to be mine: inserting {} box to list'.format(box))
                        car.l5boxes.append(box)
                        car.box_hit()
                        car.active = True
                        box_processed = True

                # We have iterated through all cars but box overlap still not found
                # this will be treated as new car
                if not box_processed:
                    #print('Box does not belong to any car: creating new car and inserting {} box'.format(box))
                    newcar = Car()
                    newcar.l5boxes.append(box)
                    self.cars.append(newcar)



        #After processing all boxes from 1 frame do car cleanup
        for carnumber, car in enumerate(self.cars):

            # bump up frames alive counter
            car.frames_alive += 1
            #print('car: {} is alive for: {} frames and hits {}'.format(carnumber, car.frames_alive, car.box_hits_total))

            # if car is alive more then deque len frames and with less then boxes
            # it's probably stale so remove it
            #if car.frames_alive >= car.l5boxes.maxlen and (len(car.l5boxes) < (car.l5boxes.maxlen / 6)):
            if car.frames_alive >= car.l5boxes.maxlen and (len(car.l5boxes) < 2):
                #print('removing car:', carnumber)
                car.active = False
                car.frames_alive = 0
                #del self.cars[carnumber]

            elif car.frames_alive >= car.l5boxes.maxlen:
                # when frame is alive for some time, pop one box from deque left, to prevent stalenes
                car.l5boxes.popleft()


            # TODO: fix this var name buggete, requirement for cars merging
            car_number = carnumber

            # merge duplicate boxes
            for other_car_number, other_car in enumerate(self.cars):

                # check if that's me, and pass that iteration
                if other_car_number == car_number:
                    continue

                else:

                    if len(car.ploted_boxes) and len(other_car.ploted_boxes):
                        xOverlap, yOverlap = getBoxOverlap(car.ploted_boxes[-1], other_car.ploted_boxes[-1])

                        # if boxes are overlapping merge them into one with higher frames alive
                        if xOverlap >= 0.8 and yOverlap >= 0.8:
                            # check which car has lived longer
                            if car.frames_alive >= other_car.frames_alive:
                                # merge other_car to car
                                car.frames_alive += other_car.frames_alive
                                car.l5boxes += other_car.l5boxes
                                # and remove merged other_car from list
                                del self.cars[other_car_number]
                            else:
                                # merge car to other_car
                                other_car.frames_alive += car.frames_alive
                                other_car.l5boxes += car.l5boxes
                                # and remove merged car from list
                                del self.cars[car_number]

