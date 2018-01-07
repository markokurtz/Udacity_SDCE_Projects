import cv2
import numpy as np


from yolotiny import yolotinyNN
yolotiny = yolotinyNN()

class Pipeline(object):

    def __init__(self, tracker, vis_debug):
        self.tracker  = tracker
        self.vis_debug = vis_debug

    def ytpipeline(self, image):

        tracker = self.tracker
        vis_debug =self.vis_debug

        ystart = 350
        ystop = 700

        box_list = yolotiny.pipeline(image, ystart, ystop)
        box_list_orig = box_list

        #print('nr of boxes:', len(box_list))
        tracker.process_boxes(box_list)

        # get list of active boxes
        box_list = tracker.active_boxes()

        draw_img = np.copy(image)


        # draw polynomials
        for car in tracker.cars:
            if (car.frames_alive >= car.l5boxes.maxlen) or (car.box_hits_total >= 40):
            #if car.active and car.frames_alive >= car.l5boxes.maxlen:

                fit = car.pravac_mean_fit

                ploty = np.linspace(0, draw_img.shape[0]-1, draw_img.shape[0] )
                fitx = fit[0]*ploty + fit[1]

                pts = np.array([np.transpose(np.vstack([fitx, ploty]))])

                if vis_debug:
                    cv2.polylines(draw_img,np.int32([pts]),True,(0,255,255))

                # tocke

                # draw main box
                main_box = car.ploted_boxes[-1]
                x1 = int(main_box[0][0])
                y1 = int(main_box[0][1])
                x2 = int(main_box[1][0])
                y2 = int(main_box[1][1])

                cv2.rectangle(draw_img, (x1, y1),(x2, y2),(255,0,0), 4)

                for box in car.ploted_boxes:
                    center_x = np.int((box[0][0] + box[1][0]) / 2)
                    center_y = np.int((box[0][1] + box[1][1]) / 2)

                    #print(center_x, center_y)

                    if vis_debug:
                        cv2.circle(draw_img, (center_x, center_y), 4, (255,0,0))

                    # print box
                    x1 = int(box[0][0])
                    y1 = int(box[0][1])
                    x2 = int(box[1][0])
                    y2 = int(box[1][1])

                    if vis_debug:
                        cv2.rectangle(draw_img, (x1, y1),(x2, y2),(255,0,0), 4)

        #plot all green boxes from detector
        for box in box_list_orig:

            # print box
            x1 = int(box[0][0])
            y1 = int(box[0][1])
            x2 = int(box[1][0])
            y2 = int(box[1][1])

            if vis_debug:
                cv2.rectangle(draw_img, (x1, y1),(x2, y2),(0,255,0), 3)


        return draw_img
        #return heatmap
        #return image
