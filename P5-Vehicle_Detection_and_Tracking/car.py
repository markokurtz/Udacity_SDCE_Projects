from collections import deque

class Car(object):

    def __init__(self):

        self.active = True
        self.previous_heat = False
        self.l5boxes = deque(maxlen=30)
        self.ploted_boxes = deque(maxlen=40)
        self.box_hits_total = 0
        #self.box_hits_frame_delta = 0
        self.frames_alive = 0
        self.pravac = deque(maxlen=40)
        self.pravac_mean_fit = 0
        self.plotbox = 0

    def box_hit(self):
        self.box_hits_total += 1
