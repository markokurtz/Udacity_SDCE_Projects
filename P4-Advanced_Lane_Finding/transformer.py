import numpy as np
import cv2

class Transformer(object):

    def __init__(self, src_coordinates = [[700, 458], [1130, 720], [200, 720], [590, 458]],
                 dst_coordinates = [[915, 0], [915, 700], [395, 700], [395, 0]]):

    # def __init__(self, src_coordinates = [[720, 458], [1150, 720], [200, 720], [570, 458]],
    #              dst_coordinates = [[915, 0], [915, 720], [395, 720], [395, 0]]):

    # neko sranje od tocaka s interneta
    # def __init__(self, src_coordinates = [[240, 720], [575, 460], [715, 460], [1150, 720]],
    #              dst_coordinates = [[440, 720], [440, 0], [950, 0], [950, 720]]):

        self.src_coordinates = np.float32(src_coordinates)
        self.dst_coordinates = np.float32(dst_coordinates)

        self.M = cv2.getPerspectiveTransform(self.src_coordinates, self.dst_coordinates)
        self.Minv = cv2.getPerspectiveTransform(self.dst_coordinates, self.src_coordinates)

    def transform(self, image, Mparam):

        img_size = (image.shape[1], image.shape[0])

        warped = cv2.warpPerspective(image, Mparam, img_size, flags=cv2.INTER_LINEAR)

        return warped
