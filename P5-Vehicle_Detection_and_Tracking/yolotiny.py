import argparse
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from yad2k.models.keras_yolo import yolo_eval, yolo_head

class yolotinyNN(object):

    def __init__(self):
        # TODO: parametrize init and review variable usage
        self.model_path = './model_data/tiny-yolo-voc.h5'
        assert self.model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        self.anchors_path = './model_data/tiny-yolo-voc_anchors.txt'
        self.classes_path = './model_data/pascal_classes.txt'
        self.score_threshold=0.4
        self.iou_threshold=0.5

        self.sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        with open(self.classes_path) as f:
            self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        with open(self.anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1, 2)

        self.yolo_model = load_model(self.model_path)

        self.is_fixed_size = True

        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]


        self.yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
        self.yolo_outputs,
        self.input_image_shape,
        score_threshold=self.score_threshold,
        iou_threshold=self.iou_threshold)


    def resizeNormalize(self, image):

        # resize
        if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
            # print(self.model_image_size)
            resized_image = image.resize(tuple(reversed(self.model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')

        # normalize
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        return image_data

    def detector(self, image_data, image):

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # print('Found {} boxes'.format(len(out_boxes)))
        # print(out_boxes, out_classes, out_scores)
        return out_boxes, out_scores, out_classes

    def drawBBoxes(self, image, out_boxes, out_scores, out_classes):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                    for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image


    def pipeline(self, image, ystart, ystop):
        # convert to pil image
        image = Image.fromarray(image)

        noremalizedImage = (self.resizeNormalize(image))

        # detect objects
        out_boxes, out_scores, out_classes = self.detector(noremalizedImage, image)

        # return only bound boxes from car class
        bbox_list = []
        for out_box, out_class in zip(out_boxes.astype(int), out_classes):
            if self.class_names[out_class] == 'car' and out_box[0] >= ystart and out_box[2] <= ystop:
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                bbox_list.append(((out_box[1], out_box[0]),(out_box[3], out_box[2])))


        return bbox_list

        # draw boxes on original image
        # image = self.drawBBoxes(image, out_boxes, out_scores, out_classes)

        # return output image suited for moviepy videofileclip
        # return np.array(image)
