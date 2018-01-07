import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# function shall be used in lambda of keras model
def conv2gray(img):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(img)

# load log lines into lines list for further processing
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# 20% split of dataset
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# generator function prepares batches, uses all 3 camera images adjusts steering, and augments with flipping
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            lines = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in lines:

                # use images from all 3 cameras
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    imagepath = './data/IMG/' + filename
                    image = cv2.imread(imagepath)
                    images.append(image)
                    measurement = float(line[3])
                # left image, add left steering correction
                    if i == 1:
                        measurement += 0.2
                # right image, add right steering correction
                    elif i == 2:
                        measurement -= 0.2
                    measurements.append(measurement)

            # augment data set performing horizontal image flip with cv2.flip()
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip (images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)


# generator function for training and validation
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# implemented model guided with udacity lectures and Nvidia paper "End to End Learning for Self-Driving Cars"
model = Sequential()
# normalize, crop and convert to grayscale
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Lambda(conv2gray))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

print(model.summary())

model.compile(loss='mse', optimizer=Adam())
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * 6, nb_epoch=20)

model.save('model.h5')
