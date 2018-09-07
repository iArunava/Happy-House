import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse
import warnings

from utils import *
from app import *

from keras.models import load_model

FLAGS = None
WEIGHTS = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--framework',
        type=str,
        default='keras',
        help='The Deep Learning Framework to use for the training or inference')

    parser.add_argument('--path-to-train-set',
        type=str,
        default='./Dataset/train_happy.h5',
        help='Path to the train set in h5 file format.')

    parser.add_argument('--path-to-test-set',
        type=str,
        default='./Dataset/test_happy.h5',
        help='Path to test set in h5 file format.')

    parser.add_argument('--show-shapes-and-numbers',
        type=bool,
        default=False,
        help='If True, then the shapes and number of the train and test are shown')

    parser.add_argument('--inference',
        type=bool,
        default=True,
        help='If True, then the trained model is used to perform inference. \
         Else the model is trained and then used to perform inference. \
         Default: True')

    parser.add_argument('--model-path',
        type=str,
        default='./models/model_84.h5',
        help='Path to the model in h5 format')

    FLAGS, unparsed = parser.parse_known_args()

    # Getting the train and test images
    x_train, y_train, x_test, y_test = load_dataset(FLAGS.path_to_train_set,
                                                    FLAGS.path_to_test_set)

    # Preprocess the images
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Show the image shapes and numbers
    if FLAGS.show_shapes_and_numbers:
        print ('# Training Examples: ', len(x_train))
        print ('# Test Examples: ', len(x_test))
        print ('Shape of x_train: ', x_train.shape)
        print ('Shape of y_train: ', y_train.shape)
        print ('Shape of x_test: ', x_test.shape)
        print ('Shape of y_test: ', y_test.shape)

    # Load the model
    if os.path.isfile(FLAGS.model_path):
        model = load_model(FLAGS.model_path)
    else:
        FLAGS.inference = False

    # Train and save the model
    if not FLAGS.inference:
        # TODO: Add code to perform training
        pass

    # Perform inference
    infer_with_cam(model);
