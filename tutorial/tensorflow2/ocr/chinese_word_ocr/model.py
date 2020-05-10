# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.

from tensorflow.keras import layers
import tensorflow as tf


def build_net_001(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    assert len(input_shape) == 3, "only support 3 channels"
    model = tf.keras.Sequential()
    # 64 * 64 × 1 => 62 * 62 * 32 =>
    model.add(tf.keras.layers.Conv2D(
        input_shape = input_shape, filters=32, kernel_size=(3,3), stride=(1,1),
        padding='valid', activation="relu"
    ))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model


def build_net_002(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model


def build_net_003(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        layers.Flatten(),
        # layers.Dense(1024, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

