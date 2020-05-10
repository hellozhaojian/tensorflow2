# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow2.ocr.chinese_word_ocr.data_prepare import DataPrepare
from tensorflow2.ocr.chinese_word_ocr.config import ChineseWordOcrConfig
from tensorflow2.ocr.chinese_word_ocr.model import build_net_001, build_net_002, build_net_003
import tensorflow_datasets as tfds
import functools
import logging


class Driver(object):

    def __init__(self, config: ChineseWordOcrConfig):
        self.config = config
        self.target_size = self.config.height
        self.data_prepare = DataPrepare(config)
        self.num_classes = len(self.data_prepare.word_2_index)

    def preprocess(self, x):
        """
        minus mean pixel or normalize?
        """
        # original is 64x64, add a channel dim
        x['image'] = tf.expand_dims(x['image'], axis=-1)
        x['image'] = tf.image.resize(x['image'], (self.target_size, self.target_size))
        x['image'] = (x['image'] - 128.) / 128.
        return x['image'], x['label']

    def train(self):
        preprocess = functools.partial(Driver.preprocess, self)
        train_dataset = self.data_prepare.load_ds()
        train_dataset = train_dataset.shuffle(100).map(preprocess).batch(32).repeat()

        val_dataset = self.data_prepare.load_val_ds()
        val_dataset = val_dataset.shuffle(100).map(preprocess).batch(32).repeat()

        for data in train_dataset.take(2):
            print(data)
        model = build_net_003((64, 64, 1), self.num_classes)
        model.summary()
        logging.info('model loaded.')

        start_epoch = 0
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(self.config.checkpoint_dir))
        if latest_ckpt:
            start_epoch = int(latest_ckpt.split("-")[-1].split(".")[0])
            model.load_weights(latest_ckpt)
            logging.info('model resume from {} start at epoch: {}'.format(latest_ckpt, start_epoch))
        else:
            logging.info("passing resume since weights not there. training from scratch")
        if self.config.use_keras_fit:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(self.config.checkpoint_dir,
                                                   save_weight_only=True,
                                                   verbose=1,
                                                   period=500)
            ]
            try:
                model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    validation_steps=1000,
                    epochs=15000,
                    steps_per_epoch=1024,
                    callbacks=callbacks
                )
            except KeyboardInterrupt:
                model.save_weights(self.config.checkpoint_dir.format(epoch=0))
                logging.info("keras model saved")
            model.save_weights(self.config.checkpoint_dir.format(epoch=0))
            model.save(os.path.join(os.path.dirname(self.config.checkpoint_dir), 'cn_ocr.h5'))
        else:
            loss_fn = tf.losses.SparseCategoricalCrossentropy()
            optimizer = tf.optimizers.Adam()
            train_loss = tf.metrics.Mean(name="train_loss")
            train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            for epoch in range(start_epoch, 120):
                try:
                    for batch, data in enumerate(train_dataset):
                        images,labels = data
                        with tf.GradientTape() as tape:
                            predictions = model(images)
                            loss = loss_fn(labels, predictions)
                            gradients = tape.gradient(loss, model.trainable_variables)
                            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                            train_loss(loss)
                            train_accuracy(labels, predictions)
                            if batch % 10 == 0:
                                logging.info('Epoch:{},iter:{}, train_acc:{}'.format(
                                    epoch, batch, train_loss.result(), train_accuracy.result()
                                ))
                except:
                    logging.info("interrupt")
                    model.save_weights(self.config.checkpoint_dir.formmodat(epoch=0))
                    logging.info("model saved into: {}".format(self.config.checkpoint_dir.format(epoch=0)))
                    exit(0)


if __name__ == "__main__":
    config = ChineseWordOcrConfig()
    driver = Driver(config)
    driver.train()
