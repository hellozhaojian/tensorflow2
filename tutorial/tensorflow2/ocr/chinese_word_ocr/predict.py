# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.

from tensorflow2.ocr.chinese_word_ocr.model import build_net_001, build_net_002, build_net_003
from tensorflow2.ocr.chinese_word_ocr.config import ChineseWordOcrConfig
from tensorflow2.ocr.chinese_word_ocr.data_prepare import DataPrepare
import tensorflow as tf
import glob
import os
import logging
import cv2
import numpy as np

class Predictor(object):
    """
    给定一张包含字符的图片，系统预测这个字是什么
    """

    def __init__(self, config: ChineseWordOcrConfig):
        self.config = config
        self.target_size = self.config.height
        self.data_prepare = DataPrepare(config)
        self.index_2_word = {index: word for word, index in self.data_prepare.word_2_index.items()}
        self.num_classes = len(self.data_prepare.word_2_index)
        self.checkpoint_dir = self.config.checkpoint_dir
        self.model = self.get_model()

    def get_model(self):
        model = build_net_003((64, 64, 1), self.num_classes)
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_dir))
        if latest_ckpt:
            start_epoch = int(latest_ckpt.split("-")[1].split(".")[0])
            model.load_weights(latest_ckpt)
            logging.info('model resumed from： {}, at epoch: {}'.format(latest_ckpt, start_epoch))
            return model
        else:
            return None

    def predict(self, image_f):
        ori_img = cv2.imread(image_f)
        img = tf.expand_dims(ori_img[:, :, 0], axis=-1)
        img = tf.image.resize(img, (self.target_size, self.target_size))
        img = (img - 128.)/128.
        img = tf.expand_dims(img, axis=0)
        print(img.shape)
        out = self.model(img).numpy()
        word = self.index_2_word[np.argmax(out[0])]
        print('predict : {} '.format(word))
        cv2.imwrite('assets/pred_{}.png'.format(word), ori_img)


if __name__ == "__main__":
    filename = 'assets/0.png'
    config = ChineseWordOcrConfig()
    predictor = Predictor(config)
    predictor.predict(filename)
