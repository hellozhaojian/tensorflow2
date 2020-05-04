# encoding: utf-8
# Copyright 2019 The DeepNlp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
@file: data_prepare.py
@time: 2020/5/4 3:45 下午
"""

import sys
import struct
import numpy as np
import cv2
import logging
import tensorflow as tf
import glob
import os
import importlib
from tensorflow2.ocr.chinese_word_ocr.config import ChineseWordOcrConfig
from tensorflow2.ocr.chinese_word_ocr.raw_data_reader import RawDataReaderInterface
from typing import Dict
from pathlib import Path


class DataPrepare(object):

    def __init__(self, config: ChineseWordOcrConfig=ChineseWordOcrConfig()):
        self.config = config
        reader_class = getattr(importlib.import_module(config.raw_data_reader_module),
                               config.raw_data_reader_class)
        self.raw_train_reader: RawDataReaderInterface = reader_class(config, "train")
        self.raw_test_reader: RawDataReaderInterface = reader_class(config, "test")
        self.word_2_index: Dict[str, int] = self.raw_test_reader.get_word_index()
        self.train_tf_path = Path(self.config.train_tf_dir)
        self.train_tf_path.mkdir(exist_ok=True)
        self.test_tf_path = Path(self.config.test_tf_dir)
        self.test_tf_path.mkdir(exist_ok=True)

    @staticmethod
    def parse_example(record):
        """
        latest version format
        :param record:
        :return:
        """
        features = tf.io.parse_single_example(record,
                                              features={
                                                  'width':
                                                      tf.io.FixedLenFeature([], tf.int64),
                                                  'height':
                                                      tf.io.FixedLenFeature([], tf.int64),
                                                  'label':
                                                      tf.io.FixedLenFeature([], tf.int64),
                                                  'image':
                                                      tf.io.FixedLenFeature([], tf.string),
                                              })
        img = tf.io.decode_raw(features['image'], out_type=tf.uint8)
        # we can not reshape since it stores with original size
        w = features['width']
        h = features['height']
        img = tf.cast(tf.reshape(img, (w, h)), dtype=tf.float32)
        label = tf.cast(features['label'], tf.int64)
        return {'image': img, 'label': label}

    def load_ds(self):
        input_files = [str(self.train_tf_path/"tfrecord")]
        ds = tf.data.TFRecordDataset(input_files)
        # ds = ds.map(DataPrepare.parse_train_example)
        ds = ds.map(DataPrepare.parse_example)
        return ds

    def load_val_ds(self):
        input_files = [str(self.test_tf_path/"tfrecord")]
        ds = tf.data.TFRecordDataset(input_files)
        ds = ds.map(DataPrepare.parse_example)
        return ds

    def convert_to_tf_record(self):
        """
        将原始的训练，测试文件转换tf_record的格式
        """
        def write_tf_record(tf_path: Path, raw_reader: RawDataReaderInterface):
            tfrecord_f = str(tf_path / "tfrecord")
            logging.info('tfrecord file saved into: {}'.format(tfrecord_f))
            i = 0
            with tf.io.TFRecordWriter(tfrecord_f) as tfrecord_writer:
                for img, label, filename in raw_reader.get_data_iter():
                    try:
                        # why do you need resize?
                        w = img.shape[0]
                        h = img.shape[1]
                        # img = cv2.resize(img, (64, 64))
                        index = self.word_2_index.get(label)
                        if index is None:
                            continue
                        # save img, label as example
                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                            }))
                        tfrecord_writer.write(example.SerializeToString())
                        if i % 5000 == 0:
                            print('solved {} examples. {}: {}'.format(i, label, index))
                        i += 1
                    except Exception as e:
                        logging.error(e)
                        e.with_traceback()
                        continue
            logging.info('done.')

        write_tf_record(self.train_tf_path, self.raw_train_reader)
        write_tf_record(self.test_tf_path, self.raw_test_reader)


if __name__ == "__main__":
    data_prepare = DataPrepare()
    data_prepare.convert_to_tf_record()
    ds = data_prepare.load_ds()
    val_ds = data_prepare.load_val_ds()
    val_ds = val_ds.shuffle(100)
    # val_ds = ds.shuffle(100)
    charactors = {index: word for word, index in  data_prepare.word_2_index.items()}

    is_show_combine = True
    if is_show_combine:
        combined = np.zeros([32*10, 32*20], dtype=np.uint8)
        i = 0
        res = ''
        for data in val_ds.take(200):
            # start training on model...
            img, label = data['image'], data['label']
            img = img.numpy()
            print(img.shape, "=======")
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, (32, 32))
            label = label.numpy()
            label = charactors[label]
            print(label)
            row = i // 20
            col = i % 20
            print(i, col)
            print(row, col)
            combined[row*32: (row+1)*32, col*32: (col+1)*32] = img
            i += 1
            res += label
        cv2.imshow('rr', combined)
        print(res)
        cv2.imwrite('assets/combined.png', combined)
        cv2.waitKey(0)
            # break
    else:
        i = 0
        for data in val_ds.take(36):
            # start training on model...
            img, label = data['image'], data['label']
            img = img.numpy()
            img = np.array(img, dtype=np.uint8)
            print(img.shape)
            # img = cv2.resize(img, (64, 64))
            label = label.numpy()
            label = charactors[label]
            print(label)
            cv2.imshow('rr', img)
            cv2.imwrite('assets/{}.png'.format(i), img)
            i += 1
            cv2.waitKey(0)
            # break
