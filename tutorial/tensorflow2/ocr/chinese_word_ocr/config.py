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
@file: config.py
@time: 2020/5/4 3:48 下午
"""

import sys, os
from pydantic import BaseModel
import yaml
import os

class ChineseWordOcrConfig(BaseModel):

    data_dir = "/mnt/workspace/taozw/data/ocr"

    # #####中科院手写字体###############
    # how to get files below ?
    # wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
    # wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
    # 数据解压工具 sudo apt-get install unalz
    casia_train_dir = os.path.join(data_dir, "casia/train")
    casia_test_dir = os.path.join(data_dir, "casia/test")
    casia_test_tfrecord_dir = os.path.join(data_dir, "casia/test_tf")
    casia_train_tfrecord_dir = os.path.join(data_dir, "casia/train_tf")
    raw_data_reader_module = "tensorflow2.ocr.chinese_word_ocr.raw_data_reader"
    casia_reader_class = "CASIAHandWriteDataReader"

    raw_data_reader_class = casia_reader_class
    train_tf_dir = casia_train_tfrecord_dir
    test_tf_dir = casia_test_tfrecord_dir
    height = 64
    width = 64

    checkpoint_dir="/mnt/workspace/taozw/my_github/tensorflow2/run/chinese_ocr/cn_ocr-{epoch}.ckpt"
    use_keras_fit = True

    @staticmethod
    def load(filename):
        with open(filename, mode="r") as file_h:
            tmp = yaml.safe_load(file_h)
            return ChineseWordOcrConfig(**tmp)

    def dump(self, filename):
        with open(filename, mode="w") as file_h:
            file_h.write(yaml.dump(self.dict()))

