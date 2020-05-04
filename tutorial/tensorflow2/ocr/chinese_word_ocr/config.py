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


class ChineseWordOcrConfig(BaseModel):

    data_dir = "/Users/alchemy_taotaox/Desktop/mygithub/ocrcn_tf2/dataset"

    #
    # how to get files below ?
    # wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
    # wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
    train_zip_file = "HWDB1.1trn_gnt.zip"
    test_zip_file = "HWDB1.1tst_gnt.zip"

    @staticmethod
    def load(filename):
        with open(filename, mode="r") as file_h:
            tmp = yaml.safe_load(file_h)
            return ChineseWordOcrConfig(**tmp)

    def dump(self, filename):
        with open(filename, mode="w") as file_h:
            file_h.write(yaml.dump(self.dict()))

