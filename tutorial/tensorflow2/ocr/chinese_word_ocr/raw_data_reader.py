# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.

import numpy as np
import glob
import os
import struct
from typing import Dict
import logging
from tensorflow2.ocr.chinese_word_ocr.config import ChineseWordOcrConfig
import cv2


class RawDataReaderInterface(object):
    """
    原始数据读取的抽象类，子类需要实现函数get_data_iter,其yield image, label, filename
    """
    def get_data_iter(self):
        """
        @return yield image, label, filename
        """
        raise NotImplementedError("this is a abstract function")

    def get_word_index_file(self):
        """
        @return filepath, the format of the filepath is :
                one line one word
        """
        raise NotImplementedError("this is a abstract function")

    def get_word_index(self) -> Dict[str, int]:
        file_name = self.get_word_index_file()
        result = dict()
        if os.path.isfile(file_name):
            print("load from file {}".format(file_name))
            logging.info("load from file {}".format(file_name))
            index = 0
            with open(file_name, "r") as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    result[line.strip()] = index
                    index += 1
                return result
        else:
            print("write file {}".format(file_name))
            f = open(file_name, "w")
            index = 0
            for image, label,_ in self.get_data_iter():
                if label in result:
                    continue
                else:
                    result[label] = index
                    index += 1
                    f.write(label + "\n")
            f.close()
            return result


class CASIAHandWriteDataReader(RawDataReaderInterface):
    """
    http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
    """

    def get_word_index_file(self):
        return os.path.join(self.data_dir, "character.txt")

    def __init__(self, config:ChineseWordOcrConfig, data_type="test"):
        if data_type == "test":
            self.data_dir = config.casia_test_dir
        else:
            self.data_dir = config.casia_train_dir
        self.height = config.height
        self.width = config.width
        print(self.data_dir)

    def get_data_iter(self):
        all_hwdb_gnt_files = glob.glob(os.path.join(self.data_dir, '*.gnt'))
        header_size = 10
        for filename in all_hwdb_gnt_files:

            with open(filename, 'rb') as f:
                while True:
                    header = np.fromfile(f, dtype='uint8', count=header_size)
                    if not header.size:
                        break
                    sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                    tagcode = header[5] + (header[4] << 8)
                    width = header[6] + (header[7] << 8)
                    height = header[8] + (header[9] << 8)
                    if header_size + width * height != sample_size:
                        break
                    image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                    size = (self.width, self.height)
                    shrink = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    label = struct.pack('>H', tagcode).decode('gb2312')
                    label = label.replace('\x00', '')
                    yield shrink, label, filename


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tensorflow2.ocr.chinese_word_ocr.config import ChineseWordOcrConfig
    config = ChineseWordOcrConfig()
    casia_data_reader = CASIAHandWriteDataReader(config)
    count = 0
    for image, tagcode,_ in casia_data_reader.get_data_iter():
        print(tagcode)
        count += 1
        plt.imshow(image)
        plt.show()

        print("here")
        break
    #
    # word_2_index = casia_data_reader.get_word_index()
    #
    # word_2_index_2 = casia_data_reader.get_word_index()
    # wrong = False
    # if len(word_2_index) == len(word_2_index_2):
    #     for item in word_2_index:
    #         if item not in word_2_index_2:
    #             print ("fuck", item)
    #             wrong = True
    #             break
    # if wrong:
    #     print("fuck")
    # else:
    #     print("ok")
    # print(len(word_2_index))

