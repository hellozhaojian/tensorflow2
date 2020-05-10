## 简介
中文字符的识别引擎

## 如何复现

### 准备数据
1). 准备解压工具
```bash
 sudo apt-get install unalz
```
2). 下载数据
```bash
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

```
3). 将数据解压，准备数据目录
数据目录： **/home/taozw/data/ocr/**, 对应config.py的变量： “data_dir”
测试数据
```bash
(base) taozw@mk-SMBIOS:~/data/ocr/casia/test$ pwd
/home/taozw/data/ocr/casia/test
(base) taozw@mk-SMBIOS:~/data/ocr/casia/test$ ls -trl | head
总用量 1356456
-rw-r--r-- 1 taozw taozw 25011278 5月   4 16:48 1296-c.gnt
-rw-r--r-- 1 taozw taozw 18054409 5月   4 16:48 1256-c.gnt
-rw-r--r-- 1 taozw taozw 26388919 5月   4 16:48 1272-c.gnt
-rw-r--r-- 1 taozw taozw 22473155 5月   4 16:48 1270-c.gnt
-rw-r--r-- 1 taozw taozw 19134320 5月   4 16:48 1254-c.gnt
-rw-r--r-- 1 taozw taozw 19502177 5月   4 16:48 1269-c.gnt
-rw-r--r-- 1 taozw taozw 24030976 5月   4 16:48 1294-c.gnt
-rw-r--r-- 1 taozw taozw 23086784 5月   4 16:48 1250-c.gnt
-rw-r--r-- 1 taozw taozw 26934432 5月   4 16:48 1290-c.gnt

```
训练数据
```bash
(base) taozw@mk-SMBIOS:~/data/ocr/casia/train$ ls -trl | head
总用量 5216044
-rw-r--r-- 1 taozw taozw 15533528 12月  1  2012 1002-c.gnt
-rw-r--r-- 1 taozw taozw 19937251 12月  1  2012 1001-c.gnt
-rw-r--r-- 1 taozw taozw 25413709 12月  1  2012 1003-c.gnt
-rw-r--r-- 1 taozw taozw 16315733 12月  1  2012 1006-c.gnt
-rw-r--r-- 1 taozw taozw 17411395 12月  1  2012 1005-c.gnt
-rw-r--r-- 1 taozw taozw 23634073 12月  1  2012 1004-c.gnt
-rw-r--r-- 1 taozw taozw 24557713 12月  1  2012 1009-c.gnt
-rw-r--r-- 1 taozw taozw 22000745 12月  1  2012 1008-c.gnt
-rw-r--r-- 1 taozw taozw 20949380 12月  1  2012 1007-c.gnt

```

4) 讲数据处理成tfrecord的格式，并获得字符的列表
```bash
cd bin
bash prepare.sh
```
5) 准备训练
准备模型的存储目录 /mnt/workspace/taozw/my_github/tensorflow2/run/chinese_ocr/， 对应config.py中的变量：checkpoint_dir
```bash
cd bin
bash train.sh
```

6) 预测
```bash
cd bin
bash predict.sh
```

## 参考
1. [中科院中文字符识别](https://github.com/hellozhaojian/ocrcn_tf2) 