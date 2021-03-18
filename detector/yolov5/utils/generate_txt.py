# -*- coding: UTF-8 -*-
'''
@author: mengting gu
@contact: 1065504814@qq.com
@time: 2021/3/4 上午11:52
@file: generate_txt.py
@desc: reference https://blog.csdn.net/qqyouhappy/article/details/110451619
'''

import os
import random

trainval_percent = 1
train_percent = 0.9
xmlfilepath = '/home/gmt/datasets/action/prevention/voc/Annotations'
txtsavepath = '/home/gmt/datasets/action/prevention/voc/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('/home/gmt/datasets/action/prevention/voc/ImageSets/Main/trainval.txt', 'w')
ftest = open('/home/gmt/datasets/action/prevention/voc/ImageSets/Main/test.txt', 'w')
ftrain = open('/home/gmt/datasets/action/prevention/voc/ImageSets/Main/train.txt', 'w')
fval = open('/home/gmt/datasets/action/prevention/voc/ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    print(name)
    if int(name[2]) <= 4:
        ftrain.write(name)
    else:
        ftest.write(name)
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    # else:
    #     ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()