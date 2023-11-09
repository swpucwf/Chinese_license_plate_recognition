#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023-11-09 21:48
# @Author  : 陈伟峰
# @Site    : 
# @File    : vertify_datasets.py
# @Software: PyCharm
import os

images_path  = r"G:\data\train\images"
labels_path = r"G:\data\train\labels"

for file in os.listdir(images_path):
    if not file.endswith("jpg"):
        os.remove(os.path.join(images_path,file))
    else:
        label_name = file.replace("jpg","txt")
        label_path = os.path.join(labels_path,label_name)
        if os.path.exists(label_path):
            continue
        else:
            os.remove(os.path.join(images_path, file))
