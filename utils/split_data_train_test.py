#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023-11-08 22:17
# @Author  : 陈伟峰
# @Site    : 
# @File    : split_data_train_test.py
# @Software: PyCharm
# ccpd 2019测试集 ，验证集
import os
import shutil

train_image_path = r"G:\data\train\images"
train_labels_path = r"G:\data\train\labels"

crpd_train_path= r"G:\data\train_data1\train_data\CRPD_TRAIN"
ccpd_train_path=r"G:\data\train_data1\train_data\CCPD"
val_path = r"G:\data\val_detect"
val_labels_path = r"G:\data\val\labels"
val_images_path = r"G:\data\val\images"

if __name__ == '__main__':
    for file in os.listdir(crpd_train_path):
        if file.endswith(".jpg"):
            # txt
            label_file_name = file.replace("jpg","txt")
            label_file_path = os.path.join(crpd_train_path,label_file_name)
            is_exist_name = os.path.join(train_image_path,file)
            # 存在
            if os.path.isfile(is_exist_name):
                # 存在即删除
                os.remove(os.path.join(crpd_train_path,file))
                os.remove(label_file_path)
            else:
                if os.path.exists(label_file_path):
                    shutil.move(os.path.join(crpd_train_path,file),is_exist_name)
                    shutil.move(label_file_path,os.path.join(train_labels_path,label_file_name))
    for file in os.listdir(ccpd_train_path):
        if file.endswith(".jpg"):
            # txt
            label_file_name = file.replace("jpg", "txt")
            label_file_path = os.path.join(ccpd_train_path, label_file_name)
            is_exist_name = os.path.join(train_image_path, file)
            # 存在
            if os.path.isfile(is_exist_name):
                # 存在即删除
                os.remove(os.path.join(ccpd_train_path, file))
                os.remove(label_file_path)
            else:
                if os.path.exists(label_file_path):
                    shutil.move(os.path.join(ccpd_train_path, file), is_exist_name)
                    shutil.move(label_file_path, os.path.join(train_labels_path, label_file_name))

    for dir in os.listdir(val_path):
        dir_path = os.path.join(val_path,dir)
        for file in os.listdir(dir_path):
            if file.endswith(".jpg"):
                # txt
                label_file_name = file.replace("jpg", "txt")
                label_file_path = os.path.join(dir_path, label_file_name)
                is_exist_name = os.path.join(val_images_path, file)
                # 存在
                if os.path.isfile(is_exist_name):
                    # 存在即删除
                    os.remove(os.path.join(dir_path, file))
                    os.remove(label_file_path)
                else:
                    if os.path.exists(label_file_path):
                        shutil.move(os.path.join(dir_path, file), is_exist_name)
                        shutil.move(label_file_path, os.path.join(val_labels_path, label_file_name))




