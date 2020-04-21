import os 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np

#类别
#tfrecords格式文件名
writer= tf.python_io.TFRecordWriter("D:/Dataset/CTDenoisingDataset/Testshortcutset/0.08/008_full_test.tfrecords") 
#the path for store the noised jpg images
class_path1='D:/Dataset/CTDenoisingDataset/Testshortcutset/0.08/val/Noised_full/'
#the path for store the standard jpg images
class_path2='D:/Dataset/CTDenoisingDataset/Testshortcutset/0.08/val/Standard_full/'
count=1
for img_name in os.listdir(class_path1): 
    img_path1=class_path1+img_name #the full path of images
    img_path2=class_path2+img_name
    img1=Image.open(img_path1)
    img2=Image.open(img_path2)
    img_noised=img1.tobytes()#将图片转化为二进制格式
    img_standard=img2.tobytes()#将图片转化为二进制格式
    example = tf.train.Example(features=tf.train.Features(feature={
            #value=[index]决定了图片数据的类型label
            'img_noised': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_noised])),
            'img_standard': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_standard]))
        })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字符串
    count=count+1
    if count % 1000 == 0:
        print("Processed {}.".format(count))
writer.close()
