#!/usr/bin/python  
# -*- coding: utf-8 -*- 
import cv2
import sys
import numpy as np

import os
import os.path


def load_image(filename):
    img = np.asarray(cv2.imread(filename), 'float32')
    #width = img.shape[0]
    #height = img.shape[1]
    return img
def load_image_2(filename):
    mean_rgb = [123.68, 116.779, 103.939]
    img = np.asarray(cv2.imread(filename), 'float32')
    if len(img.shape) != 3:
        return []
    img2 = np.copy(img)
    img[0, :] = img2[2, :] - mean_rgb[0]
    img[1, :] = img2[1, :] - mean_rgb[1]
    img[2, :] = img2[0, :] - mean_rgb[2]
    return img
def load_image_3(filename):
    img = np.asarray(cv2.imread(filename), 'float32')
    if len(img.shape) != 3:
        return []
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return GrayImage
    #ret,thresh1=cv2.threshold(GrayImage,70,255,cv2.THRESH_BINARY)
    #return thresh1

def load_image_4(filename):
    img = cv2.imread(filename)
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gaussian = cv2.GaussianBlur(GrayImage, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    #ret, binary = cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return GrayImage

if __name__ == '__main__':
    rootdir = sys.argv[1]
    for parent,dirnames,filenames in os.walk(rootdir):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        #for dirname in  dirnames:
        #    print "parent is:" + parent
        #    print  "dirname is" + dirname
        for filename in filenames:
            #print "parent is:" + parent
            #print "filename is:" + filename
            #print "the full name of the file is:" + os.path.join(parent,filename) #输出文件路径信息
            fullpath = os.path.join(parent,filename)
            print fullpath
            img = load_image_3(fullpath)
            if len(img) != 0:
                print img.shape
                #cv2.imwrite(os.path.join(parent,"tmp_"+filename),img[0:520,520:116])
                cv2.imwrite(os.path.join(parent,"tmp_"+filename),img)

