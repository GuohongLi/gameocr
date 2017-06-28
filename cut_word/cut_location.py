#!/usr/bin/python  
# -*- coding: utf-8 -*- 
import cv2
import sys
import numpy as np

import os
import os.path

def erosion_dialtion(binary):
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    
    opened1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element1)
    cv2.imshow('opened1',opened1)
    cv2.waitKey(0)
    closed1 = cv2.morphologyEx(opened1, cv2.MORPH_CLOSE, element2)
    cv2.imshow('closed1',closed1)
    cv2.waitKey(0)
    opened2 = cv2.morphologyEx(closed1, cv2.MORPH_OPEN, element1)
    cv2.imshow('opened2',opened2)
    cv2.waitKey(0)
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(opened2, element2, iterations = 1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations = 1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2,iterations = 1)
    erosion2 = cv2.erode(dilation2, element1, iterations = 1)
    dilation3 = cv2.dilate(erosion2, element2,iterations = 1)
    cv2.imshow('dilation3',dilation3)
    cv2.waitKey(0)

def binaryimg(img):
    hits,bin_edges = np.histogram(img,bins = 30,range = (0,255),density = False)
    sum_h = 0.0
    for h in hits:
    	sum_h += h
    now = 0.0
    cover = []
    for i in range(len(hits)):
    	now += hits[len(hits) - i -1]
    	cover.insert(0,now*1.0/sum_h)
    bin_index = 0
    for i in range(len(cover)):
    	if cover[i] < 0.10:#20%
    		bin_index = i
    		break
    ret, binary = cv2.threshold(img, bin_edges[bin_index+1], 255, cv2.THRESH_BINARY)
    #binary = cv2.bitwise_not(binary)
    return binary

def preprocess(gray):
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    sobel_x = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
    sobel_y = cv2.Sobel(median, cv2.CV_8U, 0, 1, ksize = 3)
    #gradient = cv2.subtract(sobel_x, sobel_y)
    gradient = cv2.add(sobel_x, sobel_y)
    gradient = cv2.convertScaleAbs(gradient)
    cv2.imshow('gradient',gradient)
    cv2.waitKey(0)
    #blurred = cv2.blur(gradient, (5, 5))
    #cv2.imshow('blurred',blurred)
    #cv2.waitKey(0)
    binary = binaryimg(gradient)
    cv2.imshow('binary',binary)
    cv2.waitKey(0)
    erosion_dialtion(binary)
    



def find_ch_location(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilation = preprocess(gray)
    

if __name__ == '__main__':
    filename = sys.argv[1]
    img = cv2.imread(filename)
    find_ch_location(img)

