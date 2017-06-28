# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
class DetectScorePlate(object):
	def preprocess( self, gray):
		# # 直方图均衡化
		# equ = cv2.equalizeHist(gray)
		# 高斯平滑
		gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
		# 中值滤波
		median = cv2.medianBlur(gaussian, 5)
		# Sobel算子，X方向求梯度
		sobel_x = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
		sobel_y = cv2.Sobel(median, cv2.CV_8U, 0, 1, ksize = 3)
		#gradient = cv2.subtract(sobel_x, sobel_y)
		#gradient = sobel_x
		gradient = cv2.add(sobel_x, sobel_y)
		gradient = cv2.convertScaleAbs(gradient)
		#cv2.imshow('gradient',gradient)
		#cv2.waitKey(0)
		blurred = cv2.blur(gradient, (9, 9))
		#cv2.imshow('blurred',blurred)
		#cv2.waitKey(0)
		# 二值化
		avreage = min(np.mean(blurred)*1.5, 255)
		#hits,bin_edges = np.histogram(blurred,bins = 30,range = (0,255),density = True)
		hits,bin_edges = np.histogram(blurred,bins = 30,range = (0,255),density = False)
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
		ret, binary = cv2.threshold(blurred, bin_edges[bin_index+1], 255, cv2.THRESH_BINARY)
		cv2.imshow('binary',binary)
		cv2.waitKey(0)
		# 膨胀和腐蚀操作的核函数
		element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
		element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
	
		opened1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element1)
		#cv2.imshow('opened1',opened1)
		#cv2.waitKey(0)
		closed1 = cv2.morphologyEx(opened1, cv2.MORPH_CLOSE, element2)
		#cv2.imshow('closed1',closed1)
		#cv2.waitKey(0)
		opened2 = cv2.morphologyEx(closed1, cv2.MORPH_OPEN, element1)
		cv2.imshow('opened2',opened2)
		cv2.waitKey(0)
		# 膨胀一次，让轮廓突出
		dilation = cv2.dilate(opened2, element2, iterations = 1)
		# 腐蚀一次，去掉细节
		erosion = cv2.erode(dilation, element1, iterations = 1)
		# 再次膨胀，让轮廓明显一些
		dilation2 = cv2.dilate(erosion, element2,iterations = 3)
		erosion2 = cv2.erode(dilation2, element1, iterations = 1)
		dilation3 = cv2.dilate(erosion2, element2,iterations = 4)
		cv2.imshow('dilation3',dilation3)
		cv2.waitKey(0)
		return dilation3
	
	def findPlateRegion( self, img, rawimg):
		img_height = img.shape[0]        
		img_width = img.shape[1]        
		region = []
		# 查找轮廓
		contours,hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
		# 筛选面积小的
		for i in range(len(contours)):
			cnt = contours[i]
			# 轮廓近似，作用很小
			'''
			#epsilon = 0.001 * cv2.arcLength(cnt,True)
			epsilon = 0.03 * cv2.arcLength(cnt,True)
			approx = cv2.approxPolyDP(cnt, epsilon, True)
			'''
			# 计算该轮廓的面积
			area = cv2.contourArea(cnt)
	
			# 面积小的都筛选掉
			if (area < 0.3*img_height*0.5*img_width):
				continue
	
			# 找到最小的矩形，该矩形可能有方向
			rect = cv2.minAreaRect(cnt)
			#print "rect is: "
			#print rect
	
			# box是四个点的坐标
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
	                #print box
	
			# 计算高和宽
			height = abs(box[0][1] - box[2][1])
			width = abs(box[0][0] - box[2][0])
			ratio1 =float(width) / float(height)
			ratio2 =float(height) / float(width)
			#print ratio1, ratio2
			ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
			xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
			ys_sorted_index = np.argsort(ys)
			xs_sorted_index = np.argsort(xs)
	
			x1 = box[xs_sorted_index[0], 0]
			x2 = box[xs_sorted_index[3], 0]
	
			y1 = box[ys_sorted_index[0], 1]
			y2 = box[ys_sorted_index[3], 1]
			if width>0.5*img_width and height>0.3*img_height and area*1.0/(height*width)>0.75 \
				and 0.2*img_height<y1<0.5*img_height<y2<0.8*img_height\
				and 0.1*img_width<x1<0.5*img_width<x2<0.9*img_width:
				if (ratio1 > 2 and ratio1<=5) or (ratio2 >= 0.2 and ratio2 <= 5):
					img_cnt = rawimg.copy()
					cv2.drawContours(img_cnt, [cnt], 0, (0, 255, 0), 2)
					cv2.imshow('img_cnt',img_cnt)
					cv2.waitKey(0)
					print height,width
					print x1,x2,y1,y2
					cv2.drawContours(rawimg, [box], 0, (0, 255, 0), 2)
					cv2.imshow('box',rawimg)
					cv2.waitKey(0)
					region.append(box)
	
		return region
	
	def detect( self, img):
		# 转化成灰度图
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# 形态学变换的预处理
		dilation = self.preprocess(gray)
	
		# 查找车牌区域
		img_org1 = img.copy()
		region = self.findPlateRegion(dilation, img_org1)
	
		# 用绿线画出这些找到的轮廓
		for box in region:
			cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
		if len(region) != 1:
			return False, img, img
		else:
			ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
			xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
			ys_sorted_index = np.argsort(ys)
			xs_sorted_index = np.argsort(xs)
		
			x1 = box[xs_sorted_index[0], 0]
			x2 = box[xs_sorted_index[3], 0]
		
			y1 = box[ys_sorted_index[0], 1]
			y2 = box[ys_sorted_index[3], 1]
		
			img_org2 = img.copy()
			img_plate = img_org2[y1:y2, x1:x2]
			#cv2.imshow('score plate', img_plate)
			#cv2.waitKey(0)
			#cv2.imwrite('score_plate.jpg', img_plate)
		
			# 带轮廓的图片
			#cv2.namedWindow('img', cv2.WINDOW_NORMAL)
			#cv2.imshow('img', img)
			#cv2.imwrite('contours.png', img)
			#cv2.waitKey(0)
			cv2.destroyAllWindows()
			return True, img, img_plate


if __name__ == '__main__':
	rootdir = sys.argv[1]
	detectScorePlate = DetectScorePlate()
	for parent,dirnames,filenames in os.walk(rootdir):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
		#for dirname in  dirnames:
			#print "parent is:" + parent
			#print  "dirname is" + dirname
		for filename in filenames:
			timt1 = time.
			#print "parent is:" + parent
			#print "filename is:" + filename
			#print "the full name of the file is:" + os.path.join(parent,filename) #输出文件路径信息
			fullpath = os.path.join(parent,filename)
			img = cv2.imread(fullpath)
			print img.shape
			is_score_plate, img, img_plate = detectScorePlate.detect(img)
			if is_score_plate:
				#print "Has Score Plate"
				cv2.imshow('score plate', img_plate)
				cv2.waitKey(0)
				print filename
				cv2.imwrite(filename, img)
