#coding=utf-8
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
import sys
from math import *

def readColorList(filename):
    colorlist = []
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split('\t')
            if len(tokens) <=2:
                continue
            colorlist.append((tokens[0], tokens[1]))
    return colorlist
def readWordList(filename):
    wordlist = []
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split('\t')
            if len(tokens) == 0:
                continue
            wordlist.append(tokens[0])
    return wordlist

# font = ImageFont.truetype("Arial-Bold.ttf",14)

def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)

    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50];
    adder = cv2.resize(adder, (50, 50));
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

def rot(img,angel,shape,max_angel):
    """ 使图像轻微的畸变

        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸

    """
    size_o = [shape[1],shape[0]]

    size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])


    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]));

    pts1 = np.float32([[0,0]         ,[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):

        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,size);

    return dst;

def rotRandrom(img, factor, size):
    shape = size;
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);
    return dst;



def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);

    '''
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);
    '''
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.8+ np.random.random()*0.2);
    hsv[:,:,2] = hsv[:,:,2]*(0.8+ np.random.random()*0.2);

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img

def random_envirment(img,data_set):
    index=r(len(data_set))
    env = cv2.imread(data_set[index])

    env = cv2.resize(env,(img.shape[1],img.shape[0]))

    bak = (img==0);
    bak = bak.astype(np.uint8)*255;
    inv = cv2.bitwise_and(bak,env)
    img = cv2.bitwise_or(inv,img)
    return img

def random_bg(img,data_set):
    index=r(len(data_set))
    bg = cv2.imread(data_set[index])

    bg = cv2.resize(bg,(img.shape[1],img.shape[0]))
    img = cv2.bitwise_or(img,bg);
    return img

def GenCh(f,val,size, color):
    img=Image.new("RGB", size,(0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((5, 0),val,fill=color,font=f)
    A = np.array(img)
    return A

def GenEng(f,val,size, color):
    img=Image.new("RGB", size,(0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((5, 0),val.decode('utf-8'),color,font=f)
    A = np.array(img)
    return A

def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1));


def r(val):
    return int(np.random.random() * val)
def rfloat(val):
    return np.random.random() * val

def AddNoiseSingleChannel(single):
    diff = 255-single.max();
    noise = np.random.normal(0,0.5+rfloat(0.5),single.shape);
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise;
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0]);
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1]);
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2]);
    return img;


class GenWord:


    def __init__(self,fontCh,fontEng,BackGrounds,colorfile,wordsfile, imgsize):
        self.fontC =  ImageFont.truetype(fontCh,23,0);
        self.fontE =  ImageFont.truetype(fontEng,23,0);
        self.img=np.array(Image.new("RGB", imgsize,(0,0,0)))
        self.imgsize = imgsize
        self.words = readWordList(wordsfile)
        self.colors = readColorList(colorfile)
        #self.bg  = cv2.resize(cv2.imread("./images/template.bmp"),(226,70));
        #self.smu = cv2.imread("./images/smu2.jpg");
        self.bg_path = [];
        for parent,parent_folder,filenames in os.walk(BackGrounds):
            for filename in filenames:
                path = parent+"/"+filename;
                self.bg_path.append(path);


    def draw(self,val,size, color):
        self.img = GenCh(self.fontC,val,size,color);
        return self.img

    def generate(self,text, colors):
        if len(text) != 0:
            print "text:%s" % text
            sys.stdout.flush()
            color_index = r(len(colors))
            print "color:",colors[color_index]
            fg = self.draw(text.decode(encoding="utf-8"), self.imgsize, colors[color_index][1]);
            #cv2.imshow('fg',fg)
            #cv2.waitKey(0)
            #fg = cv2.bitwise_not(fg);
            #cv2.imshow('fg-not',fg)
            #cv2.waitKey(0)
            #com = cv2.bitwise_or(fg,self.bg);
            com = random_bg(fg, self.bg_path)
            #cv2.imshow('com-random_bg',com)
            #cv2.waitKey(0)
            #com = rot(com,r(30)-15,com.shape,15);
            #cv2.imshow('com-rot',com)
            #cv2.waitKey(0)
            #com = rotRandrom(com,10,(com.shape[1],com.shape[0]));
            #com = AddSmudginess(com,self.smu)
            #cv2.imshow('com-rotrandom',com)
            #cv2.waitKey(0)

            com = tfactor(com)
            #cv2.imshow('com-tfactor',com)
            #cv2.waitKey(0)
            #com = random_envirment(com,self.bg_path);
            #cv2.imshow('com-random_envirment',com)
            #cv2.waitKey(0)
            com = AddGauss(com, 1+r(1));
            #cv2.imshow('com-AddGauss',com)
            #cv2.waitKey(0)
            #com = addNoise(com);
            #cv2.imshow('com-addNoise',com)
            #cv2.waitKey(0)
            return com

    def genWordString(self,val,wordset, maxrange):
        wordStr = ""
        if val == "":
            wordStr = wordset[r(maxrange)]
        else:
            wordStr = val

        return wordStr;

    def genBatch(self, batchSize,wordset,colors,outputPath,size):
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in xrange(batchSize):
                wordStr = self.genWordString("",wordset, len(wordset))
                img =  self.generate(wordStr,colors);
                img = cv2.resize(img,size);
                cv2.imshow("final",img)
                cv2.waitKey(0)
                cv2.imwrite(outputPath + "/" + str(i).zfill(2) + ".jpg", img);
                

G = GenWord("./fonts/msyh.ttf",'./fonts/msyh.ttf',"./background","./color/color.txt", "./words/words.txt",(32,32))
G.genBatch(100,G.words,G.colors,"./sample",(32,32))

