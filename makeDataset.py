import os
import numpy as np
from PIL import Image
import pickle
import random
import traceback
import time
import cv2


def getTracebackStr():
    lines = traceback.format_exc().strip().split('\n')
    rl = [lines[-1]]
    lines = lines[1:-1]
    lines.reverse()
    for i in range(0,len(lines),2):
        rl.append('^\t%s at %s' % (lines[i].strip(),lines[i+1].strip()))
    return '\n'.join(rl)


    
def searchFile(dirname):
    result = []

    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png':
                result.append(path+ "/"+filename)
    return result


class imageLabels():
    def __init__(self):
        self.kyes = [ 'L_eye_brow_ctr' , 'R_eye_brow_ctr' , 'L_lip_ctr' , 'R_lip_ctr']
        self.kyes = [ 'R_eye_brow_ctr' ,'R_lip_ctr']
        self.images = []
        self.labels = []
        

    def makeArray(self, imgs):
        print("***************")
        
        if len(imgs) == 0:
            
            self.labels = []
            self.images = []
            return 
            
        print("len of data" , len(imgs))
        for i, x in enumerate(imgs):
            if i % 100 == 0:
                try:
                    print("total : " , len(imgs) , "cnt : " , i  , "images : " , len(self.labels) , self.labels[-1].shape)
                except:
                    print("total : " , len(imgs) , "cnt : " , i  , "images : " , len(self.labels))
                
            im = Image.open(x).convert('L')
            width, height = im.size
            
            im = im.crop((0, 0, width/2, height))
            w = 64
            h = 64
            im = im.resize((28, 28), Image.ANTIALIAS)
            #im.save("test.png")
            print("size" , im.size)
            greyscale_map = np.array(list(im.getdata()))

            greyscale_map = (greyscale_map / 255.0 * 0.99) + 0.00001

            
            self.images.append(greyscale_map)
            #print("x",x)
            txtName = x.split('.png')[0]+".txt"
            with open(txtName , 'rb') as f:
                rawLabel= pickle.load(f)
            labels = []
            for k in self.kyes:
                labels.append(rawLabel[k][0])
                labels.append(rawLabel[k][1])
            self.labels.append(labels)
        self.labels = np.array(self.labels)
        self.images = np.array(self.images)
        self.num_examples = len(self.labels)
        #print("test",self.images.shape)
        print("len",len(greyscale_map))
                

            

class dataSet():
    def __init__(self , allImages):
        #random.shuffle(allImages)
        allNum = len(allImages)
        print("1/2 : make Train Dataset")
        self.train = imageLabels()
        self.train.makeArray(allImages[:allNum - int(allNum/9)])
        print("2/2 : make Train Dataset")
        self.test = imageLabels()
        self.test.makeArray(allImages[allNum -int(allNum/9):])
        self.nextCnt = 0

        
    def next_batch(self, size):
        #print("shape1", self.train.images.shape)
        #print("self.nextCnt , size" , self.nextCnt , size , self.train.images.shape , self.train.labels.shape)
        #print(type(self.nextCnt) , type(self.nextCnt+size) , type(self.nextCnt+size) , type(self.nextCnt) , type(self.nextCnt+size) )
        result = ( self.train.images[self.nextCnt:self.nextCnt+size , : ] ,  self.train.labels[self.nextCnt:self.nextCnt+size , : ] )
        self.nextCnt = self.nextCnt+ size
        #print("shape",result[0].shape)
        
        return result
    

def loadData():
    return  searchFile("./")
    
