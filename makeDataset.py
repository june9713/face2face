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
        self.nextCnt = 0
        

    def makeArray(self, imgs):
        self.labels = []
        self.images = []
        self.labels = np.array(self.labels)
        self.images = np.array(self.images)
        self.num_examples = len(self.labels)

                
    def next_batch(self, size):
        result = ( self.images[self.nextCnt:self.nextCnt+size , : ] ,  self.labels[self.nextCnt:self.nextCnt+size , : ] )
        self.nextCnt = self.nextCnt+ size
        return result
            

class dataSet():
    def __init__(self , allImages):
        #random.shuffle(allImages)
        allNum = len(allImages)
        #print("1/2 : make Train Dataset")
        self.train = imageLabels()
        self.train.makeArray(allImages)
        #print("2/2 : make Train Dataset")
        self.test = imageLabels()
        self.test.makeArray(allImages)
        self.nextCnt = 0

        
    def next_batch(self, size):
        result = ( self.train.images[self.nextCnt:self.nextCnt+size , : ] ,  self.train.labels[self.nextCnt:self.nextCnt+size , : ] )
        self.nextCnt = self.nextCnt+ size
        return result
    

def loadData():
    return  searchFile("./")
    
