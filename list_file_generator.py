# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:16:22 2019

@author: Ben3
"""

#import csv
#import os
#path = "C:\\Users\\Ben3\\Desktop"
#with open('list_file_test1.csv', 'w', newline='') as newfile:
#    writer = csv.writer(newfile)
#    for x in os.listdir(path):
#        try:
#            files = os.listdir(path + "\\" + x)
#            numframes = len(files)
#            writer.writerow([x, numframes])
#        except:
#            print("Exception Occurred")

import json
import csv
import math
import os

path = "/data/vision/oliva/scratch/cfosco/deepfake_detection/fb_dfd_release_0.1_final"
realcount = 0
fakecount = 0
trainrealdata = []
testrealdata = []
with open('dataset.json') as json_file:
    data = json.load(json_file)
    with open('list_file_train.csv', 'w', newline='') as trainfile:
        trainwriter = csv.writer(trainfile)
        for f in data.keys():
            s = f.split("/",maxsplit=1)
            s[0] = s[0] + "_frames"
            last = s[1]
            fname = s[0] + "/" + s[1]
            set = data[f]['set']
            if set == "train":
                numframes = os.listdir(path + "/" + s[0] + "/" + last[-1])
                if data[f]['label'] == 'real':
                    label = 1
                    trainrealdata.append([fname, str(numframes), str(label)])
                    realcount = realcount + 1
                if data[f]['label'] == 'fake':
                    label = 0
                    fakecount = fakecount + 1
                trainwriter.writerow([fname, str(numframes), str(label)])
       
        diff = fakecount - realcount
        trainrealdata_expand = trainrealdata * (math.ceil(diff/len(trainrealdata)))
        for x in range(len(trainrealdata_expand)): 
            if fakecount > realcount:
                fname = trainrealdata_expand[x][0]
                label = 1
                trainwriter.writerow([fname, str(label)])
                realcount = realcount + 1
    realcount = 0            
    fakecount = 0            
    with open('list_file_test.csv', 'w', newline='') as testfile:
        testwriter = csv.writer(testfile)
        for f in data.keys():
            s = f.split("/",maxsplit=1)
            s[0] = s[0] + "_frames"
            last = s[1]
            fname = s[0] + "/" + s[1]
            set = data[f]['set']
            if set == "test":
                numframes = os.listdir(path + "/" +  s[0] + "/" + last[-1])
                if data[f]['label'] == 'real':
                    label = 1
                    testrealdata.append([fname, str(numframes), str(label)])
                    realcount = realcount + 1
                if data[f]['label'] == 'fake':
                    label = 0
                    fakecount = fakecount + 1
                testwriter.writerow([fname, str(numframes), str(label)])
        
        diff = fakecount - realcount
        testrealdata_expand = testrealdata * (math.ceil(diff/len(testrealdata)))
        for x in range(len(testrealdata_expand)):
            if fakecount > realcount:
                fname = testrealdata_expand[x][0]
                label = 1
                testwriter.writerow([fname, str(label)])
                realcount = realcount + 1
            