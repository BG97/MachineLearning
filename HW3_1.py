# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:59:02 2018

@author: benny
"""

import sys
import random
import math

def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance index
and value as the class index
'''
def getLabelData(labelFile):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict
def matrixMultiply(a,b):
    result = []
    for i in range(len(a)):
        vec = []
        for j in range(len(b[0])):
            elem = 0
            for k in range(len(b)):
                elem += a[i][k] * b[k][j]
            vec.append(elem)
        result.append(vec)
    return result
import numpy as np
import math
import sys
import random

def sigmoid(y):
    if isinstance(y,list):
        return [sigmoid(item) for item in y]
    else:
        return 1.0/(1+math.exp(-y))


def dotProduct(v,u):
    sum = 0
    for i in range(len(u)):
        sum += u[i] * v[i]
    return sum

def norm(v):
    return math.sqrt(dotProduct(v,v)) 

def matrixMultiply(a,b):
    result = []
    for i in range(len(a)):
        vec = []
        for j in range(len(b[0])):
            elem = 0
            for k in range(len(b)):
                elem += a[i][k] * b[k][j]
            vec.append(elem)
        result.append(vec)
    return result

def vectorMultiply(a,v):
    result = []
    for row in a:
        result.append(dotProduct(row,v))
    return result

def matrixTranspose(a):
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


fullDataSetA = getFeatureData(sys.argv[1])

trainDataSet = getLabelData(sys.argv[2])
'''
testDataSet = getLabelData(sys.argv[3])

fullDataSetA = getFeatureData("ionosphere.data")
trainDataSet = getLabelData("ionosphere.trainlabels.0")
'''
#testDataSet = getFeatureData("inputData.9")
for i in trainDataSet:
    if trainDataSet[i] == 0:
        trainDataSet[i] = -1
for i in (range(len(fullDataSetA))):
    fullDataSetA[i].append(float(1))
fullDataSet=[]
testSet=[]
predictionLabels = []
train_list = list(trainDataSet.keys())
for i in range(len(fullDataSetA)):
    if (i in train_list):
        fullDataSet.append(fullDataSetA[i])
    else:
        testSet.append(fullDataSetA[i])
        predictionLabels.append(i)

rows = len(fullDataSet)
cols = len(fullDataSet[0])
# print(data)

#print("rows=", rows, " cols=", cols)

# print("this is new code")
w = []
for j in range(0, cols, 1):
    # print(random.random())
    w.append(0.02 * random.random() - 0.01)



eta = 0.0001
hingloss = rows * 10
diff = 1
count = 0

while ((diff) > 0.0001):
    dellf = [0] * cols
    i=0
    for j in range(rows):
        if (trainDataSet.get(int(j))) != None:
            dp = dotProduct(w, fullDataSet[i])
            condition = (trainDataSet.get(int(j)) * (dotProduct(w, fullDataSet[i])))
            for k in range(cols):
                if (condition < 1):
                    dellf[k] += -1 * (trainDataSet.get(int(j)) * fullDataSet[i][k])
                else:
                    dellf[k] += 0
            i+=1    

    for j in range(cols):
        w[j] = w[j] - eta * dellf[j]
    prev = hingloss
    hingloss = 0

    i=0
    for j in range(rows):
        
        if (trainDataSet.get(int(j)) != None):
            hingloss += max(0, 1 - (trainDataSet[j] * dotProduct(w, fullDataSet[i])))
            i+=1
        diff = abs(prev - hingloss)



    #print ("hingloss = " + str(diff))

# print("w= ")
normw = 0
for i in range((cols - 1)):
    normw += w[i] ** 2


#print (w)

normw = math.sqrt(normw)


d_orgin = abs(w[len(w) - 1] / normw)

#print ("Distance to origin = " + str(d_orgin))

#################################
###### calc of prediction #######
#################################
count0 = 0
count1 = 0
predResult = {}
for i in range(len(testSet)):
    #if (trainDataSet.get(str(i)) == None):
    dp = dotProduct(w, testSet[i])
    if ((dp) > 0):
        #print(dp)
        #print (int(1)+int(predictionLabels[i]))
        predResult[predictionLabels[i]] = 1
        #count1 += 1
    else:
        ##print (int(0)+int(predictionLabels[i]))
        predResult[predictionLabels[i]] = 0
        #count0 += 1
for i in predResult:
    print(predResult[i], ' ' , i)
'''
for i in range(0, rows, 1):
    if (trainDataSet.get(str(i)) == None):
        dp = dot_product(w, fullDataSet[i])
        if (dp > 0):
            print("1 " + str(i))
        else:
            print("0 " + str(i))
'''
            