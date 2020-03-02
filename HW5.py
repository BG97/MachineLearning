# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:08:25 2018

@author: benny
"""
#Get dataset
'''Get feature data from file as a matrix with a row per data instance'''
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



def error_function(X,Y,B):
    n = len(Y)
    BX = vectorMultiply(X,B)
    BX = np.asarray(BX)
    error = np.sum((BX**2)/(2*n))
    return error

#print (error_function(X,Y,B))

def gradient_descent(X, Y, B, eta):
    #error_history = [0] * 1000
    n = len(Y)
    
    for iteration in range(1000):
        if error_function(X,Y,B) > 0.001:
            h = vectorMultiply(X,B)
            loss = h - Y
            gradient = X.T.dot(loss)/n
            B = B-eta*gradient
            #error = error_function(X,Y,B)
            #error_history[iteration] = error
    
    return B[1:]

'''
fullDataSetA = getFeatureData(sys.argv[1])

trainDataSet = getLabelData(sys.argv[2])
testDataSet = getLabelData(sys.argv[3])
'''
fullDataSetA = getFeatureData("ionosphere.data")
#trainDataSet = getLabelData("ionosphere.trainlabels.0")
#testDataSet = getLabelData("ionosphere.trainlabels.0")
#fullDataSet = getFeatureData("breast_cancer.data")
#trainDataSet = getLabelData("labelData.8")
#testDataSet = getFeatureData("inputData.9")
#labelfile = sys.argv[2]

f = open("ionosphere.trainlabels.0")
trainDataSet = {}
n = [0, 0]
l = f.readline()
while (l != ''):
    a = l.split()
    trainDataSet[a[1]] = int(a[0])
    #    trainlabels_size[a[0]] = trainlabels_size[a[0]]+1
    if (trainDataSet[a[1]] == 0):
        trainDataSet[a[1]] = -1;
    l = f.readline()
    
    n[int(a[0])] += 1


f.close()
#print(trainDataSet)

for i in (range(len(fullDataSetA))):
    fullDataSetA[i].append(float(1))

fullDataSet=[]
testSet=[]
predictionLabels = []
train_list = list(trainDataSet.keys())
for i in range(len(fullDataSetA)):
    if (str(i) in train_list):
        fullDataSet.append(fullDataSetA[i])
    else:
        testSet.append(fullDataSetA[i])
        predictionLabels.append(i)

#Retrive data from featurefile to labelFile
#Gradient Descent
eta_list = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001,0.00000000001]
bestobj = 10000000000
besteta = 0
stop = 0.001

w = []
for i in range(len(fullDataSet[0])):
    w.append(random.uniform(-0.01,0.01))



cols = len(fullDataSet[0])
rows = len(fullDataSet)



for e in range(len(eta_list)):
    eta = eta_list[e]
    d=1
    w = []
    error=0
    for i in range(len(fullDataSet[0])):
        w.append(random.uniform(-0.01,0.01))
    while(d>stop):
        deri =[]
        #gradient
        k=0
        for m in range(0, cols, 1):
            deri.append(0)
        for i in range(rows):
            if (trainDataSet.get(str(i)) != None):
                dp = dotProduct(w, fullDataSet[k])
                for j in range(cols):
                    deri[j] += (trainDataSet.get(str(i)) - dp) * fullDataSet[k][j]
                k+=1
        prev = error
        error = 0
        w_val = []
        #compute error
        k=0
        for i in range(rows):
            if (trainDataSet.get(str(i)) != None):
                error += (trainDataSet.get(str(i)) - (dotProduct(w, fullDataSet[k])))
                k+=1
        #update
        
        for j in range(cols):
            w[j] = w[j] + eta * deri[j]
            w_val.append(w[j])
        
        #print(error)
        if (prev > error):
            d = prev - error
        else:
            d = error - prev
        #print(d)

    print(eta, error)
    if error < bestobj:
        bestobj = error
        besteta = eta
'''
norm = 0
for i in range(cols -1):
    norm += w[i]**2

#print (w)
norm = math.sqrt(norm)
d_origin = abs(w[len(w)-1]/norm)
#print ("Distance is " + str(d_origin))
'''
'''
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
print (besteta, bestobj)
