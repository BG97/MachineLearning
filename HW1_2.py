# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:04:10 2018

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


import math
import sys
'''
fullDataSet = getFeatureData(sys.argv[1])
trainDataSet = getLabelData(sys.argv[2])
#testDataSet = getLabelData(sys.argv[3])
'''

fullDataSet = getFeatureData("ionosphere.data")
trainDataSet = getLabelData("ionosphere.trainlabels.0")
testDataSet = getLabelData("ionosphere.labels")

'''
fullDataSet = getFeatureData("breast_cancer.data")
trainDataSet = getLabelData("breast_cancer.trainlabels.0")
testDataSet = getLabelData("breast_cancer.labels")
'''
#Retrive data from featurefile to labelFile
#Seperate classes

class0=[]
class1=[]
for i in trainDataSet:
    if trainDataSet[i] == 0:
        class0.append(fullDataSet[i])
    elif trainDataSet[i] == 1:
        class1.append(fullDataSet[i])


list0 = 0
means0 = []


for row in class0:
    list0 = [sum(x) for x in zip(*class0)]
for sample in list0:
    means0.append(sample/len(class0))


list1 = 0
means1 = []
for row in class1:
    list1 = [sum(x) for x in zip(*class1)]
for sample in list1:
    means1.append(sample/len(class1))

#Set column to row by using zip
class0 = list(zip(*class0))
class1 = list(zip(*class1))

#sd
sd0 = []
meanCul = 0
for row in range(len(class0)):
    
    diffMean2 = 0 
    for column in range(len(class0[row])):
        diffMean2 += ((class0[row][column] - means0[meanCul])**2)
    if diffMean2 == 0:
        diffMean2 = 1
        diffMean2 = math.sqrt(diffMean2/(len(class0[row])))
    else:
        diffMean2 = math.sqrt(diffMean2/(len(class0[row])))
    sd0.append(diffMean2)
    meanCul += 1


sd1 = []
meanCul = 0
for row in range(len(class1)):
    diffMean2 = 0 
    for column in range(len(class1[row])):
        diffMean2 += ((class1[row][column] - means1[meanCul])**2)
    if diffMean2 == 0:
        diffMean2 =1
        diffMean2 = math.sqrt(diffMean2/(len(class0[row])))
    else:
        diffMean2 = math.sqrt(diffMean2/(len(class0[row])))
    sd1.append(diffMean2)
    meanCul += 1



#Get the testSet
classT = []
num = 0
predictionLabels = []
train_list = list(trainDataSet.keys())
for i  in range(len(fullDataSet)):
    if (i not in train_list):
        classT.append(fullDataSet[i])
        predictionLabels.append(i)
    #classT[num].insert(len(testDataSet), testDataSet[i])



#prediction 
pred0 = []

for row in range(len(classT)):
    resultT = 0
    for column in range(len(classT[row])-1):
        resultT += (((classT[row][column] - means0[column])/sd0[column])**2)
    pred0.append(resultT)



pred1 = []
for row in range(len(classT)):
    resultT = 0
    for column in range(len(classT[row])-1):
        resultT += (((classT[row][column] - means1[column])/sd1[column])**2)
    pred1.append(resultT)


predResult = {}
for i in range(len(pred0)):
    if pred0[i] <= pred1[i]:
        predResult[predictionLabels[i]] = 0
    elif pred0[i] > pred1[i]:
        predResult[predictionLabels[i]] = 1
print(predResult)
-'''
true = 0
false = 0
for i in range(len(predictionLabels)):
    if predResult.get(predictionLabels[i]) == testDataSet[predictionLabels[i]]:
        true += 1
    else:
        false += 1
accuracy = true/(true+false)
print(accuracy)
'''