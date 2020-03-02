
"""
Course Project : SVM classifcation/Prediction using Pearson Correlation feature selection for SNP dataset.
This is a simulated dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features). Amongst all SNPs are 15 causal 
ones which means they and neighboring ones discriminate between case and 
controls while remainder are noise.

Author: Raj Ponnusamy[rp775] and Zibin Guan[zg56]
"""
import sys
import array
import copy
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid


def extractCol(matrix, i):
    return [[row[i]] for row in matrix]


def mergeCol(a, b):
    return [x + y for x, y in zip(a, b)]


def Pearson(x, y, fi):
    sumX = 0
    sumX2 = 0
    ro = len(x)
    co = len(x[0])
    switch = 0
    pc = array.array("f")
    for i in range(0, co, 1):
        switch += 1
        sumY = 0
        sumY2 = 0
        sumXY = 0
        for j in range(0, ro, 1):
            if (switch == 1):
                sumX += y[j]
                sumX2 += y[j] ** 2
            sumY += x[j][i]
            sumY2 += x[j][i] ** 2
            sumXY += y[j] * x[j][i]
        r = (ro * sumXY - sumX * sumY) / ((ro * sumX2 - (sumX ** 2))
                                          * (ro * sumY2 - (sumY ** 2))) ** (0.5)
        pc.append(abs(r))

    savedToPrint = array.array("f")
    myFeatures = array.array("i")
    for i in range(0, fi, 1):
        selected = max(pc)
        savedToPrint.append(selected)
        featureIndex = pc.index(selected)
        pc[featureIndex] = -1
        myFeatures.append(featureIndex)
    return myFeatures


def CreateDataSet(fea, dat):
    newData = extractCol(dat, fea[0])
    newLab = array.array("i")
    fea.remove(fea[0])
    length = len(fea)
    for i in range(0, length, 1):
        temp = extractCol(dat, fea[0])
        newData = mergeCol(newData, temp)
        fea.remove(fea[0])
    return newData


##################
# Read data
##################
#datafile = sys.argv[1]
datafile = 'traindata'
data = []
print("project code is running. please wait to see the results ....")
print("Reading training data")
with open(datafile, "r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        data.append(l)

####################
# Read labels
####################

#labelfile = sys.argv[2]
labelfile = 'trueclass'

trainlabels = array.array("i")
with open(labelfile, "r") as infile:
    for line in infile:
        temp = line.split()
        trainlabels.append(int(temp[0]))

print("Completed reading training data", end="")

feat = 10


rows = len(data)
cols = len(data[0])
rowsl = len(trainlabels)


# Dimensionality Reduction

neededFea = Pearson(data, trainlabels, 2000)

print("Completed ", end="")

savedFea = copy.deepcopy(neededFea)

data1 = CreateDataSet(neededFea, data)


clf_svm = svm.SVC(gamma=0.001)
clf_log = linear_model.LogisticRegression()
clf_gnb = GaussianNB()
clf_nc = NearestCentroid()

allAccuracies = array.array("f")
allFeatures = []

accuracy_svm = 0
accuracy_score = 0
accuracy_log = 0
accuracy_gnb = 0
accuracy_nc = 0

my_accuracy = 0

iterations = 5
print("\ncross validation iteration: ", end="")
for i in range(iterations):

    print("\ncross validation iteration: ", i)

    X_train, X_test, y_train, y_test = train_test_split(
        data1, trainlabels, test_size=0.3)

    newRows = len(X_train)
    newCols = len(X_train[0])
    newRowst = len(X_test)
    newColst = len(X_test[0])

    newRowsL = len(y_train)

    PearFeatures = Pearson(X_train, y_train, feat)

    allFeatures.append(PearFeatures)
    argument = copy.deepcopy(PearFeatures)

    data_fea = CreateDataSet(argument, X_train)

    clf_svm.fit(data_fea, y_train)
    clf_log.fit(data_fea, y_train)
    clf_gnb.fit(data_fea, y_train)
    clf_nc.fit(data_fea, y_train)

    TestFeatures = Pearson(X_test, y_test, feat)

    test_fea = CreateDataSet(TestFeatures, X_test)

    len_test_fea = len(test_fea)
    counter_svm = 0
    counter_log = 0
    counter_gnb = 0
    counter_nc = 0
    my_counter = 0

    for j in range(0, len_test_fea, 1):
        predLab_svm = int(clf_svm.predict([test_fea[j]]))
        predLab_log = int(clf_log.predict([test_fea[j]]))
        predLab_gnb = int(clf_gnb.predict([test_fea[j]]))
        predLab_nc = int(clf_nc.predict([test_fea[j]]))
        h = predLab_svm + predLab_log + predLab_gnb + predLab_nc
        if (h >= 3):
            my_predLab = 1
        elif (h <= 1):
            my_predLab = 0
        else:
            my_predLab = predLab_svm
        if (my_predLab == y_test[j]):
            my_counter += 1
        if (predLab_svm == y_test[j]):
            counter_svm += 1
        if (predLab_log == y_test[j]):
            counter_log += 1
        if (predLab_gnb == y_test[j]):
            counter_gnb += 1
        if (predLab_nc == y_test[j]):
            counter_nc += 1

    accuracy_svm += counter_svm / len_test_fea
    accuracy_log += counter_log / len_test_fea

    accuracy_gnb += counter_gnb / len_test_fea
    accuracy_nc += counter_nc / len_test_fea

    my_accuracy += my_counter / len_test_fea
    allAccuracies.append(my_counter / len_test_fea)


print(" Completed", end="")

bestAc = max(allAccuracies)
bestInd = allAccuracies.index(bestAc)
bestFeatures = allFeatures[bestInd]

print("\nSelected Features: ", feat)

originalFea = array.array("i")
for i in range(0, feat, 1):
    realIndex = savedFea[bestFeatures[i]]
    originalFea.append(realIndex)

print("The features are: ", originalFea)

#####################################################
# Calculate Accuracy
#####################################################
argument1 = copy.deepcopy(originalFea)
AccData = CreateDataSet(argument1, data)


clf_svm.fit(AccData, trainlabels)
clf_log.fit(AccData, trainlabels)
clf_gnb.fit(AccData, trainlabels)
clf_nc.fit(AccData, trainlabels)

svm_counter = 0
LeCounter = 0
k = len(AccData)
for i in range(0, k, 1):
    predLab_svm = int(clf_svm.predict([AccData[i]]))
    predLab_log = int(clf_log.predict([AccData[i]]))
    predLab_gnb = int(clf_gnb.predict([AccData[i]]))
    predLab_nc = int(clf_nc.predict([AccData[i]]))
    h = predLab_svm + predLab_log + predLab_gnb + predLab_nc
    if (h >= 3):
        my_predLab = 1
    elif (h <= 1):
        my_predLab = 0
    else:
        my_predLab = predLab_svm
    if (my_predLab == trainlabels[i]):
        LeCounter += 1
    if (predLab_svm == trainlabels[i]):
        svm_counter += 1

FinalAcc = LeCounter / k
SVMAc = svm_counter / k
print("The Accuracy is: ", FinalAcc * 100)


print("\nPredicted labels of the test data are saved in test_data.labels file using SVM Pearson Correlation feature selection method")

##################
# Read Test data
##################

#testfile = sys.argv[3]
testfile = 'testdata'
testdata = []
print("Reading test data...")
with open(testfile, "r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        testdata.append(l)

print("Completed reading test data")


argument2 = copy.deepcopy(originalFea)
testdata1 = CreateDataSet(argument2, testdata)


f1 = open("predicted_test_data.labels", "w+")

for i in range(0, len(testdata1), 1):
    lab1 = int(clf_svm.predict([testdata1[i]]))
    lab2 = int(clf_log.predict([testdata1[i]]))
    lab3 = int(clf_gnb.predict([testdata1[i]]))
    lab4 = int(clf_nc.predict([testdata1[i]]))
    h = lab1 + lab2 + lab3 + lab4
    if (h >= 3):
        f1.write(str(1) + " " + str(i) + "\n")
    elif (h <= 1):
        f1.write(str(0) + " " + str(i) + "\n")
    else:
        f1.write(str(lab1) + " " + str(i) + "\n")

print("saved output results into the predicted_test_data.labels file on the current directory ")