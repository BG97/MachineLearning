# -*- coding: utf-8 -*-


#from sklearn import svm
import array
import random
import time
import os
import math
import sys
"""
#############################################################
################## FUNCTIONS ################################
#############################################################
"""
def chi_square(data,top):
    """
    Calculate Chi-Square
    Input: Dataset contain labels as the last column
    top: number of top features to select from dataset.
    Output: the top indexs column of dataset.
    """
    label = [row[-1] for row in data]
    rows = len(data)
    print('Rows in data',str(rows))
    cols = len(data[0])-1
    print('Cols in data',str(cols))

    T = []
    for j in range(0, cols):
        print("Feature Selection On column: ", str(j))
        ct= [[1,1],[1,1],[1,1]]    # contingency table
                            #To avoid the Expected Value to equal Zero;
                            #The contingency table is initalized to at least 1 observation
                            #
        for i in range(0, rows):
	        if label[i] == 0:
	            if data[i][j] == 0:
	                ct[0][0] += 1
	            elif data[i][j] == 1:
	                ct[1][0] += 1
	            elif data[i][j] == 2:
	                ct[2][0] += 1
	        elif label[i] == 1:
	            if data[i][j] == 0:
	                ct[0][1] += 1
	            elif data[i][j] == 1:
	                ct[1][1] += 1
	            elif data[i][j] == 2:
	                ct[2][1] += 1
        col_totals = [ sum(x) for x in ct]# number of times 0,1,2 appears wihtin one column
        row_totals = [ sum(x) for x in zip(*ct) ] # the number of classicfications case=0 or control=0
        total = sum(col_totals) #total number of observation

        exp_value = [[(row*col)/total for row in row_totals] for col in col_totals]

        sqr_value = [[((ct[i][j] - exp_value[i][j])**2)/exp_value[i][j] for j in range(0,len(exp_value[0]))] for i in range(0,len(exp_value))]	    
        chi_2 = sum([sum(x) for x in zip(*sqr_value)])
        T.append(chi_2)
    indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)# sorts the T list from greatest to least
    idx = indices[:top]# this will get the "top" number of indices
    return(idx)

def feature_extraction(data,FeatureIndex_col):
    new_data=[]
    columns=list(zip(*data))
    for i in FeatureIndex_col:
        new_data.append(columns[i])
    new_data=list(zip(*new_data))
    return(new_data)

def subsample(dataset,labels,sampleRatio):
    sample = list()
    sampleLabel=list()
    n_sample = round(len(dataset)*sampleRatio)

    row_index=[random.randint(0, n_sample-1) for i in range(0,n_sample)] #select rows WITH replacements
        
    for ii in row_index:
        sample.append(dataset[ii])
        sampleLabel.append(labels[ii])
    return sample,sampleLabel

def bagging_predict(svmModels, row):
    predictions = [list(mod.predict([row])) for mod in svmModels]
    pred=[i[0] for i in predictions]
    return max(set(pred), key=pred.count)
def buildSvm(samData,samLabel):
    model = svm.SVC(kernel='linear', C=0.1) 
    model.fit(samData, samLabel)
    model.score(samData, samLabel)
    return model

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
def tupleToList(list_of_tuples):
    list_of_lists = [list(elem) for elem in list_of_tuples]
    return(list_of_lists)
"""
###############################################################################
########################  MAIN BODY ###########################################
###############################################################################
"""
"""
################################################################
###################### Load Data ##############################
################################################################
"""
print('Beginning Bagged SVM Classifier: Approximate Runtime: 10 min')
start_time = time.time()
print('Loading Data',time.strftime("%H:%M:%S", time.gmtime(start_time)))

file_path = os.path.dirname(os.path.abspath('__file__'))

#datafile= sys.argv[1]#"traindata.txt" #
datafile= "traindata"
f=open(datafile,'r')
data=[]
i=0
l=f.readline()
while(l !=''):
    a=l.split()
    l2=array.array("f",[])
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    data.append(l2)
    l=f.readline()

datafile = "trainingLabels.txt"
#datafile= sys.argv[2]#"tureClass.txt" #
f=open(datafile,'r')
label=[]
i=0
l=f.readline()
while(l !=''):
    a=l.split()
    l2=array.array("f",[])
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    label.append(l2)
    l=f.readline()
    

#Open testdata file
datafile = "testdata"
#datafile= sys.argv[3]#"testdata" # #Feature selected Train data
f=open(datafile,'r')
realtest=[]
i=0
l=f.readline()
while(l !=''):
    a=l.split()
    l2=array.array("f",[])
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    realtest.append(l2)
    l=f.readline()
    
del(datafile,l,l2,a,i,j)

trainLabels,index=zip(*label)
trainLabels=list(trainLabels)

#COMBINE data and labels
for ii in range(0,len(data)):
    data[ii].append(trainLabels[ii])
len(data)
len(data[0])

del(index,ii,label)
print('End of Loading Data')

"""
######################################################
###### Split Data 70% train and 30% test #############
######################################################
"""
print('Splitting Data')

ratio=0.70
length_data=len(data)
train_size=int(length_data*ratio)


index_train=random.sample(range(length_data),train_size)

train_subset=[]
test_subset=[]
for ii in range(len(data)):
    if ii in index_train:
        train_subset.append(data[ii])
    else:
        test_subset.append(data[ii])
        
del(data,length_data,train_size,ii,trainLabels)
print('End of Splitting Data')
"""
#############################################
############ Feature Selection ##############
#############################################
"""
print('Feature Selection Begins')

top_col=chi_square(train_subset,12)
new_ReatTestSet=feature_extraction(realtest,top_col) # real test data is feature selected
top_col.append(len(train_subset[0])-1)#ADD LABELS COLUMN which the last columns
new_dataSet=feature_extraction(train_subset,top_col)
new_TestSet=feature_extraction(test_subset,top_col)
del(realtest)

print('Feature Selection ENDS')

"""
####################################################################
############## Implement SVM Model With Bagging ####################
####################################################################
"""

new_dataSet=tupleToList(new_dataSet)
new_TestSet=tupleToList(new_TestSet)
new_ReatTestSet=tupleToList(new_ReatTestSet)


RealDataLabel = [row[-1] for row in new_dataSet]    
for row in new_dataSet:
    del(row[-1])
    
RealTestLabel = [row[-1] for row in new_TestSet] 
for row in new_TestSet:
    del(row[-1])
  

print('Start of 100 Bags of SVM')
numBags=100
svmModels=[]*numBags
#Creating 100 bags
for bag in range(0,numBags):
    #Create with replacement N' sample set
    samData,samLabel=subsample(new_dataSet,RealDataLabel,1)
    #SVM linear Model
    m=buildSvm(samData,samLabel)
    #print("Model # created: ",str(bag+1))
    svmModels.append(m)
    #print("Model # ",str(bag+1),"appended")
    print("Number of Models Created: ",len(svmModels))
#end of baggin models

#Prediction

predictions = [bagging_predict(svmModels, row) for row in new_TestSet]
accuracy=accuracy_metric(RealTestLabel, predictions)

print('Accuracy of',str(int((1-ratio)*100)),'percent unseen test-train data',accuracy)


print('End of Bagging SVM')

"""
##############################################################################
######## Real Test Prediction ################################################
##############################################################################
"""
print('Running Real Test predictions')

realPredict= [bagging_predict(svmModels, row) for row in new_ReatTestSet]
########## Save file ########################

file_path = os.path.dirname(os.path.abspath('__file__'))

path=label_path = file_path+'\\test_data.labels'
f = open(path, "a")
w=0
for item in realPredict:
  f.write(' '.join(map(str, [int(item)]))+' '+str(w)+ "\n")
  w=w+1
f.close()
print('Real Test File Prediction Saved.')
del(top_col[-1])
print('The feature column index used: ', str(len(top_col)))
print('The feature column index used: ', str(top_col))
print('Prediction file Saved in: ',path)

elapsed_time = time.time() - start_time
#print(elapsed_time)
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('SVM Classification: ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))