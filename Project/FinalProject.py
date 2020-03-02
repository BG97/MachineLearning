#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:19:28 2018

@author: hitesh
"""

import pandas as pd 
import os 
import sys 
import sklearn 
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Imputer 
import itertools  
import math

datafile = sys.argv[1]

data = pd.read_csv(datafile, sep = ' ')

data.columns = range(data.shape[1])
labelfile= sys.argv[2]
labels = pd.read_csv(labelfile, sep = ' ')
testfile = sys.argv[3]
test_data = pd.read_csv(testfile, sep = ' ')

test_data.columns = range(test_data.shape[1])

#print('Number of Columns : ',len(data.columns))
#print('Number of Rows: ',len(data))

# Drop columns that have all rows as Nan's
data = data.dropna(how='all',axis = 'columns')
test_data = test_data.dropna(how = 'all', axis = 'columns')
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
#imputer = imputer.fit(data)
#dataset = imputer.transform(data)

# Number of columns dropped = 16

#column_variance = data.var()
#column_variance.to_csv(path='column_variance.csv')
#mean_variances = column_variance.mean()
#cut_off_margin = 0.25
#margin = mean_variances - cut_off_margin*mean_variances
#df = data.loc[:, data.var() > margin]
#selected_features = []
#selected_features.append(data.var() > margin)                
#print("length of features", len(df.columns))
#print("length of rows", len(df))

# ------------ Variance Threshold ---------------------------------

#cutoff_threshold = 0.513320024;
#selector = VarianceThreshold(threshold= cutoff_threshold);
#print('Selector is created. Now fitting the data')
#selected_features = selector.fit_transform(dataset);
#print("The selected features", selected_features)
#print('After Variance threshold features are reduced to',selected_features.shape)
#data_new = pd.DataFrame(data= selected_features)
#print('Length of the features', len(data_new.columns))

# --------------------- PCA ---------------------------------------

'''pca = PCA(n_components = 30, whiten= 'True')
data = pca.fit(data).transform(data)
pca.explained_variance_

'''

#train_labels = y_train.iloc[:,0].ravel()
#test_labels = y_test.iloc[:,0].ravel()

#----------------------  SVD -------------------------------------
#svd = TruncatedSVD(n_components = 30)
#data = svd.fit(data).transform(data)


#------------- Pearson's Correlation--------------------------------

def pearson_correlation(numbers_x, numbers_y):
    mean_x = sum(numbers_x)/len(numbers_x)
    mean_y = sum(numbers_y)/len(numbers_y)

    subtracted_mean_x = [j - mean_x for j in numbers_x]
    subtracted_mean_y = [k - mean_y for k in numbers_y]

    x_times_y = [a * b for a, b in list(zip(subtracted_mean_x, subtracted_mean_y))]

    x_squared = [j * j for j in numbers_x]
    y_squared = [k * k for k in numbers_y]

    corr_value = sum(x_times_y) / math.sqrt(sum(x_squared) * sum(y_squared))

    return corr_value

ls = []
for i in range(len(data.columns)):
    ls.append(pearson_correlation(data.iloc[:,i].values,labels_new))

selected_col = []
count = 0
for i in ls:
    if i > 0.035:
         selected_col.append(count)
    count += 1   


print("Number of features Selected", len(selected_col))
print("The columns used", selected_col)

train_data = data.iloc[:,selected_col]
train_label = labels.iloc[:,0].values
test_data = test_data.iloc[:,selected_col]

# Cross Validation 
#X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.2, random_state = 0)


# -------------- SVC ---------------------------------------
#from sklearn.svm import SVC 

#clf = SVC()

#clf.fit(X_train, y_train)

#y_pred = clf.predict(X_test)
# -------------------- Logistic Regression ---------------

#from sklearn.linear_model import LogisticRegression

#clf = LogisticRegression(random_state = 0)

#clf.fit(X_train, y_train)

#y_pred = clf.predict(X_test)

# -------------------- Naive  Bayes----------------------------------

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

#from sklearn.metrics import accuracy_score 
#acc = accuracy_score(y_test, y_pred)

#print("accuracy", acc)

# ------------------ Logistic Regression ---------------------------

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0)

clf.fit(train_data, train_label)

y_pred = clf.predict(test_data)

print("The Predcited Values:", y_pred)

#from sklearn.metrics import accuracy_score 
#acc = accuracy_score(y_test, y_pred)

