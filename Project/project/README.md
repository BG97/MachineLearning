# Course Project : SVM classifcation/Prediction using Pearson Correlation feature selection for SNP dataset.
This is a simulated dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features). Amongst all SNPs are 15 causal 
ones which means they and neighboring ones discriminate between case and 
controls while remainder are noise.

Author: Raj Ponnusamy[rp775] and Zibin Guan[zg56]

Observation:
my deskttop it took at max 10mins.
AFS it took one hour
Kong machine it took at max 15 to 20 minx.
Execution time various depends on io resource. 


Copied the following files to repo.

1.project.py ==>source code of the project
2. predicated output: "test_data.labels"
3.Output_feature_column ==> output of selected feature [10]



Steps to run the code: 
1. Change directory to project location
Example: 
cd /afs/cad/courses/ccs/f18/cs/675/101/rp775/project

2. Run the python project.py  <INPUT TRAINING DATASET FILE> <INPUT TRAINING LABEL FILE> <INPUT TEST DATASET FILE>

Example:
python project.py traindata trueclass testdata


Expected Result: code should print out result 
Output:

kong-44 ProjectPearon >: python project.py traindata trueclass testdata 
project code is running. please wait to see the results ....
Reading training data
Completed reading training dataCompleted 
cross validation iteration: 
cross validation iteration:  0

cross validation iteration:  1

cross validation iteration:  2

cross validation iteration:  3

cross validation iteration:  4
 Completed
Selected Features:  10
The features are:  array('i', [9004, 27013, 3001, 15007, 999, 1000, 7003, 21010, 19009, 997])
The Accuracy is:  65.4375

Predicted labels of the test data are saved in test_data.labels file using SVM Pearson Correlation feature selection method
Reading test data...
Completed reading test data
saved output results into the predicted_test_data.labels file on the current directory 
kong-45 ProjectPearon >: ls -ltra
total 581132
-rw-r--r-- 1 rp775 afs     54890 Sep  5 01:27 trueclass
-rw-r--r-- 1 rp775 afs 474096000 Sep  5 01:27 traindata
-rw-r--r-- 1 rp775 afs 118524000 Sep  5 01:28 testdata
drwxr-xr-x 4 rp775 afs      4096 Dec  1 19:21 ../
-rw-r--r-- 1 rp775 afs        61 Dec  1 19:21 Output_feature_column.txt
-rw-r--r-- 1 rp775 afs     12890 Dec  1 19:50 test_data.labels
-rw-r--r-- 1 rp775 afs      7952 Dec  1 22:34 project.py
-rw-r--r-- 1 rp775 afs      1702 Dec  1 22:35 README.md
drwxr-xr-x 2 rp775 afs      4096 Dec  1  2018 ./
-rw-r--r-- 1 rp775 afs     12890 Dec  1  2018 predicted_test_data.labels
kong-46 ProjectPearon >: 