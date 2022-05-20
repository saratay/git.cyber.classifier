# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:51:31 2019

@author: Sara
"""

import os
#machine learning in neuroimaging 
import nilearn

#niftimasker puts 3D labels into 1D array
from nilearn.input_data import NiftiLabelsMasker

#reading neuroimaging files
import nibabel as nib

import numpy as np

import pandas as pd

import sklearn as sk

#list with train subject names, train in best friends
trainSubjects = ['S5168','S5184','S5234','S5242','S5243','S5259','S5296','S5327','S5360',
       'S5381','S5435','S5436','S5496','S5563','S5613','S5292']

#list with test subject names, change to reflect full list
#test in pilot data
testSubjects = ['S4938','S4943','S4947','S4956','S4962','S4966','S4969','S4970','S4976','S4983',
      'S4984','S4990','S4991','S4996','S4997','S4998','S4999','S5002','S5003','S5014',
      'S5019','S5021','S5023','S5027','S5028', 'S5029','S5037','S5044','S5045','S5051',
      'S5065','S5069','S5076','S5088', 'S5089','S5092','S5093', 'S5094','S5111','S5130']

#load the atlas
aal = nilearn.datasets.fetch_atlas_aal()

#initialize lists
trainImageList = []
trainPredictorList = []

#loop through and load trainSubject images
for sub in trainSubjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\FirstLevel_CB3\\Cyberball_Methoden_Logfiles\\'+ sub + '_Makro.xls')
    for i in range(1,17):
        #update column name for the inclusion / exclusion labels
        if subFile.Condition[i-1] == 1 or subFile.Condition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            trainImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\FirstLevel_CB3\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            trainPredictorList.append(0)
        if subFile.Condition[i-1] == 2 or subFile.Condition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            trainImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\FirstLevel_CB3\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list,append 1 for inclusion in either performance or free game
            trainPredictorList.append(1)
            
#replace NaNs with zeros
cleanTrainImage = nilearn.image.clean_img(trainImageList, ensure_finite = True)

#initialize masker
masker = NiftiLabelsMasker(aal.maps)
masker.fit()
#apply masker to image list
cleanTrainImage = masker.transform(cleanTrainImage)

#initialize lists
testImageList = []
testPredictorList = []

#update variable names to match testSubjects
#loop through and load subject test images
#test in pilot data
for sub in testSubjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\TrainData\\Cyberball_Logfiles\\'+ sub + '.xls')
    for i in range(1,21):
        if subFile.Condition[i-1] == 1 or subFile.Condition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            testImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\TrainData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            testPredictorList.append(0)
        if subFile.Condition[i-1] == 2 or subFile.Condition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            testImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\TrainData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list,append 1 for inclusion in either performance or free game
            testPredictorList.append(1)
    
        
#replace NaNs with zeros
cleanTestImage = nilearn.image.clean_img(testImageList, ensure_finite = True)


#initialize masker
masker = NiftiLabelsMasker(aal.maps)
masker.fit()
#apply masker to image list
masker.transform(cleanTestImage) 

cleanTestImage = masker.transform(cleanTestImage)



clf = sk.svm.SVC(kernel='linear')

#fit classifier to data
clf.fit(cleanTrainImage,trainPredictorList)

#test pilot data
y_pred = clf.predict(cleanTestImage)
#test sensitivity & specificity (input: y_true, y_pred)
SensSpec = sk.metrics.classification_report(testPredictorList, y_pred)


#map clf weights to regions
regionWeights = zip(aal.labels, clf.coef_)

#dataframe out of region weights
regionDF = pd.DataFrame(data = clf.coef_, columns = aal.labels)

#sort by absolute value of coefficients
regionDFsort = regionDF.abs()
regionDFsort = regionDFsort.sort_values (by = 0, axis = 1, ascending = False)

