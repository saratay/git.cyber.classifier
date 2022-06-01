# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:18:38 2019

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

import scipy.stats as st


#list with train subject names, change to reflect full list
trainSubjects = ['S4938','S4943','S4947','S4956','S4962','S4966','S4969','S4970','S4976','S4983',
      'S4984','S4990','S4991','S4996','S4997','S4998','S4999','S5002','S5003','S5014',
      'S5019','S5021','S5023','S5027','S5028', 'S5029','S5037','S5044','S5045','S5051',
      'S5065','S5069','S5076','S5088', 'S5089','S5092','S5093', 'S5094','S5111','S5130']

#list with test subject names, change to reflect full list
testSubjects = ['S5168','S5184','S5234','S5242','S5243','S5259','S5296','S5327','S5360',
       'S5381','S5435','S5436','S5496','S5563','S5613','S5169','S5202','S5207','S5210',
       'S5225','S5254','S5264','S5269','S5271','S5273','S5278','S5298','S5307','S5319','S5364',
       'S5444', 'S5457', 'S5476']

#load the atlas
aal = nilearn.datasets.fetch_atlas_aal()

#initialize lists
trainImageList = []
trainPredictorList = []

#loop through and load trainSubject images
for sub in trainSubjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('\\TrainData\\Cyberball_Logfiles\\'+ sub + '.xls')
    for i in range(1,21):
        #update column name for the inclusion / exclusion labels
        if subFile.Condition[i-1] == 1 or subFile.Condition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            trainImageList.append('\\TrainData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            trainPredictorList.append(0)
        if subFile.Condition[i-1] == 2 or subFile.Condition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            trainImageList.append('\\TrainData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list,append 1 for inclusion in either performance or free game
            trainPredictorList.append(1)

#replace NaNs with zeros
cleanTrainImage = nilearn.image.clean_img(trainImageList, ensure_finite = True)

#initialize masker
masker = NiftiLabelsMasker(aal.maps)
masker.fit()
#apply masker to image list
cleanTrainImage = masker.transform(cleanTrainImage)

#initialize classifier
clf = sk.svm.SVC(kernel='linear')

#fit classifier to data
clf.fit(cleanTrainImage,trainPredictorList)

#initialize lists
testImageList = []
testPredictorList = []
precisionList = []
recallList = []
f1scoreList = []

#update variable names to match testSubjects
#loop through and load subject test images
for sub in testSubjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('\\FirstLevel_CB3\\Cyberball_Methoden_Logfiles\\'+ sub + '_Makro.xls')
    for i in range(1,17):
        if subFile.Condition[i-1] == 1 or subFile.Condition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            testImageList.append('\\FirstLevel_CB3\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            testPredictorList.append(0)
        if subFile.Condition[i-1] == 2 or subFile.Condition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            testImageList.append('\\FirstLevel_CB3\\' + sub + '\\beta_' + '%04d' %i + '.nii')
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
    #test data
    y_pred = clf.predict(cleanTestImage)
    #test sensitivity and specificity for each subject
    SensSpec = sk.metrics.classification_report(testPredictorList, y_pred)
    precisionList.append(float(SensSpec[179:183]))
    recallList.append(float(SensSpec[189:193]))
    f1scoreList.append(float(SensSpec[199:203]))
    
#calculate confidence interval in our sample for precision, recall, and f1score
precisionCI = st.t.interval(0.95, len(precisionList)-1, loc=np.mean(precisionList), scale=st.sem(precisionList))
recallCI = st.t.interval(0.95, len(recallList)-1, loc=np.mean(recallList), scale=st.sem(recallList))
f1scoreCI = st.t.interval(0.95, len(f1scoreList)-1, loc=np.mean(f1scoreList), scale=st.sem(f1scoreList))

