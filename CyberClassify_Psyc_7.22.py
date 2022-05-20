# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:18:38 2019

@author: Sara
"""
#train traditional E/I in pilot data and test how well it performs in patient data


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

#list with train subject names, change to reflect full list
trainSubjects = ['S4938','S4943','S4947','S4956','S4962','S4966','S4969','S4970','S4976','S4983',
      'S4984','S4990','S4991','S4996','S4997','S4998','S4999','S5002','S5003','S5014',
      'S5019','S5021','S5023','S5027','S5028', 'S5029','S5037','S5044','S5045','S5051',
      'S5065','S5069','S5076','S5088', 'S5089','S5092','S5093', 'S5094','S5111','S5130']

#list with test subject names, change to reflect full list
#including only patients who have a psyc diagnosis
testSubjects = ['S6720','S6715','S6698','S6646','S6541','S6540',
                'S6539','S6482','S6456','S6439','S6435',
                'S6402','S6396','S6367','S6324','S6321','S6274',
                'S6246','S6166','S6151','S6131','S6119','S6101','S6098','S6086',
                'S6079','S6056','S6028','S6009','S6000',
                'S5970','S5945','S5942','S5908','S5906','S5859',
                'S5820','S5801','S5793','S5784','S5771','S5770',
                'S5746','S5691','S5615','S5586','S5574',
                'S5566','S5557','S5534','S5428','S5396',
                'S5361','S5343','S5320','S5250']

#including only patients who do not have a psyc diagnosis
test2Subjects = ['S6727','S6715','S6580','S6560','S6541','S6497','S6410',
                 'S6342','S6282','S6084','S6049','S6047','S6005','S5953',
                 'S5923','S5892','S5859','S5843','S5752','S5731','S5720',
                 'S5577','S5428','S5377']
                

#load the atlas
aal = nilearn.datasets.fetch_atlas_aal()

#initialize lists
trainImageList = []
trainPredictorList = []

#loop through and load trainSubject images
for sub in trainSubjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\TrainData\\Cyberball_Logfiles\\'+ sub + '.xls')
    for i in range(1,21):
        #update column name for the inclusion / exclusion labels
        if subFile.Condition[i-1] == 1 or subFile.Condition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            trainImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\TrainData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            trainPredictorList.append(0)
        if subFile.Condition[i-1] == 2 or subFile.Condition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            trainImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\TrainData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
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
NewCondition = []

#update variable names to match testSubjects
#loop through and load subject test images
for sub in testSubjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\PatientData\\Patientenlogfiles\\'+ sub + '.xls')
    NewCondition = np.array(list(subFile.Condition[subFile.Condition < 6]))
    for i in range(1,15):
        if NewCondition[i-1] == 1 or NewCondition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            testImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\PatientData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            testPredictorList.append(0)
        if NewCondition[i-1] == 2 or NewCondition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            testImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\PatientData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
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

#test 2 - psyc diagnosis controls
#initialize lists
test2ImageList = []
test2PredictorList = []
New2Condition = []

#update variable names to match testSubjects
#loop through and load subject test images
for sub in test2Subjects:
    #loop through and load predictor labels
    #update directory and file name for the inclusion / exclusion labels
    subFile = pd.read_excel('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\PatientData\\Patientenlogfiles\\'+ sub + '.xls')
    New2Condition = np.array(list(subFile.Condition[subFile.Condition < 6]))
    for i in range(1,15):
        if New2Condition[i-1] == 1 or New2Condition[i-1] == 4:
            #make list of images for each subject for each block, only add image if during a performance/free block
            test2ImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\PatientData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list, append 0 for exclusion in either performance or free game
            test2PredictorList.append(0)
        if New2Condition[i-1] == 2 or New2Condition[i-1] == 5:
            #make list of images for each subject for each block, only add image if during a performance/free block
            test2ImageList.append('C:\\Users\\Sara\\Documents\\Actual Documents\\UPenn Second Year\\IRTG\\Cyberball\\PatientData\\' + sub + '\\beta_' + '%04d' %i + '.nii')
            #make predictor list,append 1 for inclusion in either performance or free game
            test2PredictorList.append(1)


#replace NaNs with zeros
cleanTest2Image = nilearn.image.clean_img(test2ImageList, ensure_finite = True)


#initialize masker
masker = NiftiLabelsMasker(aal.maps)
masker.fit()
#apply masker to image list
masker.transform(cleanTest2Image) 

cleanTest2Image = masker.transform(cleanTest2Image)

#initialize classifier
#can look through documentation and see if it makes sense to change any default parameters
clf = sk.svm.SVC(kernel='linear')

#fit classifier to data
clf.fit(cleanTrainImage,trainPredictorList)

#test classifier on psyc diagnosis
y_pred = clf.predict(cleanTestImage)

#test sensitivity & specificity (input: y_true, y_pred)
SensSpec = sk.metrics.classification_report(testPredictorList, y_pred)

#test classifier on psyc diagnosis controls
y_pred2 = clf.predict(cleanTest2Image)

#test sensitivity & specificity (input: y_true, y_pred)
SensSpec2 = sk.metrics.classification_report(test2PredictorList, y_pred2)

