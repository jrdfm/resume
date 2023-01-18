#!/usr/bin/env python

"""
ENEE 436: Project 1 -- Loading the datasets in python
"""

#%% Import Modules

# import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

# plt.ioff() 
plt.close('all')

#%% Problem Parameters

# Dataset
data_folder = './Data/' 

# Test Ratio
test_ratio = 0.2

# Random Seed
np.random.seed(13)

#%% LOAD FACE DATA

Ns = 200 
face = loadmat(data_folder+'data.mat')['face']
face_n = [face[:,:,3*n] for n in range(Ns)] # neutral
face_x = [face[:,:,3*n+1] for n in range(Ns)] # expression
face_il = [face[:,:,3*n+2] for n in range(Ns)] # illumination variation

if True:
    fig, axs = plt.subplots(3)
    fig.suptitle('DATA: Example of the 3 images of a random subject')
    i = random.choice(np.arange(Ns))
    axs[0].imshow(face_n[i])
    axs[1].imshow(face_x[i])
    axs[2].imshow(face_il[i])

# Convert the dataset in data vectors and labels for 
# neutral VS facial expression classification
data = []
labels = []
for subject in range(Ns):
    # neutral face: label 0
    data.append(face_n[subject].reshape(-1))
    labels.append(0)
    # face with expression: label 1
    data.append(face_x[subject].reshape(-1))
    labels.append(1)

# Split to train and test data
N = int( (1-test_ratio)*len(data) )
idx = np.arange(len(data))
random.shuffle(idx)
train_data = [data[i] for i in  idx[:N]]
train_labels = [labels[i] for i in  idx[:N]]
test_data = [data[i] for i in  idx[N:]]
test_labels = [labels[i] for i in  idx[N:]]

#%% LOAD POSE DATA
    
pose = loadmat(data_folder+'pose.mat')['pose']

# Show some examples of the dataset 
if True:
    fig, axs = plt.subplots(3)
    fig.suptitle('POSE: Example of 3 images of a random subject')
    s = random.choice(np.arange(pose.shape[3]))
    axs[0].imshow(pose[:,:,0,s])
    axs[1].imshow(pose[:,:,1,s])
    axs[2].imshow(pose[:,:,2,s])

# Convert the dataset in data vectors and labels for subject identification    
data = []
labels = []
for subject in range(pose.shape[3]):
    for image in range(pose.shape[2]):
        data.append(pose[:,:,image,subject].reshape(-1))
        labels.append(subject)

# Split to train and test data
N = int( (1-test_ratio)*len(data) )
idx = np.arange(len(data))
random.shuffle(idx)
train_data = [data[i] for i in  idx[:N]]
train_labels = [labels[i] for i in  idx[:N]]
test_data = [data[i] for i in  idx[N:]]
test_labels = [labels[i] for i in  idx[N:]]
        
#%% LOAD ILLUMINATION DATA

illum = loadmat(data_folder+'illumination.mat')['illum']

# Show some examples of the dataset 
if True:
    fig, axs = plt.subplots(3)
    fig.suptitle('ILLUMINATION: Example of 3 images of a random subject')
    s = random.choice(np.arange(illum.shape[2]))
    axs[0].imshow(illum[:,0,s].reshape((48,40),order='F'))
    axs[1].imshow(illum[:,1,s].reshape((48,40),order='F'))
    axs[2].imshow(illum[:,2,s].reshape((48,40),order='F'))

# Convert the dataset in data vectors and labels for subject identification
data = []
labels = []
for subject in range(illum.shape[2]):
    for image in range(illum.shape[1]):
        data.append(illum[:,image,subject])
        labels.append(subject)

# Split to train and test data        
N = int( (1-test_ratio)*len(data) )
idx = np.arange(len(data))
random.shuffle(idx)
train_data = [data[i] for i in  idx[:N]]
train_labels = [labels[i] for i in  idx[:N]]
test_data = [data[i] for i in  idx[N:]]
test_labels = [labels[i] for i in  idx[N:]]
        
        
        
        
        