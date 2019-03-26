#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from keras.models import load_model
from preprocessing import *
import pandas as pd
def read():
    tmp=[]
    fname=[]
    path='test2/test/'
    for i in range(10500):
        print(i)
        wav_file=path+str(i+1)+'.wav'
        wav=wav2mfcc(wav_file)
        tmp.append(wav)
        temp = os.path.split(wav_file)[-1]   
        fname.append(temp)
        
    return np.array(tmp), fname
def transform(listdir,label):
    label_str=[]
    for i in range (10500):
        temp=listdir[label[i]]
        label_str.append(temp)

    return label_str
labels, label_indices,_=get_labels()

model=load_model('ASR.h5')

X,fname=read()
np.save('test.npy', X)
X=np.load('test.npy')
X=X.reshape(X.shape[0],50,32,1)
output=model.predict(X)
y_classes = output.argmax(axis=1)
label_str=transform(labels,y_classes)
for i in range(10500):
    if label_str[i].startswith('_'):
        label_str[i]='unknown'
pd.DataFrame({"id":list(range(1,len(label_str)+1)) ,"word": label_str}).to_csv('test_score.csv', index=False, header=True)