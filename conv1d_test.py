# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:35:17 2018

@author: huanghao
"""

import matplotlib.pyplot as plt
from keras.models import load_model  
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.initializers import RandomUniform
from keras.regularizers import l1, l2
from scipy.io import loadmat
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
KTF.set_session(session)

classes = ['BPSK','QPSK','8PSK','OQPSK','PI4QPSK']

def model():    
    model = Sequential()
    model.add(Conv1D(32,20,activation='relu',input_shape=[16000,1]))
    model.add(Conv1D(32,10,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,10,activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(len(classes),activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=sgd)
    model.summary()
    return model

def Z_ScoreNormalization(x,mu,sigma):  
    x = (x - mu) / sigma;  
    return x;  

def load_data(data,num = 2000):
    file = open(data,'rb')
    data = file.read()
    data = np.frombuffer(data,np.int16)
#    data = np.array(data)
    lenth = 16000
    lis = []
    for i in range(num):
        dat = data[i*lenth:i*lenth+lenth]/32768
        lis.append(Z_ScoreNormalization(dat,np.average(dat),np.std(dat)))
#    arr = np.array(data[:1000*num])/32768
    arr = np.array(lis)              
#    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    
#    arr = np.array(arr[:1000*num])
    datat = arr.reshape(-1,16000,1)
    file.close()
    return datat

def get_train_data(filepath,num):
    files = os.listdir(filepath)
    trainx = []
    trainy = []
    testx = []
    testy = []
    for file in files:        
        data = load_data(os.path.join(filepath,file),num)
        data1 = np.array(data[0:8])
        data2 = np.array(data[8:10])
        cl = file.split('_')[0]
  #      print(cl)
        i = classes.index(cl)
        trainx.append(data1)
        trainy.append([i]*np.shape(data1)[0])
        testx.append(data2)
        testy.append([i]*np.shape(data2)[0])
    train_x = np.array(trainx).reshape(-1,16000,1)
    train_y = np.array(trainy).reshape(-1,1)
    train_y = keras.utils.to_categorical(train_y, num_classes=len(classes))
    test_x = np.array(testx).reshape(-1,16000,1)
    test_y = np.array(testy).reshape(-1,1)
    test_y = keras.utils.to_categorical(test_y, num_classes=len(classes))
    
    return train_x,train_y,test_x,test_y

best_model = ModelCheckpoint('./rfns16000_3conv_best_sig5_1.h5', save_best_only=True, 
                                 save_weights_only=True, verbose=1)

stopping = EarlyStopping(patience=8)
lr = ReduceLROnPlateau(verbose=1, factor=0.5, patience = 2)

if __name__ == '__main__':
    model = model()
    num = 10
    train_x,train_y,test_x,test_y = get_train_data('/home/sigdata',num)
    history = model.fit(train_x,train_y, epochs=100, batch_size=64,verbose=2,validation_data = (test_x,test_y),
                        callbacks = [best_model,stopping,lr])

    plt.plot(history.history['val_acc'])

    model.save('new_rfns16000_3conv_best_sig5_1.h5')
    plt.savefig('/home/val_acc5.png')
    
    rst = history.history
    plt.style.use('ggplot')
    plt.figure(figsize=(16, 10))
    plt.plot(rst['acc'], 'r', label = 'train_acc')
    plt.plot(rst['val_acc'], 'b', label = 'test_acc')
    plt.title('Accuracy')
    plt.xlabel('train step')
    plt.legend()
    plt.savefig('16000sig5_cls_acc_1.png')
    
    plt.figure(figsize=(16, 10))
    plt.plot(rst['loss'], 'r', label = 'train_loss')
    plt.plot(rst['val_loss'], 'b', label = 'test_loss')
    plt.title('Loss')
    plt.xlabel('train step')
    plt.legend()
    plt.savefig('16000sig5_cls_loss_1.png')