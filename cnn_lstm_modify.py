import time
starttime=time.time()
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
#from keras.optimizers import adam
import matplotlib.pyplot as plt
import matplotlib#import seaborn as sns
import pickle, random, sys, keras
import sys
from imp import reload
reload(sys)
import os,random
import numpy as np
import pickle, random, sys
import numpy as np
from keras import layers
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam,Adamax,Nadam
from keras  import optimizers
import numpy as np
import time
from keras.layers.normalization import BatchNormalization
starttime=time.time()
from keras.utils import np_utils

import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D,ZeroPadding2D
from keras.regularizers import *#from keras.optimizers import adam
#import matplotlib.pyplot as plt
#import matplotlib#import seaborn as sns
import pickle, random, sys, keras
import sys
import scipy.io as sio

from keras.layers.convolutional import Convolution2D as Conv2D
from imp import reload
reload(sys)
#from mozi.layers.recurrent import BiLSTM, LSTM
import os,random
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np
from keras.models import Model
import pickle, random, sys
from keras.layers import LSTM,TimeDistributed,Dense,GRU,Bidirectional
Xd = pickle.load(open("2016.04C.multisnr.pkl",'rb'),encoding='iso-8859-1')

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
snrs=snrs[8:]#adde by wjx
#mods=['8PSK','BPSK','QAM16','QAM64','QPSK']
X_train = [] 
y_tlb=[]
y_elb=[] 
X_test=[]
mod_num={'16APSK':0,'32APSK':1, '8PSK':2, 'BPSK':3, 'QAM16':4, 'QAM32':5, 'QPSK':6}
lbl = []
res=[0]*100
z=0
n_examples =84000
np.random.seed(2016)
n_train = n_examples*8//10
adam=Adam(0.005+0.000)
for mod in mods:
    for snr in snrs:
        X_train.append(Xd[(mod,snr)][:700])
        for i in range(int((Xd[(mod,snr)][:700].shape[0]))):  
            lbl.append((mod,snr))
            y_tlb.append(mod_num[mod])
X_train = np.vstack(X_train)
for mod in mods:
    for snr in snrs:
        X_test.append(Xd[(mod,snr)][700:])
        for i in range(int((Xd[(mod,snr)][700:].shape[0]))):  
            lbl.append((mod,snr))
            y_elb.append(mod_num[mod])
X_test = np.vstack(X_test)
##xi=X[:,0,:]
##xq=X[:,1,:]
##X=np.hstack([xi,xq])
###sio.savemat('USRP_commandata.mat',{'X':X})
##sio.savemat('wodedata.mat',{'X':X})
#

#            train_idx = np.random.choice(range(0,int(n_examples)), size=int(n_examples*0.7), replace=False)
#train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
#test_idx = list(set(range(0,n_examples))-set(train_idx))
#X_train = X[train_idx]#train set 
#n_train_N=int(n_train/4)
#train_idx_n4=np.random.choice(range(0,n_train), size=n_train_N, replace=False)
#train_idx_rem=np.array(list(set(range(0,n_train))-set(train_idx_n4)))
##train_idx_n6=train_idx_rem[np.random.choice(range(0,len(train_idx_rem)), size=n_train_N, replace=False)]
#train_idx_n6=train_idx_rem[np.random.choice(range(0,len(train_idx_rem)), size=n_train_N, replace=False)]
#
#rem1=np.array(list(set(range(0,n_train))-set(train_idx_n4)
#-set(train_idx_n6)))
#train_idx_n8=rem1[np.random.choice(range(0,len(rem1)), size=n_train_N, replace=False)]
#train_idx_n16=np.array(list(set(range(0,n_train))-set(train_idx_n4)
#-set(train_idx_n6)-set(train_idx_n8)))
#X_train1=X_train
#n_test=n_examples-n_train
#X_test =  X[test_idx]#test set
#X_test1=X_test
#test_idx_n8=np.random.choice(range(0,n_test), size=n_test, replace=False)
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(y_tlb)
Y_test = to_onehot(y_elb)
in_shp = list(X_train.shape[1:])
classes = mods    
dr = 0.65
TIME_STEPS = 2     # same as the height of the image
INPUT_SIZE = 128     # same as the width of the image
BATCH_SIZE = 550
BATCH_INDEX = 0
OUTPUT_SIZE = 7
CELL_SIZE = 60
model1 = models.Sequential()
model2 = models.Sequential()
model1.add(Reshape(in_shp+[1], input_shape=in_shp))#[1]+[2,128]=[1,2,128]
model2.add(Reshape(in_shp, input_shape=in_shp))#[1]+[2,128]=[1,2,128]
#model.add(ZeroPadding2D((0, 2)))
model1.add(Convolution2D(128, 2,5, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
model1.add(Dropout(dr))
##            model.add(ZeroPadding2D((0, 2)))

#            model1.add(Convolution2D(105, (1, 3), border_mode="valid",activation="relu", name="conv2", init='glorot_uniform'))
#            model1.add(Dropout(0.25+0.05*k_size))
#            model1.add(Convolution2D(105, (1, 3), border_mode="valid",activation="relu", name="conv3", init='glorot_uniform'))
#            model1.add(Dropout(0.25+0.05*k_size))
#          
model1.add(AveragePooling2D(pool_size=(1,2),strides=2))

#            model1.add(Dense(256,activation='relu'))
model2.add(LSTM(48, kernel_initializer='orthogonal', 
              input_dim=INPUT_SIZE, input_length=TIME_STEPS,      # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
#              output_dim=128,
#                          unroll=True,
#                return_sequences=True,
              bias_initializer='ones', 
              dropout=0.025, recurrent_dropout=0.20))
model2.add(Dropout(0.025))

#            model2.add(LSTM(24,activation='tanh',dropout=0.025, return_sequences=True,recurrent_dropout=0.25))
##            model2.add(Dropout(0.25+0.05*k_size))
#model2.add(LSTM(12,activation='tanh',dropout=0.025, recurrent_dropout=0.25))

model1.add(Flatten())
conc=(layers.Concatenate()([model1.layers[4].output,model2.layers[2].output]))
conc=Dense(256,activation='relu')(conc)
#            model1.add(Dropout(0.4))
out=Dense(len(classes),activation='softmax')(conc)
model=Model(inputs=[model1.input,model2.input],outputs=out)
#model.add(ZeroPadding2D((0, 2)))
#model.add(Convolution2D(40, 2, 3, border_mode="same", activation="relu", name="conv3", init='glorot_uniform'))
#model.add(Dropout(dr))
#model.add(ZeroPadding2D((0, 2)))
#model.add(Convolution2D(20, 2, 3, border_mode="same", activation="relu", name="conv4", init='glorot_uniform'))
#model.add(Dropout(dr))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=adam)
#            model.summary()
nb_epoch = 50# number of epochs to train on
batch_size = 250+50#350# training batch size660-96.52
history = model.fit([X_train,X_train],
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,#modified by wjx
    verbose=2,
    validation_data=([X_test,X_test], Y_test),
    shuffle=True,
    )
score = model.evaluate([X_test,X_test], Y_test, 
                       verbose=0, batch_size=batch_size)
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_Y_hat = model.predict([X_test,X_test], batch_size=batch_size)
count=0
for i in range(0,test_Y_hat.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    if k==j:
        count=count+1
#acc = {}
#for snr in snrs:
#        test_SNRs = map(lambda x: lbl[x][1], test_idx)
#        test_X_i = X_test[np.where(np.array(list((test_SNRs)))==snr)]
#        test_SNRs = map(lambda x: lbl[x][1], test_idx)
#        test_Y_i = Y_test[np.where(np.array((list(test_SNRs)))==snr)]    
#        test_Y_i_hat = model.predict([test_X_i,test_X_i])
#        conf = np.zeros([len(classes),len(classes)])
#        confnorm = np.zeros([len(classes),len(classes)])
#        for i in range(0,test_X_i.shape[0]):
#            j = list(test_Y_i[i,:]).index(1)
#            k = int(np.argmax(test_Y_i_hat[i,:]))
#            conf[j,k] = conf[j,k] + 1
#        for i in range(0,len(classes)):
#            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
##                plt.figure() 
##                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
##                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix ")
#        cor = np.sum(np.diag(conf))
#        ncor = np.sum(conf) - cor   
##                print ("Overall Accuracy: ", cor / (cor+ncor))
#        acc[snr] = 1.0*cor/(cor+ncor)
#acc1 = {}
#for mod in mods:
#    test_SNRs = map(lambda x: lbl[x][0], test_idx)
#    test_X_i = X_test[np.where(np.array(list((test_SNRs)))==mod)]
#    test_SNRs = map(lambda x: lbl[x][0], test_idx)
#    test_Y_i = Y_test[np.where(np.array((list(test_SNRs)))==mod)]    
#    test_Y_i_hat = model.predict([test_X_i,test_X_i])
#    conf = np.zeros([len(classes),len(classes)])
#    confnorm = np.zeros([len(classes),len(classes)])
#    count=0
#    for i in range(0,test_X_i.shape[0]):
#            j = list(test_Y_i[i,:]).index(1)
#            k = int(np.argmax(test_Y_i_hat[i,:]))
#            if j==k:
#                count=count+1;
#            conf[j,k] = conf[j,k] + 1
#    for i in range(0,len(classes)):
#            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
##                plt.figure() 
##                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
##                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix ")
#    cor = np.sum(np.diag(conf))
#    ncor = np.sum(conf) - cor   
##                print ("Overall Accuracy: ", cor / (cor+ncor))
#    acc1[mod]=count/test_X_i.shape[0]
acc2 = {}
for snr in snrs:
    for mod in mods:
        test_X_i = X_test[((mod_num[mod]+1)*(snrs.index(snr)+1)-1)*300:(mod_num[mod]+1)*(snrs.index(snr)+1)*300]
        test_Y_i = Y_test[((mod_num[mod]+1)*(snrs.index(snr)+1)-1)*300:(mod_num[mod]+1)*(snrs.index(snr)+1)*300]    
        test_Y_i_hat = model.predict([test_X_i,test_X_i])
        count=0
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            if j==k:
                count=count+1
#                plt.figure() 
#                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
#                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix ")
#                print ("Overall Accuracy: ", cor / (cor+ncor))
#        acc[snr] = 1.0*cor/(cor+ncor)
        acc2[(snr,mod)]=count/test_X_i.shape[0]
##    #            print (acc)
##    #            print(sum(map(lambda x:acc[x],snrs))/len(list(map(lambda x:acc[x],snrs))))
##                
###                res[z]=acc
###                z=z+1
###            canshu=[]
#endtime=time.time()
#print(endtime-starttime)
##            ##fd = open('results_cnn2_d0.5.dat','wb')
##            ##pickle.dump( ("CNN2", 0.5, acc) , fd )
##            ##plt.plot(snrs,map(lambda x:acc[x],snrs))
##            ##plt.xlabel("信号信噪比/dB")
##            ##plt.ylabel("正确识别率")
##
