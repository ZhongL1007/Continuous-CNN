#%%1
#Python version：3.7.13
#keras version：2.7.0
#tensorflow version：2.7.0

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os

#%%2
#Build model,Setting parameters, compile and fit
def CNN(modeling_X,modeling_Y,Output_Y):
    model = Sequential()   
    model.add(Conv2D(24,(1,3),strides=(1,1),padding='same',activation='relu',input_shape=(1, modeling_X.shape[2], 1)))
    model.add(Conv2D(24,(1,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2),padding='same'))
        
    model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2),padding='same'))
    
    model.add(Flatten())
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(Output_Y,activation='linear'))
    model.summary()
    
    SaveBestModel = ModelCheckpoint('CNN_BestModel.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001),loss='mse')
    model.fit(modeling_X,modeling_Y,epochs=10000,validation_split=0.25, verbose=2, shuffle=True, callbacks=[SaveBestModel,early_stopping])
    
#%%3
#File path
os.chdir(r"D:")

#Read all data in an excel table
df=pd.read_excel('data.xlsx',sheet_name=0)#Read the divided data
data = df.iloc[:,:].values
x_num = 10#Number of image spectral bands
y_num = 59#Number of feature bands after laboratory spectral pre-processing
STN_column = 70#Column of STN content

#Data set segmentation
modeling_number = 146
modeling = data[:modeling_number]
testing = data[modeling_number:]

#image spectra
modeling_image = modeling[:,:x_num]
modeling_image = modeling_image.reshape(modeling_image.shape[0], 1, x_num, 1)  
testing_image = testing[:,:x_num]
testing_image = testing_image.reshape(testing_image.shape[0], 1, x_num, 1)

#laboratory spectra  
modeling_laboratory = modeling[:,x_num:x_num+y_num]
testing_laboratory = testing[:,x_num:x_num+y_num]

#STN content 
modeling_STN = modeling[:,STN_column-1]
testing_STN = testing[:,STN_column-1]

#%%4
#Spectral transformation
CNN_T = CNN(modeling_image,modeling_laboratory,y_num)
model = tf.keras.models.load_model('CNN_BestModel.h5')

#Modeling result
modeling_predict = model.predict(modeling_image)
pre=pd.DataFrame(data=modeling_predict)
pre.to_excel('Modeling result(Spectral transformation).xlsx')

#Testing result
testing_predict = model.predict(testing_image)
pre=pd.DataFrame(data=testing_predict)
pre.to_excel('testing result(Spectral transformation).xlsx')

#%%5
#STN modeling
modeling_predict_x = modeling_predict.reshape(modeling_predict.shape[0], 1, modeling_predict.shape[1], 1)  
testing_predict_x = testing_predict.reshape(testing_predict.shape[0], 1, testing_predict.shape[1], 1)  
CNN_S = CNN(modeling_predict_x,modeling_STN,1)
model = tf.keras.models.load_model('CNN_BestModel.h5')

#Modeling result
modeling_predict  = model.predict(modeling_predict_x)
pre=pd.DataFrame(data=modeling_predict)
pre.to_excel('Modeling result(STN modeling).xlsx')

#Testing result
testing_predict = model.predict(testing_predict_x)
pre=pd.DataFrame(data=testing_predict)
pre.to_excel('testing result(STN modeling).xlsx')