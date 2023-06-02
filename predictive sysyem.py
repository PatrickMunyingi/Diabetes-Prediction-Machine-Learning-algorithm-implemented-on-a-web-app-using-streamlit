# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#Loading the saved model
loaded_model=pickle.load(open('C:/Users/HP/Diabetes Prediction/trained_model.sav','rb'))



input_data=(8,183,64,0,0,23.3,0.672,32)


#Change the input data to a numpy array
input_data_np=np.asarray(input_data)

#Reshape the array as we are predicting for one instance
input_data_reshape=input_data_np.reshape(1,-1)



prediction=loaded_model.predict(input_data_reshape)

print(prediction)

if (prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")