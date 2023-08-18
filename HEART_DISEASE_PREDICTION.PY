#-----------------------------------------about data
#Coronary artery disease symptoms may be different for men and women. 
# For instance, men are more likely to have chest pain. 
# Women are more likely to have other symptoms along with chest discomfort, such as shortness of breath, nausea and extreme fatigue.
#Symptoms of coronary artery disease can include:
#Chest pain, chest tightness, chest pressure and chest discomfort (angina)
#Shortness of breath
#Pain in the neck, jaw, throat, upper belly area or back
#Pain, numbness, weakness or coldness in the legs or arms if the blood vessels in those body areas are narrowed
#You might not be diagnosed with coronary artery disease until you have a heart attack, angina, stroke or heart failure. 
# It's important to watch for heart symptoms and discuss concerns with your health care provider. 
# Heart (cardiovascular) disease can sometimes be found early with regular health checkups.

#------------------------------------------------import useful 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#-------------------------------------------collection of data
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/heart_disease_data.csv")
print(data.columns)
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())#in this data set no null data is here
print(data.head(5))
print(data.tail(5))



#-----------------------------------------separtion od data

x = data.drop(columns="target", axis=1)
y = data["target"]

#----------------------------------------train-test-split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

#---------------------------------------model use
model = LogisticRegression()
model.fit(x_train, y_train)

#---------------------------------------prediction of train data
y_train_p = model.predict(x_train)
print(y_train_p, "this is our prediction values", np.array(y_train), "this is true values")
accur = accuracy_score(y_train_p, y_train)
print(accur, "this is accurancy of train data set")

#--------------------------------------prediction of test data

y_test_p = model.predict(x_test)
print(y_test_p, 'this is our prediction',np.array( y_test), "this is true values")
accur = accuracy_score(y_test_p, y_test)
print(accur, "this is accurancy score of test data")

#----------------------------------------preddiction of single data
x = [57, 0, 0, 140, 241, 0, 1, 123, 1, 0.2, 1, 0, 3]
x_new = np.asarray(x)
x_reshape =x_new.reshape(1, -1)
y_pred = model.predict(x_reshape)
print(y_pred)
















