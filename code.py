# importing basic dependencies
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
# importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values 
# Feature scaling the variables
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x = sc.fit_transform(x) 
# training the regression model on the whole dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x,y)
# prediction
a = int(input("give the age of the user:"))
b = int(input("give the salary of the user:"))
value = classifier.predict(sc.fit_transform([[a,b]]))
for items in value:
    if(items == 0):
        print("This user won't buy the product")
    else:
        print("This user will buy the product")
