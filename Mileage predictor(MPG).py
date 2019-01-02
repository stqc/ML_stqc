#This program is for predicting the mpg(miles per gallon) of a car
#The data set used for this project is available on https://archive.ics.uci.edu/ml/datasets/auto+mpg
#There are two models in this particular project RandomForestReggressor and BaggingRegressor
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

#loading the dataset
df=pd.read_csv(r'/home/prateek/Desktop/auto-mpg.csv')
#Replacing the useless/unknown  values with NaN and dropping them
df.replace(" ",np.nan,inplace=True)
df=df.dropna()
df.replace("?",np.nan,inplace=True)
df=df.dropna()

#dropping the useless columns and separating the features and the target for our model
df=df.drop(['origin','name','year'],axis=1)
target=df.iloc[:,0]
fe=df.iloc[:,1:]

#conversion to a numpy array
fe=np.array(fe)
target=np.array(target)

#splitting the data in training and testing variables with a split ratio of 20%
x_train,x_test,y_train,y_test=train_test_split(fe,target,test_size=.20,random_state=28)

#The RandomforestRegressor model
rfr=RandomForestRegressor(n_estimators=10,min_samples_leaf=1,random_state=25,n_jobs=25,warm_start=True)
l2=rfr.fit(x_train,y_train)
p2=l2.score(x_test,y_test)

#The BaggingRegresor Model
bag=BaggingRegressor(n_estimators=10,warm_start=True,n_jobs=1,random_state=25,)
lrb=bag.fit(x_train,y_train,sample_weight=None)
pb=lrb.score(x_test,y_test)
#85-86%~ accuray for RandomForest and 86-87%~ for BaggingRegressor
print("RandomForest",p2*100,"% error=",mean_squared_error(y_test,l2.predict(x_test)))
print("Bagging",pb*100,"% meanSquared Error =",mean_squared_error(y_test,lrb.predict(x_test)))
