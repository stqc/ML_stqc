#importing the required libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#reading the CSV as a dataframe with pandas
df=pd.read_csv(r'/home/prateek/Desktop/breast-cancer-wisconsin.csv',header=None)
df=df.iloc[:,1:]
#dropping unncessary values
df.replace("?", np.nan, inplace=True)
df=df.dropna()

#separating features and targets
attributes=df.iloc[:,:9]
target=df.iloc[:,-1]

#converting to an array
x=np.array(attributes)
y=np.array(target)

#splitting the training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=11)


knn=KNeighborsClassifier(n_neighbors=9)
#fitting the data to our knn model
learner=knn.fit(x_train,y_train)

#accurcacy score
print(accuracy_score(y_test,learner.predict(x_test))*100)

