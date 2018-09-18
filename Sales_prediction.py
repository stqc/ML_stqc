#A sample program for predicting the sales based on various adverstisement expenditure
#The data set used for making this model was taken from https://www.github.com/chandanverma07/DataSets
#importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#importing data using pandas and extracting nencessary data from the imported data
adv=pd.read_csv(r'C:/Users/admin/Desktop/DataSets-master/DataSets-master/Advertising.csv')
expen=adv[["TV","radio","newspaper"]]
sale=adv["sales"]
print("The following data is used for the model:\n\n",adv[["TV","radio","newspaper","sales"]])
#creating arrays from the data
exp_arr=expen.values.reshape(-1,3)
sales_arr=sale.values.reshape(-1,1)

#splitting the data into test and train
exp_train,exp_test,sale_train,sale_test=train_test_split(exp_arr,sales_arr,test_size=.30)

#creating a teacher and learner
teacher=LinearRegression()
learner=teacher.fit(exp_train,sale_train)
predicted=learner.predict(exp_test)

#finding the margin of error
error=np.sqrt(mean_squared_error(sale_test,predicted))

#creating a dataframe to compare actual and predicted sales
actual=list(sale_test)
predict=list(predicted)
avp=pd.DataFrame({"Actual Sales":actual,"Predicted Sales":predict})

#Ploting a graph
plt.plot(exp_test,sale_test,"go")
plt.plot(exp_test,predicted,"r*")
plt.ylabel("Sales(actual/predicted)")
plt.xlabel("Advertisement Exp")
plt.legend(["Actual Sales","Predicted Sales"])
plt.show()

#printing the error
print("Estimated error is:",error)

#printing the dataframe
print("\n\nthe following table shows the actual vs predicted sales\n\n",avp)