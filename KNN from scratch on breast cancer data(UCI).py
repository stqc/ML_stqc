import numpy as np
from collections import Counter
import pandas as pd
import random

def knn(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('K is less than voting groups')
    
    dist=[]
    for group in data:
        for features in data[group]:
            euclidean_dist=np.linalg.norm(np.array(features)-np.array(predict))
            dist.append([euclidean_dist,group])

    votes= [i[1] for i in sorted(dist) [:k]]
    vote_result=Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv(r'/home/prateek/Desktop/breast-cancer-wisconsin.csv',header=None)
df.replace('?',np.nan,inplace=True)
df=df.dropna()
df=df.iloc[:,1:]

full_data=df.astype(float).values.tolist()

random.shuffle(full_data)

test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}

train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])


correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote= knn(train_set,data,k=9)
        if group == vote:
            correct+=1
        total+=1
print('accuracy:',correct/total)
