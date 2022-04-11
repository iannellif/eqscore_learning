import pandas as pd
import numpy as np



#number of examples
m=100
#number of features (semantics + flavours)
n=4



np.random.seed(0)
xsem = 1- 0.5 * np.random.randint(0,3, size=(1, m))

np.random.seed(0)
xfla = 1- 0.5 * np.random.randint(2, size=(n-1, m))

x = np.concatenate((xsem,xfla), axis=0)

xfull = np.concatenate((np.asarray([np.ones(m)]),x), axis=0)

np.random.seed(0)
y = (np.random.randint(50,100,m)/100).round(1)




# # Linear Model

np.random.seed(0)
weights = np.random.randint(1,10,n+1)
wnorm = weights/sum(weights)

ylin = np.dot(xfull.T,wnorm).round(1)



# # Datasets

data_label = np.concatenate((x, np.asarray([ ylin ])), axis=0)
data_unlab = x


colnames_label = ['semantics', 'frequency', 'vulgarity', 'universality', 'score']
colnames_unlab = ['semantics', 'frequency', 'vulgarity', 'universality']

dataset_label = pd.DataFrame(data_label.T,columns=colnames_label)

path = 'data/'
filename = 'data_eqs_random_label.csv'
dataset_label.to_csv(path+filename,index=False)

dataset_unlab = pd.DataFrame(data_unlab.T,columns=colnames_unlab,dtype=float)

path = 'data/'
filename = 'data_eqs_random_unlab.csv'
dataset_unlab.to_csv(path+filename,index=False)



