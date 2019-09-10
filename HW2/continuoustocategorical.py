import pandas as pd
import numpy as np

train = pd.read_csv("creditcard_train.csv")
test = pd.read_csv("creditcard_test.csv")
X_train = train.drop(['Class'], axis=1)
X_test = test.drop(['Class'], axis=1)

medians = X_train.median()
counterName = 0
columnName = "V" + str(counterName)

for median in medians:
    counterName += 1
    print(median)
    columnName = "V" + str(counterName)
    for index, row in X_train.iterrows():
       if row[columnName] > median:
           row[columnName] = 1
       else:
           row[columnName] = 0
    for index, row in X_test.iterrows():
       if row[columnName] > median:
           row[columnName] = 1
       else:
           row[columnName] = 0

X_train.to_csv(path_or_buf="creditCard_Train_Binary.csv")
X_test.to_csv(path_or_buf="creditCard_Test_Binary.csv")
