# from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

trainData = pd.read_csv("./TrainingDataset.csv", sep=';')
testData = pd.read_csv("./ValidationDataset.csv", sep=';')

# Create Classification version of target variable
#df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
# Separate feature variables and target variable
A = trainData.drop(['quality'], axis=1)
B = trainData['quality']
ATest = testData.drop(['quality'], axis=1)
BTest = testData['quality']

print(trainData['quality'].value_counts())
print(testData['quality'].value_counts())


# Normalize feature variables
# from sklearn.preprocessing import StandardScaler
# X_features = X
# X = preprocessing.scale(X)

# creating scaler scale var.
norm = StandardScaler()
# fit the scale
norm_fit = norm.fit(A)
# transformation of training data
A = norm_fit.transform(A)
# transformation of testing data
norm_fit = norm.fit(ATest)
ATest = norm_fit.transform(ATest)

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(
    A, B, test_size=.15, random_state=3)

# Logistic Regression
# from sklearn.linear_model import LogisticRegression
# logisticRegr = LogisticRegression(solver = 'lbfgs')
# logisticRegr.fit(X_train, y_train)
# y_pred2 = logisticRegr.predict(X_test)
# print(classification_report(y_test, y_pred2))

# Random Forset
model = RandomForestClassifier(n_estimators=500, random_state=0)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
Y_pred = model.predict(ATest)
print(classification_report(BTest, Y_pred))
print(BTest, Y_pred)


# SVC
# from sklearn.svm import SVC
# svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
# svc2.fit(X_train, y_train)
# pred_svc2 = svc2.predict(X_test)
# print(classification_report(y_test, pred_svc2))

# Serialize model
dump(model, './savedData')
