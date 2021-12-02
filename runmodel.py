#!/usr/bin/env
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report


def runmodel():
    data = pd.read_csv('ValidationDataset.csv', sep=';')
    X_test = data.drop(['quality'], axis=1)
    Y_test = data['quality']
    from sklearn.preprocessing import StandardScaler
    norm = StandardScaler()
    ATest = norm.fit_transform(X_test)
    BTest = data['quality']
    model = load('./savedData')
    Y_pred = model.predict(ATest)
    print(classification_report(BTest, Y_pred))


if __name__ == '__main__':
    runmodel()
