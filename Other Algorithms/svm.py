from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("processed.cleveland.dataclean_standardized_nosymmetrybreaking.csv")
clf = svm.SVC()
X_train = df.iloc[:179,1:-1]
print(X_train.shape)
print(X_train.head())

Y_train = df['result'][:179]
print(Y_train.shape)
print(Y_train.head())

X_test = df.iloc[179+59:,1:-1]
print(X_test.shape)
print(X_test.head())

Y_test = df['result'][179+59:]
print(Y_test.shape)
print(Y_test.head())


clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
target_names =['class 0', 'class 1']
print(classification_report(Y_test, y_pred, labels=[0,1],
                            target_names=target_names,
                            digits=4))


