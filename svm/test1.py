import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm


names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

df = pd.read_csv('./iris.data.txt', header=None, names=names)

x = np.array(df.ix[:, 0:4])
y = np.array(df['class'])

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=43)

knn = svm.SVC()

knn.fit(x_train, y_train)

pred = knn.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred) * 100 )


