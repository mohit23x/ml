import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

x = np.array([[10000, 80000, 35], [7000, 1200000, 57], [100, 23000, 22], [223, 18000, 26]])
y = np.array([1, 1, 0, 0])

clf = LogisticRegression()

clf.fit(x, y)

z = np.array([[5500, 50000, 25]])

print(clf.predict(z))
