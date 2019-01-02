import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = numpy.random.RandomState(1000)
x = 4 * rng.rand(100)
y = 4 * x - 1 + rng.randn(100)

model = LinearRegression(fit_intercept = True)
model.fit(x[:, numpy.newaxis], y)

xfit = numpy.linspace(0, 5, 50)
yfit = model.predict(xfit[:, numpy.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
