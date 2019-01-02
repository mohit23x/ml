import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

x = np.array([112, 343, 198, 305, 372, 550, 302, 420, 578])
y = np.array([1120, 1623, 2120, 2230, 2600, 3200, 3409, 3689, 4460])

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'ro', color='black')

plt.ylabel('prices')
plt.xlabel('size of houses')

#plt.axis([0, 600, 0, 5000])

plt.plot(x, x*slope+intercept, 'b')



#prediction
newx = 150
newY = newx*slope+intercept

print(newY)

plt.plot(newx, newY,'ro', color='red')

plt.show()
