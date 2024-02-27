# Méthode des moindres carrés :
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_observation = 100
X = np.linspace(0 , 10 , n_observation).reshape((n_observation , 1))
Y = X + np.random.randn(n_observation , 1)
plt.scatter(X , Y)
plt.show()
n = len(X)
x_moy = 0
y_moy = 0
for i in range(n) :
    x_moy += X[i]
    y_moy = x_moy + Y[i]
x_moy = x_moy / n
y_moy = y_moy / n
num = 0
den = 0
for i in range(n) :
    num = (X[i] - x_moy ) * (Y[i] - y_moy)
    den = (X[i] - x_moy )**2
thata1 = num / den
thata0 = y_moy - thata1 * x_moy
print(thata1 , thata0) # [1.05959389] [-0.19394958]

prediction = thata1 * X + thata0
plt.scatter(X , Y)
plt.plot(X , prediction , color = 'red')