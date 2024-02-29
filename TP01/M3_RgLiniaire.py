# La descente de gradient :
import numpy as np
import matplotlib.pyplot as plt

def fonction_cout(X, y, theta):
    n = len(y)
    predictions = X.dot(theta)
    cout = (1 / (2 * n)) * np.sum(np.square(predictions - y))
    return cout


def gradient(X, y, theta):
    m = len(y)
    return (1 / m) * X.T.dot(X.dot(theta) - y)


def descente_gradient(X, y, theta, alpha, n_iterations):
    j_historique = np.zeros((n_iterations, 1))
    for i in range(n_iterations):
        gradient_theta = gradient(X, y, theta)
        theta = theta - alpha * gradient_theta
        j_historique[i] = fonction_cout(X, y, theta)
    return theta, j_historique


np.random.seed(0)
n_observation = 100
X = np.linspace(0, 10, n_observation).reshape((n_observation, 1))
y = X + np.random.randn(n_observation, 1)

X_biais = np.c_[np.ones((n_observation, 1)), X]

alpha = 0.02
n_iterations = 1000

theta_init = np.array([[0], [0]])

theta_final, j_historique = descente_gradient(X_biais, y, theta_init, alpha, n_iterations)

print("Paramètres finaux (theta):", theta_final) # [[0.20804923][0.9703308 ]]
plt.plot(range(n_iterations), j_historique,)
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Fonction Coût')
plt.title('Descente de Gradient')
plt.show()

def calcul_perfformence ( Ypred , Y ) :
    n = len(Y)
    A = 0
    B = 0
    for i in range(n) :
        A += (Ypred[i] - Y[i] ) ** 2
        B += (Ypred[i] - np.mean(Y) ) ** 2
    r = 1 - A/B
    return r

# coef_det = calcul_perfformence(prediction , y)
# print(f"La perfremence de votre model de regression liniaire est : {coef_det}")