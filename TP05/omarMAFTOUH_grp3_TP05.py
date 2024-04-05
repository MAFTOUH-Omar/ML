import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('mnist.csv')

n, m = data.shape
np.random.shuffle(data.values)

######### test data #########
data_test = data.iloc[0:1000].values
X_test = data_test[:, 1:].T
y_test = data_test[:, 0].T
X_test = X_test / 255

######### train data #########
data_train = data.iloc[1000:n].values
X_train = data_train[:, 1:].T
y_train = data_train[:, 0].T
X_train = X_train / 255

def initialisation():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(z, 0)

def softmax(z):
    A = np.exp(z) / np.sum(np.exp(z), axis=0)
    return A

def prop_avant(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def derive_relu(z1, a1, z2, a2, w1, w2, X, y):
    m = X.shape[1]
    y_encoded = np.eye(10)[y].T
    dz2 = a2 - y_encoded
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (z1 > 0)
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2

def mettre_a_jour(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.size

def prediction(A2):
    return np.argmax(A2, axis=0)

def descent_gradient(X, Y, alpha, iteration):
    W1, b1, W2, b2 = initialisation()
    for i in range(iteration):
        Z1, A1, Z2, A2 = prop_avant(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = derive_relu(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = mettre_a_jour(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("iteration", i)
            y_pred = prediction(A2)
            print(accuracy(y_pred, Y))
    return W1, b1, W2, b2

def new_prediction(X, W1, b1, W2, b2):
    _, _, _, A2 = prop_avant(W1, b1, W2, b2, X)
    y_pred = prediction(A2)
    return y_pred

def test_prediction(index, W1, b1, W2, b2):
    prediction = new_prediction(X_test[:, index, None], W1, b1, W2, b2)
    label = y_test[index]
    print("Prediction:", prediction)
    print("Label:", label)

W1, b1, W2, b2 = descent_gradient(X_train, y_train, 0.1, 500)
test_prediction(6, W1, b1, W2, b2)