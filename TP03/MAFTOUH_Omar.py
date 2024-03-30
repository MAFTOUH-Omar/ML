import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
epsilon = 1e-6

class AnalyseDiscriminante():
    def __init__(self):
        self.means = None
        self.covariances = None
        self.priors = None

    def fit(self, X, y):
        self.means = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
        self.covariances = np.array([np.cov(X[y == c], rowvar=False) for c in np.unique(y)])
        self.priors = np.array([np.mean(y == c) for c in np.unique(y)])

    def discriminant_function(self, x):
        discriminants = []
        for mean, covariance, prior in zip(self.means, self.covariances, self.priors):
            det_cov = np.linalg.det(covariance)
            if det_cov == 0:
                covariance += epsilon * np.eye(covariance.shape[0])  # Ajout de régularisation
                det_cov = np.linalg.det(covariance)
            inv_cov = np.linalg.inv(covariance)
            diff = x - mean
            discriminant = -0.5 * np.dot(np.dot(diff, inv_cov), diff.T) - 0.5 * np.log(det_cov) + np.log(prior)
            discriminants.append(discriminant)
        return np.argmax(discriminants)


# Données d'exemple
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 1], [2, 2], [3, 3], [4, 4]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
y_test = np.array([0, 1])

# 1/ Calcul des matrices de covariance
analyse_discriminante = AnalyseDiscriminante()
analyse_discriminante.fit(X_train, y_train)

# 2/ Implémentation de la fonction de discriminante linéaire multivariée
predictions_custom = [analyse_discriminante.discriminant_function(x) for x in X_test]

# 3/ Comparaison avec scikit-learn
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions_sklearn = lda.predict(X_test)

# Mesure de précision
accuracy_custom = accuracy_score(y_test, predictions_custom)
accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

print("Précision de l\'implémentation personnalisée :", accuracy_custom) # 0.5
print("Précision de LinearDiscriminantAnalysis de scikit-learn :", accuracy_sklearn) # 0.0