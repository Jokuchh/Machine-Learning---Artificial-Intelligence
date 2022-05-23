import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm


# charger les données de défauts de rails
data = np.loadtxt("defautsrails.dat")
X = data[:,:-1]  # tout sauf dernière colonne
y = data[:,-1]   # uniquement dernière colonne


G = np.zeros((140,4))


for k in range(0,4):
    Yk = 2*(y==k+1)-1
    model = svm.LinearSVC(C=1)
    model.fit(X, Yk)
    Ypred = model.predict(X)
    Error = np.mean(Ypred != Yk)
    print("Erreur modèle", k+1, " = ", Error)
    G[:,k] = model.decision_function(X)



# Prédictions et calcul de l'erreur généralisée
Ypredict = np.argmax(G,axis=1)+1
ErrApp = np.mean(Ypredict != y)
print("Erreur d'apprentissage : ", ErrApp)



# Validation croisée

erreurs=0
for i in range(0,140):
    x_i = np.delete(X, i, axis=0)
    y_i = np.delete(y, i)
    G = np.zeros(4)
    
    for k in range(0,4):
        Yk = 2*(y_i==k+1)-1
        model = svm.LinearSVC(C=1)
        model.fit(x_i, Yk)
        G[k] = model.decision_function([X[i,:]])
        
    Ypred = np.argmax(G)+1
    erreurs += (Ypred != y[i])

ErreurLOO = erreurs/len(y)
print("Nombre erreurs = ", erreurs)
print("Erreur LOO = ", ErreurLOO)











