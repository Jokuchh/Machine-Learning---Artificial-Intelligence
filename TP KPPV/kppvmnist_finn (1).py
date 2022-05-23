import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import neighbors


#### Fonctions de chargement et affichage de la base mnist ####

def load_mnist(m,mtest):

	X = np.load("mnistX.npy")
	y = np.load("mnisty.npy")

	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]
	X = X.reshape((X.shape[0], -1))

	return train_test_split(X, y, train_size=m, test_size=mtest)


def showimage(x):
	plt.imshow( 255 - np.reshape(x, (28, 28) ), cmap="gray")
	plt.show()
	

#############################
#### Programme principal ####

# chargement de la base mnist:
Xtrain, Xtest, ytrain, ytest = load_mnist(11000, 1000)
Xapp = Xtrain[:10000,:]
Yapp = ytrain[:10000]
Xval = Xtrain[10000:,:]
Yval = ytrain[10000:]

print("Taille de la base d'apprentissage : ", Xapp.shape)
print("Taille de la base de test : ", Xtest.shape)
print("Taille de la base de validation : ", Xval.shape)

Kmax = 0
Error_min = 0.1
for K in range(1,11):
    kppv = neighbors.KNeighborsClassifier(K)
    kppv.fit(Xapp, Yapp)
    Ypred = kppv.predict(Xval)
    """ Evaluation de l'erreur de prédiction --> risque du classifieur """
    Error = np.mean(Ypred != Yval)
    print("Erreur de validation (K=",K,") = ", Error)
    if Error < Error_min :
        Error_min = Error
        Kmax = K

print("Meilleur K : ", Kmax)


Xtrain, Xtest, ytrain, ytest = load_mnist(60000, 1000)

kppv = neighbors.KNeighborsClassifier(Kmax)
kppv.fit(Xtrain, ytrain)
Ypred = kppv.predict(Xtest)

""" Evaluation de l'erreur de prédiction --> risque du classifieur """
Error = np.mean(Ypred != ytest)
print("Erreur de test = ", Error)





