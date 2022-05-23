import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.kernel_ridge import KernelRidge
from sklearn import neighbors


def kernel(X1,X2,sigma):
	"""
		Retourne la matrice de noyau K telle que K_ij = K(X1[i], X2[j])
		avec un noyau gaussien K(x,x') = exp(-||x-x'||^2 / 2sigma^2)
	"""	
	m1 = X1.shape[0]
	m2 = X2.shape[0]
	K = np.zeros((m1,m2))
	for i in range(m1):
		for j in range(m2):
			K[i,j] = math.exp(- np.linalg.norm(X1[i] - X2[j])**2 / (2*sigma**2))
	return K

def krrapp(Xapp,Yapp,Lambda,sigma):
    K = kernel(Xapp,Xapp,sigma)
    A = (K + np.eye(len(K)))*Lambda
    beta = np.linalg.solve(A,Y)
    return beta
	
def krrpred(Xtest,Xapp,beta,sigma):
    Ktest = kernel(Xtest,Xapp,sigma)
    ypred = Ktest * beta
    return ypred
	

def kppvreg(Xtest, Xapp, Yapp, K):
	n = Xtest.shape[0]  # nb de points de test
	m = Xapp.shape[0]   # nb de points d'apprentissage
	ypred = np.zeros(n)

	
	return ypred

#################################################
#### Programme principal ########################
#################################################

## 1) générer une base de données de 1000 points X,Y

m = 1000
X = 6 * np.random.rand(m) - 3
Y = np.sinc(X) + 0.2 * np.random.randn(m)


# 2) Créer un base d'apprentissage (Xapp, Yapp) de 30 points parmi ceux de (X,Y) et une base de test(Xtest,Ytest) avec le reste des données

indexes = np.random.permutation(m)  # permutation aléatoire des 1000 indices entre 0 et 1000 
indexes_app = indexes[:30]  # 30 premiers indices
indexes_test = indexes[30:] # le reste

Xapp = X[indexes_app]
Yapp = Y[indexes_app]

Xtest = X[indexes_test]
Ytest = Y[indexes_test]

# ordronner les Xtest pour faciliter le tracé des courbes
idx = np.argsort(Xtest)
Xtest = Xtest[idx]
Ytest = Ytest[idx]

# tracer la figure

plt.figure()
plt.plot(Xtest,Ytest,'.r')
plt.plot(Xapp,Yapp,'*b')
plt.plot(Xtest,np.sinc(Xtest) , 'g')
plt.legend(['base test', 'base app', 'f_reg(x)'] )


### Tests de la Kernel ridge regression... 
beta = krrapp(Xapp, Yapp, 0.1, 1)
ypred = krrpred(Xtest, Xapp, beta, 1)
plot(Xtest, ypred, 'y')


### Tests avec les Kppv...
kppv = neighbors.KNeighborsRegressor(K)
kppv.fit(Xapp.reshape(-1,1), Yapp)
ypred = kppv.predict(Xtest.rehape(-1,1))



# Affichage des graphiques : 
# (à ne faire qu'en fin de programme)
plt.show() # affiche les plots et bloque en attendant la fermeture de la fenêtre

