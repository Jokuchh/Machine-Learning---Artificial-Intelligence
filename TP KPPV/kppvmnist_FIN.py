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
Xtrain, Xtest, ytrain, ytest = load_mnist(60000, 1000)
# =============================================================================
# Xapp = Xtrain[: 10000, :]
# Yapp = ytrain[:10000]
# Xvalid= Xtrain[10000:,:]
# Yvalid= ytrain[10000:]
# minV = 1
# index = 1
# for i in range(1,10):
#     kppv = neighbors.KNeighborsClassifier(i) # création du modèle
#     kppv.fit(Xapp, Yapp)
#     ypred = kppv.predict(Xvalid)
#     print(i ,':', np.mean(ypred != Yvalid))
#     if minV > np.mean(ypred != Yvalid):
#         minV = np.mean(ypred != Yvalid)
#         index = i
# 
# print("Taille de la base d'apprentissage : ", Xtrain.shape)
# print(minV)
# =============================================================================

kppv = neighbors.KNeighborsClassifier(3) # création du modèle
kppv.fit(Xtrain, ytrain)
ypred = kppv.predict(Xtest)
print(np.mean(ypred != ytest))

# à compléter... 







