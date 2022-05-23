import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans, SpectralClustering


#### programme principal à implémenter dans cette fonction ####
def monprogramme(X, K): 
    print("Kmeans clustering lancé avec " + str(len(X)) + " points et K = ", K)
    
    #model = KMeans(n_clusters=K, init='random', n_init=1 )
    #2
    K = 2
    model = KMeans(n_clusters=K, n_init=1) # change till 100 
    model.fit(X)
    #3
    Y = model.labels_
    infinity = model.cluster_centers_
    print(Y)
     
    colours = ['b','r','g','y','m','c']

    #4 --useless
    #Xbleu = X[Y==1,:]
    #plt.plot(Xbleu[:,0], Xbleu[:,1], '.b')
    for k in range(K):
        Xk = X[Y==k] 
        plt.plot(Xk[:,0], Xk[:,1], '.' + colours[k])
	
        plt.plot(infinity[k,0], infinity[k,1],'x' + colours[k])

        fig.canvas.draw()
	


### programme pour Spectral clustering ###
def monprogrammeSpectralClustering(X, K, sigma):
    print("Spectral Clustering lancé avec " + str(len(X)) + " points et sigma = ", sigma)
    
	# à compléter ...
    model = SpectralClustering(n_clusters=K, gamma=1/2*(sigma**2))
    #affinity='nearest_neighbors'
    model.fit(X)
    #3
    Ypred = model.labels_
    
    A = model.affinity_matrix_
    A = 1 - A
	# ... et tracer le résultat 
    plt.figure(1)
    colours = ['b','r','g','y','m','c','k']
	# à compléter pour le tracé... 
    for k in range(K):
        Xk = X[Ypred==k] 
        plt.plot(Xk[:,0], Xk[:,1], '.' + colours[k])
	
	
	# création de la seconde figure : 
    fig2=plt.figure(2) 
    plt.clf()	# pour effacer le résultat précédent
	# ajouter le tracé ici...
    plt.show()	# afficher la figure
    plt.imshow(A, cmap='gray')

	# pour mettre à jour le premier graphique: 
    fig.canvas.draw()
	

##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
plotvariance = 0

K = 2
sigma = 1

def onclick(event):
	global Xplot
	
	if plotvariance == 0:
		newX = np.array([[event.xdata,event.ydata]])
	else:
		newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

	print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

	Xplot = np.concatenate((Xplot,newX))
	if event.button == 1 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.k')
	
	fig.canvas.draw()


def onscroll(event):
	global plotvariance
	if event.button == "up":
		plotvariance = round(plotvariance + 0.2, 1)
	elif event.button == "down" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
	print("Variance = ", plotvariance)

def onkeypress(event):
	global K
	global sigma
	global Xplot
	if event.key == " ":
		monprogramme(Xplot, K)
	elif event.key == "c":
		monprogrammeSpectralClustering(Xplot,K,sigma)
	elif event.key == "+" and K < len(Xplot):
		K += 1
		print("K = " , K)
	elif event.key == "-" and K > 1:
		K -= 1
		print("K = " , K)
	elif event.key == "ctrl++":
		sigma *= 2
		print("sigma = " , sigma)
	elif event.key == "ctrl+-":
		sigma /= 2
		print("sigma = " , sigma)
	elif event.key == 'delete':
		Xplot = np.zeros((0,2))	
		plt.clf()
		plt.axis([-5, 5, -5, 5])
		fig.canvas.draw()
	
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points ")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(X,K)")
print("    avec la valeur de K modifiée par +/-\n\n") 
print("\n C pour lancer la fonction monprogrammeSpectralClustering(X,K,sigma)")
print("    avec la valeur de sigma modifiée par Ctrl +/-\n\n") 

plt.show()
