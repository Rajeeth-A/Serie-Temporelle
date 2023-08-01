# -*- coding: utf-8 -*-
"""
Created on 2023-04-20 20:00:00

@author: Rajeeth-A
"""
### Importation des bibliothéques nécessaires ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
#################################################

### Exercice 1: Débruitage d'un signal ###
#1.1:
"""
Nous cherchons à minimiser la fonction f donnée par l'énoncé en trouvant le vecteur x 
qui satisfait la condition suivante : ∇f(x) = 0 car f est convexe par composition 
d'une application linéaire polynomiale et d'une application ||.|| convexe.
Par de simple calcul, on trouve ∇f(x) = 0 = (x-y) + λ*||D||² 
En résolvant cette équation pour x, nous obtenons : x = (I + λ*||D||²)^(-1)*y
En notant (x,y) deux vecteur de taille N*N, λ un scalaire positif et la notation ||x - y||² 
désigne la norme euclidienne au carré entre les vecteurs x et y et I la matrice identité.
"""
def D_mat(N):
    """_sommaire_
    Crée une matrice D de taille NxN.
    
    Args:
        N (int): La taille de la matrice D.
    
    Returns:
        _numpy.ndarray_: La matrice D de taille NxN.
    """
    D = np.zeros((N,N))
    for i in range(N-1):
        D[i,i] = -1
        D[i,i+1] = 1
    return D

#1.2:
"""
La fonction f est bien différentiable car f est la composition 
d'une fonction polynomiale et d'une application norme. 
Ainsi, la fonction f est continue et que toutes ses dérivées partielles existent et sont continues. 
Définissons dans la suite la fonction f comme étant la fonction objectif du problème d'optimisation 
et la fonction grad_f comme étant le gradient de la fonction objectif.    
"""
def f(x):
    """_sommaire_
    Calcule la fonction objectif pour le problème d'optimisation.
    
    Args:
        x (numpy.ndarray): Le vecteur des valeurs de la fonction.
    
    Returns:
        _float_: La valeur de la fonction objectif.
    """
    global y, lmbda
    D = D_mat(len(x))
    premier_terme = (1/2) * (x - y).transpose() @ (x - y)
    deuxieme_terme = (lmbda/2) * (x.transpose() @ D.transpose() @ D @ x)
    return premier_terme + deuxieme_terme

def grad_f(x):
    """_sommaire_
    Calcule le gradient de la fonction objectif pour le problème d'optimisation.
    
    Args:
        x (numpy.ndarray): Le vecteur des valeurs de la fonction.
    
    Returns:
        _numpy.ndarray_: Le gradient de la fonction objectif.
    """
    global y, lmbda
    D = D_mat(len(x))
    grad_term1 = x - y
    grad_term2 = lmbda * np.dot(D.transpose(), np.dot(D, x))
    return grad_term1 + grad_term2

#1.3:
def generate_signal(N,sigma=0.05):
    """_sommaire_
    Génère un signal bruité composé de trois niveaux.
    
    Args:
        N (int): La taille du signal.
        sigma (float, optional): Le niveau de bruit ajouté au signal. Par défaut : 0.05.
    
    Returns:
        _tuple_: Les vecteurs de temps (t) et de signal bruité (y).
    """
    t = np.linspace(0, 1, N)
    t1 = 0.1 + 0.25 * np.random.random()
    t2 = 0.35 + 0.25 * np.random.random()
    yi = np.array([-0.1, 0.8, 0.2])
    y = np.zeros(N)
    for i in range(y.size):
        if t[i] <= t1:
            y[i] = yi[0]
        elif t[i] > t1 and t[i] <= t2:
            y[i] = yi[1]
        else:
            y[i] = yi[2]
    y += sigma * (2 * np.random.random(y.size) - 1)
    return t,y

#En appliquant le code donnée dans l'énoncé pour N = 100, on génère le signal suivants :
N = 100
t, y = generate_signal(N)
plt.plot(t, y)
plt.xlabel('Temps (t)')
plt.ylabel('y')
plt.title('Signal bruité N = 100')
plt.show()

#1.4:
def gradient_met(grad_f, x0, alpha, eps, Nmax):
    """_sommaire_
    Applique la méthode du gradient pour minimiser la fonction objectif.
    
    Args:
        grad_f (callable): La fonction qui calcule le gradient de la fonction objectif.
        x0 (numpy.ndarray): Le point de départ pour l'algorithme.
        alpha (float): Le pas constant.
        eps (float): Le critère de convergence où précision.
        Nmax (int): Le nombre maximum d'itérations.
    
    Returns:
        x le minimum de la fonction objectif,
        n le nombre d'itérations effectuées,
        cvg un boléen qui indique si l'algorithme a convergé ou non.
    """
    x = x0
    n = 0
    gradfx = grad_f(x)
    while np.linalg.norm(gradfx) > eps and n < Nmax:
        x = x - alpha * gradfx
        gradfx = grad_f(x)
        n += 1
    cvg = np.linalg.norm(gradfx) <= eps
    return (x, n, cvg)
#1.5:
# Application de la méthode de gradient à notre problème d'optimisation:
N = 100
t, y = generate_signal(N, sigma=0.05)
D = D_mat(N)
x0 = y
alpha = 0.05
eps = 1e-6
Nmax = 3000
lmbda = 1
gradf_x = lambda x: grad_f(x)
min, n_iterations, cvg = gradient_met(gradf_x, x0, alpha, eps, Nmax)
print(f"Le minimum trouve est {min} en {n_iterations} iterations avec la methode du gradient.")
print(f"Pour ce minimum, la convergence est {cvg}.")

# Représentation graphique du résultat:
plt.figure(figsize=(12, 6))
plt.plot(t, y, label='Signal bruité', alpha=0.7, color='red')
plt.plot(t, min, label='Signal débruité', alpha=0.7, color='blue', linestyle='--')
plt.legend(title=f"Methode de Gradient: {n_iterations} iterations effectuées")
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.title('Signal bruité et signal débruité')
plt.show()
#1.6:
"""
Le graphique représenté ci-dessus montre que la méthode du gradient est efficace pour débruiter le signal 
et que le signal débruité est très proche du signal original.
Le résultat était attendu car la méthode du gradient est une méthode d'optimisation qui permet d'améliorer le modèle
"""
#1.7:
N = 100
t, noisy_signal = generate_signal(N, sigma=0.05)
D = D_mat(N)
x0 = noisy_signal
alpha = 0.05
eps = 1e-6
Nmax=3000
# Representation graphique du résultat:
plt.figure(figsize=(12, 6))
for lmbda in range(1, 5):
    y = noisy_signal
    min, n_iterations, _ = gradient_met(grad_f, x0, alpha, eps, Nmax)
    plt.plot(t, min, label=f"λ = {lmbda}")
plt.xlabel("Temps")
plt.ylabel("Signal")
plt.title("Signal débruitée pour différentes valeurs de λ")
plt.legend()
plt.show()
"""
Le signal reste encore un peu bruité pour λ = 1, mais pour λ = 2, 3 et 4, le signal est bien débruité.
"""
#1.8:
def grad_f_mu(x):
    """_sommaire_
    Calcule le gradient de la fonction cout avec les hyperparamètres mu et lam.
    
    Args:
        x (numpy.ndarray): Le vecteur des valeurs de la fonction.
    
    Returns:
        _numpy.ndarray_: Le gradient de la fonction objectif avec régularisation Tikhonov.
    """
    global y, lam, mu
    N = len(x)
    grad = np.zeros(N)
    for i in range(N):
        if i == 0:
            grad[i] = x[i] - y[i] + lam * (x[i] - x[i+1]) / np.sqrt((x[i+1] - x[i])**2 + mu**2)
        elif i == N-1:
            grad[i] = x[i] - y[i] + lam * (x[i] - x[i-1]) / np.sqrt((x[i] - x[i-1])**2 + mu**2)
        else:
            grad[i] = x[i] - y[i] + lam * (x[i] - x[i+1]) / np.sqrt((x[i+1] - x[i])**2 + mu**2) - (
                lam * (x[i] - x[i-1]) / np.sqrt((x[i] - x[i-1])**2 + mu**2) )
    return grad
#1.9:
# Appliquons la fonction de la méthode du gradient à notre problème d'optimisation:
N = 100
t, y = generate_signal(N, sigma=0.05)
lam = 0.01
mu = 0.01
x0 = y
alpha = 0.01
eps = 1e-6
Nmax = 3000
min, n_iterations, cvg = gradient_met(grad_f_mu, x0, alpha, eps, Nmax)
print(f"Le minimum trouve est {min} en {n_iterations} iterations avec la methode du gradient.")
print(f"L'algorithme a convergé: {cvg}")
# Representation graphique du résultat:
plt.plot(t, y, label='Signal bruité', color='red')
plt.plot(t, min, label='Signal débruité', color='blue', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title("Signal bruité et débruité par la méthode de gradient")
plt.show()
"""
On voit clairement la différence entre le signal original et le signal reconstruit.
"""
#1.10:
def BBstep(grad_f, x, xm1):
    """_sommaire_
    Calcule le pas (noté alpha_k) par la méthode de Barzilai-Borwein.
    
    Args:
        grad_f (callable): La fonction qui calcule le gradient de la fonction objectif.
        x (numpy.ndarray): La valeure du vecteur à l'étape précédente (x_k)
        xm1 (numpy.ndarray): La valeure du vecteur à l'étape précédente (x_(k-1)).

    Returns:
        _float_: Le pas dans la méthode Barzilai-Borwein.
    """
    grad_x = grad_f(x)
    grad_xm1 = grad_f(xm1)
    delta_xk = x - xm1
    delta_gk = grad_x - grad_xm1
    bbs = (np.dot(delta_xk.transpose(), delta_gk)) / (
        np.dot(delta_gk.transpose(), delta_gk) )
    return bbs

def barzilai_borwein(grad_f, x0, eps, Nmax):
    """_sommaire_
    Applique la méthode du gradient avec l'étape de Barzilai-Borwein
    pour minimiser la fonction objectif.
    
    Args:
        grad_f (callable): La fonction qui calcule le gradient de la fonction objectif.
        x0 (numpy.ndarray): Le point de départ pour l'algorithme.
        eps (float): Le critère de convergence où la précision.
        Nmax (int): Le nombre maximum d'itérations.

    Returns:
        _tuple_: Le minimum, le nombre d'itérations et si l'algorithme a convergé.
    """
    x = x0
    xm1 = x0
    n = 0
    gradfx = grad_f(x)
    alpha = 1.0
    
    while np.linalg.norm(gradfx) > eps and n < Nmax:
        x = x - alpha * gradfx
        gradfx = grad_f(x)
        alpha = BBstep(grad_f, x, xm1)
        n += 1
        
    cvg = np.linalg.norm(gradfx) <= eps
        
    return (x, n, cvg)

N = 100
t, y = generate_signal(N, sigma=0.05)
x0 = y
eps = 1e-6
Nmax = 3000

min_bb, n_iterations_bb, cvg_bb = barzilai_borwein(grad_f, x0, eps, Nmax)
print(f"Le minimum trouve est {min_bb} en {n_iterations_bb} iterations avec la methode de Barzilai-Borwein.")
print(f"L'algorithme a convergé: {cvg_bb}")

# Représentation graphique du résultat des différents méthodes gradient et BB:
plt.figure(figsize=(12, 6))
plt.plot(t, y, label='Signal bruité', alpha=0.7, color='black')
plt.plot(t, min, label='Signal débruité (Gradient)', alpha=0.7, color='red', linestyle='-.')
plt.plot(t, min_bb, label='Signal débruité (BB)', alpha=0.7, color='blue', linestyle='--')
plt.legend(title=f"Comparaison entre les méthodes")
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.title('Comparaison entre la méthode du gradient et la méthode de Barzilai-Borwein')
plt.show()
"""
On voit que la méthode de Barzilai-Borwein converge plus rapidement que la méthode du gradient.
Le graphique pour la méthode de Barzilai-Borwein est plus lisse que celui de la méthode du gradient, 
ce qui est normal puisque la méthode de Barzilai-Borwein utilise une méthode de pas adaptatif.
La méthode la plus adaptée dépendra du problème à résoudre, dans notre cas, la méthode de Barzilai-Borwein
est plus adaptée.
"""

### Nouvelle image ###
image = imread('Grey_Mona_lisa.jpg') 
imageArray = np.asarray(image, dtype=np.float64)[:, :, 0].copy() / 255.
plt.imshow(imageArray, cmap='gray')
plt.title('Image originale')
g = imageArray.copy()

### Exercice 2: Débruitage d'une image ###
#2.1:

def derivee_x(image):
    """
    Calcule la dérivée discrète en x d'une image donnée.
    
    Args:
        image (np.array): Image d'entrée.
        
    Returns:
        np.array: Image des dérivées discrètes en x.
    """
    m, n = image.shape
    dx = np.zeros_like(image)
    for i in range(m - 1):
        for j in range(n):
            dx[i, j] = (image[i + 1, j] - image[i, j])
    return dx


def derivee_y(image):
    """
    Calcule la dérivée discrète en y d'une image donnée.
    
    Args:
        image (np.array): Image d'entrée.
        
    Returns:
        np.array: Image des dérivées discrètes en y.
    """
    m, n = image.shape
    dy = np.zeros_like(image)
    for i in range(m):
        for j in range(n - 1):
            dy[i, j] = (image[i, j + 1] - image[i, j])
    return dy

# Représentation graphique des dérivées discrètes en x et en y.
dx = derivee_x(g)
dy = derivee_y(g)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(dx, cmap='gray')
plt.title('Dérivée discrète en x')

plt.subplot(1, 2, 2)
plt.imshow(dy, cmap='gray')
plt.title('Dérivée discrète en y')

plt.show()

#2.2:
def norme_derivee_discrete(image):
    return np.sqrt(derivee_x(image) ** 2 + derivee_y(image) ** 2)

# Représentation graphique de la norme dérivées discrètes en x et en y.
norme_dd = norme_derivee_discrete(g)
plt.imshow(norme_dd, cmap='gray')
plt.title('Norme dérivée discrète en x et en y')
plt.show()
"""
On constate que les dérivées discrètes en x et en y sont très bruitées.
On peut donc utiliser la norme des dérivées discrètes afin de combiner 
les informations de bruitage des derivées discrete en direction de x et y.
On peut également remarquer que la norme des dérivées discrètes
dessine mieux les contours et les bords que les derivées discretes.
"""
#2.3:
def grad_j(v, g, lam):
    """
    Calcule le gradient de la fonction objectif pour le débruitage d'image pour la méthode du gradient.
    
    Args:
        v (np.array): Image débruitée candidate.
        g (np.array): Image bruitée.
        lam (float): Paramètre de régularisation.
        
    Returns:
        np.array: Gradient de la fonction objectif (2D).
    """
    m, n = g.shape
    M = D_mat(m)
    N = D_mat(n)
    grad_terme1 = v - g
    grad_terme2 = lam * (np.dot(np.dot(M.T, M), v) + np.dot(np.dot(v, N), N.T))
    return grad_terme1 + grad_terme2


def gradient_met_j(grad_j, v0, alpha, eps, Nmax, g, lam):
    """
    Implémentation de la méthode du gradient pour minimiser la fonction objectif.
    
    Args:
        grad_j (function): Fonction pour calculer le gradient de la fonction objectif.
        v0 (np.array): Point de départ pour l'optimisation.
        alpha (float): Pas.
        eps (float): Tolérance pour la convergence.
        Nmax (int): Nombre maximum d'itérations.
        g (np.array): Image bruitée.
        lam (float): Paramètre de régularisation.
        
    Returns:
        _tuple_: (v, n, cvg) où v est le vecteur optimal, n le nombre d'itérations et cvg un booléen indiquant si la méthode a convergé.
    """
    v = v0.copy()
    n = 0
    gradjv = grad_j(v, g, lam)
    while np.linalg.norm(gradjv) > eps and n < Nmax:
        v -= alpha * gradjv
        gradjv = grad_j(v, g, lam)
        n += 1
    cvg = np.linalg.norm(gradjv) <= eps
    return v, n, cvg

# Representation graphique de l'image débruitée:
alpha = 0.1
eps = 1e-4
Nmax = 3000
v0 = g.copy()
mu = 0.1

lambdas = [0.1, 0.5, 1, 2]
n = len(lambdas)

plt.figure(figsize=(12, 6))

for i, lam in enumerate(lambdas):
    v_opt, n_ite, cvg = gradient_met_j(grad_j, v0, alpha, eps, Nmax, g, lam)

    plt.subplot(2, 2, i + 1)
    plt.imshow(v_opt, cmap='gray')
    plt.title(f'Image débruitée avec la méthode de gradient\nobtenue en {n_ite} itérations pour lam = {lam}')

plt.tight_layout()
plt.show()

"""
On voit clairement la différence entre les images débruitées obtenues avec les différentes valeurs de lambda.
Plus lambda est grand, plus le bruitage est important. Il est donc important de trouver la bonne valeur de lambda
pour obtenir une image débruitée correcte. On peut par exemple choisir lambda = 0.01 pour obtenir une image débruitée
"""

#2.4:
def J_mu(v):
    """
    Calcule la fonction de cout avec un paramètres de regularisation mu.
    
    Args:
        v (np.array): Image débruitée candidate.
        g (np.array): Image bruitée.
        lam (float): Paramètre de régularisation.
        mu (float): Paramètre de régularisation pour J_mu.
        
    Returns:
        float: Valeur de la fonction de cout.
    """
    global g, lam, mu, dx, dy
    terme1 = np.linalg.norm(v - g)**2
    terme2 = lam * np.sum(np.sqrt(dx**2 + dy**2 + mu**2))
    return terme1 + terme2

def grad_j_mu(v, g, lam):
    """
    Calcule le gradient de la fonction de cout avec un paramètre de régularisation mu.
    
    Args:
        v (np.array): Image débruitée candidate.
        g (np.array): Image bruitée.
        lam (float): Paramètre de régularisation.
        mu (float): Paramètre de régularisation pour grad_j_mu.
        
    Returns:
        np.array: Gradient de la fonction de cout (2D).
    """
    global dx, dy, mu
    norm_term = np.sqrt(dx**2 + dy**2 + mu**2)
         
    grad_terme1 = 2 * (v - g)
    grad_terme2 = lam * (dx / norm_term + dy / norm_term)

    return grad_terme1 + grad_terme2
   
   
#Représentation image du gradient: 
mu = 0.01
lam = 0.01
v_opt_mu, n_ite, cvg = gradient_met_j(grad_j_mu, v0, alpha, eps, Nmax, g, lam)
plt.imshow(v_opt_mu, cmap='gray')
plt.title(f"Représentation en image du gradien obtenu avec lam = {lam} en {n_ite} itérations")
plt.show()

"""
En utilisant la regularisation mu, on obtient une image débruitée plus nette que celle obtenue avec la regularisation lambda.
"""


#2.5:
image = imread('Le_Cri.jpg') #Le_Cri de Edvard Munch
imageArray = np.asarray(image, dtype=np.float64)[:, :, 0].copy() / 255.
h = imageArray.copy()
#a:
sigma = 0.1
image_bruit_blanc = h + sigma * np.random.randn(*h.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')

plt.subplot(1, 2, 2)
plt.imshow(image_bruit_blanc, cmap='gray')
plt.title(f"Image bruitée (gaussien avec bruit blanc, σ = {sigma})")

plt.show()

#b:
alpha = 0.1
eps = 1e-6
Nmax = 3000
lam = 0.01
v0 = image_bruit_blanc.copy()
mu = 0.1
v_opt_bb, n_ite, cvg = gradient_met_j(grad_j, v0, alpha, eps, Nmax, image_bruit_blanc, lam)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_bruit_blanc, cmap='gray')
plt.title('Image bruitée avec bruit blanc (gaussien)')

plt.subplot(1, 2, 2)
plt.imshow(v_opt_bb, cmap='gray')
plt.title(f'Image débruitée avec lam = {lam} et {n_ite} itérations après la méthode de gradient')

plt.tight_layout()
plt.show()

#c:
dx = derivee_x(h)
dy = derivee_y(h)
norme_ddh = norme_derivee_discrete(h)
v_opt, n_ite, cvg = gradient_met_j(grad_j, v0, alpha, eps, Nmax, h, lam)
v_opt_cout, n_ite, cvg = gradient_met_j(grad_j_mu, v0, alpha, eps, Nmax, h, lam)

plt.figure(figsize=(30, 10))

plt.subplot(2, 3, 1)
plt.imshow(dx, cmap='gray')
plt.title('Dérivée discrète en x')

plt.subplot(2, 3, 2)
plt.imshow(dy, cmap='gray')
plt.title('Dérivée discrète en y')

plt.subplot(2, 3, 3)
plt.imshow(norme_ddh, cmap='gray')
plt.title('Norme de la dérivée discrète')

plt.subplot(2, 3, 4)
plt.imshow(v_opt, cmap='gray')
plt.title(f'Image débruitée avec lam = {lam} sans bruit blanc')

plt.subplot(2, 3, 5)
plt.imshow(v_opt_bb, cmap='gray')
plt.title(f'Image débruitée avec lam = {lam} avec bruit blanc')

plt.subplot(2, 3, 6)
plt.imshow(v_opt_cout, cmap='gray')
plt.title(f"Représentation en image avec la fonction de coût J_mu")

plt.tight_layout()
plt.show()
#d:
"""
On arrive sur la même conclusion que dans les premières étapes.
En effet, on constate que les derivées discrètes sont très bruitée
et la norme des dérivées discrètes permet de combiner les informations
ce qui permet de faire mieux ressortir les contours et les bords.
Mais la dernière étape en utilisant la fonction de coût J_mu, on voit que les bords et
les contours sont plus visibles et l'image beaucoup plus nette.
"""
