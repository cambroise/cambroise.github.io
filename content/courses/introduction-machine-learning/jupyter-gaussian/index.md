## Simulations Gaussiennes


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap

mycolormap = ListedColormap(['#FF0000', '#0000FF'])

Exchoice = 1  # Choisir l'exemple

##############################################
# Data set generation using scipy.stats.multivariate_normal
#############################################

def generate_data(n1, n2, mu1, cov1, mu2, cov2):
    """
    Generates simulated data from two multivariate normal distributions.

    Parameters:
    - n1: Size of sample 1
    - n2: Size of sample 2
    - mu1: Mean vector for group 1 (in 2 dimensions)
    - cov1: Covariance matrix for group 1 (in 2 dimensions)
    - mu2: Mean vector for group 2 (in 2 dimensions)
    - cov2: Covariance matrix for group 2 (in 2 dimensions)
    
    Returns:
    - X: Concatenated data matrix for both groups (n1 + n2, 2)
    - Y: Associated class vector (0 for group 1, 1 for group 2)
    """
    # Generate data for each group
    xG1 = multivariate_normal(mean=mu1, cov=cov1).rvs(n1)
    xG2 = multivariate_normal(mean=mu2, cov=cov2).rvs(n2)
    
    # Concatenate data from both groups
    X = np.concatenate((xG1, xG2), axis=0)
    
    # Create class vector (labels)
    Y = np.array([0] * n1 + [1] * n2)
    
    return X, Y

n1, n2 = 50, 150
mu1 = [2, 2]  # Moyenne G1
cov1 = [[1, 0], [0, 1]]  # Matrice de covariance G1 (indépendance)
    
mu2 = [0, 0]  # Moyenne G2
cov2 = [[1, 0], [0, 2]]  # Matrice de covariance G2 (indépendance mais variance différente)
    
X, Y = generate_data(n1, n2, mu1, cov1, mu2, cov2)


# Visualisation des données générées
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=mycolormap)
plt.title('Raw data')
plt.grid()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()




```


    
![png](./index_1_0.png)
    


## Simulation de l'exemple 2 (gaussiennes côte-côte) puis affichage de la posterior

 - $P(X = x \mid Y = 1)$ est une gaussienne bivariée avec une moyenne $\mu_1 = (\mu_{11}, \mu_{12})$ et une matrice de covariance $\Sigma_1$.

 - La densité a posteriori $P(Y = 1 \mid X = x)$ peut être calculée à l'aide du théorème de Bayes :

      $$
      P(Y = 1 \mid X = x) = \frac{P(X = x \mid Y = 1) P(Y = 1)}{P(X = x)}
      $$

      où :

      $$
      P(X = x) = P(X = x \mid Y = 1) P(Y = 1) + P(X = x \mid Y = 0) P(Y = 0)
      $$

      Cela permet de calculer la probabilité a posteriori en fonction des probabilités conditionnelles et des probabilités a priori.



```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Bayes_Classifier:
    def __init__(self, mu1, cov1, mu2, cov2, p1=0.5):
        """
        Initializes the Bayes Classifier with parameters of two bivariate Gaussian distributions.

        Parameters:
        - mu1: Mean vector for class 1 (in 2 dimensions)
        - cov1: Covariance matrix for class 1 (in 2 dimensions)
        - mu2: Mean vector for class 0 (in 2 dimensions)
        - cov2: Covariance matrix for class 0 (in 2 dimensions)
        - p1: Prior probability of class 1 (default is 0.5)
        """
        self.mu1 = mu1
        self.cov1 = cov1
        self.mu2 = mu2
        self.cov2 = cov2
        self.p1 = p1
        self.p2 = 1 - p1
        self.rv1 = multivariate_normal(mu1, cov1, allow_singular=True)
        self.rv2 = multivariate_normal(mu2, cov2, allow_singular=True)

    def predict_proba(self, X):
        """
        Predicts the posterior probability of class 1 for the given data points.

        Parameters:
        - X: Data points (n_samples, 2)

        Returns:
        - p1_x: Posterior probability of class 1 for each data point
        """
        # Calculate likelihoods P(X | Y=1) and P(X | Y=0)
        px_1 = self.rv1.pdf(X)
        px_2 = self.rv2.pdf(X)

        # Total density P(X=x)
        px = px_1 * self.p1 + px_2 * self.p2

        # Posterior probability P(Y=1 | X=x) using Bayes' theorem
        p1_x = (px_1 * self.p1) / px

        return p1_x


def display_posterior_density(X, Y, classifier):
    """
    Displays the posterior density P(Y=1 | X=x) using a contour plot.

    Parameters:
    - X: Data points (n_samples, 2)
    - Y: Class labels (n_samples,)
    - classifier: Bayes_Classifier object used to predict the posterior probability
    """
        # Define the grid for evaluating the posterior probability
    min_x1, min_x2 = np.min(X, axis=0) - 1
    max_x1, max_x2 = np.max(X, axis=0) + 1

    # Generate a grid of 2D points (mesh)
    x1, x2 = np.mgrid[min_x1:max_x1:.01, min_x2:max_x2:.01]
    pos = np.dstack((x1, x2))

    # Predict posterior probability of first class for each point in the grid
    p1_x = classifier.predict_proba(pos.reshape(-1, 2))
    if p1_x.ndim > 1 and p1_x.shape[1]>1:
        p1_x=p1_x[:,0]

    p1_x=p1_x.reshape(x1.shape)      

    # Display the posterior density in 2D
    plt.contourf(x1, x2, p1_x, levels=20, cmap='coolwarm')
    plt.colorbar(label='P(Y=1 | X=x)')
    plt.contour(x1, x2, p1_x, levels=[0.5], colors='black', linewidths=2)
    plt.title("Posterior Density P(Y=1 | X=x)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, cmap='coolwarm', label='Data Points')

    # Customize the plot
    plt.title("Posterior Density P(Y=1 | X=x) with Data Points")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    # Show the plot
    plt.show()

# Example usage
n1, n2 = 200, 200
mu1 = [2, 2]
cov1 = [[1, 0], [0, 1]]
mu2 = [0, 0]
cov2 = [[1, 0], [0, 2]]

X, Y = generate_data(n1, n2, mu1, cov1, mu2, cov2)
classifier = Bayes_Classifier(mu1, cov1, mu2, cov2)

# Display posterior density
display_posterior_density(X, Y, classifier)
```


    
![png](./index_3_0.png)
    


## Mon Classifieur de Bayes Naif

Classifieur de Bayes Naif


```python
import numpy as np

class BayesNaifGaussien:
    def __init__(self):
        self.mu_0 = None  # Moyenne pour la classe 0
        self.mu_1 = None  # Moyenne pour la classe 1
        self.sigma_0 = None  # Variance pour la classe 0
        self.sigma_1 = None  # Variance pour la classe 1
        self.p_y0 = None  # Probabilité a priori de Y=0
        self.p_y1 = None  # Probabilité a priori de Y=1

    def fit(self, X, Y):
        """ Calcule les paramètres du modèle à partir des données X et Y """
        # Séparer les données selon les classes
        X_0 = X[Y == 0]  # Données pour la classe 0
        X_1 = X[Y == 1]  # Données pour la classe 1
        
        # Calculer les moyennes et variances pour chaque classe
        self.mu_0 = np.mean(X_0, axis=0)
        self.mu_1 = np.mean(X_1, axis=0)
        self.sigma_0 = np.var(X_0, axis=0)
        self.sigma_1 = np.var(X_1, axis=0)
        
        # Probabilités a priori P(Y=0) et P(Y=1)
        self.p_y0 = len(X_0) / len(X)
        self.p_y1 = len(X_1) / len(X)

    def gaussienne(self, x, mu, sigma):
        """ Calcul de la densité de probabilité gaussienne pour un point x """
        return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma)

    def predict_proba(self, X):
        """ Calcule les probabilités a posteriori P(Y=1 | X) pour chaque individu dans X """
        proba_y1_given_x = []
        
        for x in X:
            # Calculer P(X | Y=1) et P(X | Y=0) pour chaque variable
            p_x_given_y1 = np.prod(self.gaussienne(x, self.mu_1, self.sigma_1))
            p_x_given_y0 = np.prod(self.gaussienne(x, self.mu_0, self.sigma_0))
            
            # Densité totale P(X=x)
            p_x = p_x_given_y1 * self.p_y1 + p_x_given_y0 * self.p_y0
            
            # Calculer la probabilité a posteriori P(Y=1 | X=x) (Théorème de Bayes)
            p_y1_x = (p_x_given_y1 * self.p_y1) / p_x
            
            # Ajouter la probabilité pour cet individu
            proba_y1_given_x.append(p_y1_x)
        
        return np.array(proba_y1_given_x)

    def predict(self, X):
        """ Prédit la classe (0 ou 1) pour chaque individu en fonction de P(Y=1 | X) """
        proba_y1 = self.predict_proba(X)
        return (proba_y1 >= 0.5).astype(int)

# Exemple d'utilisation :

# Générer des données aléatoires (2 variables)
#np.random.seed(42)
#X = np.random.randn(100, 2)  # 100 individus, 2 variables
#Y = np.random.choice([0, 1], size=100)  # Classes aléatoires 0 ou 1

# Initialiser et entraîner le modèle
model = BayesNaifGaussien()
model.fit(X, Y)

# Prédire les probabilités a posteriori P(Y=1 | X)
proba_y1_given_x = model.predict_proba(X)

# Afficher les probabilités et les paramètres du modèle
print("Probabilités P(Y=1 | X):", proba_y1_given_x)
print("Moyenne classe 0:", model.mu_0)
print("Moyenne classe 1:", model.mu_1)
print("Variance classe 0:", model.sigma_0)
print("Variance classe 1:", model.sigma_1)


# Afficher les performances
error=(model.predict(X)!=Y).sum()/len(Y)
print("Erreur empirique",error)

```

    Probabilités P(Y=1 | X): [4.77129432e-02 1.88198612e-01 1.19806441e-02 1.61156057e-03
     7.57724865e-01 5.47979333e-03 2.86561102e-02 5.28354680e-03
     4.19710179e-03 2.32670361e-03 4.25351100e-05 2.75755452e-02
     1.66745301e-03 2.49435759e-01 1.00766811e-03 6.33178333e-02
     7.11726400e-02 8.06725044e-03 3.49788468e-02 2.33123217e-01
     1.85679153e-01 1.61388217e-02 4.73890955e-04 3.40903507e-03
     5.48351905e-02 6.11117153e-01 7.97042638e-04 2.39487337e-02
     1.81606673e-01 7.51506787e-01 1.44906911e-01 3.20614819e-03
     1.49182986e-02 2.36029577e-02 1.10421848e-03 4.06797993e-02
     2.76919459e-01 9.80876216e-03 1.51080835e-03 1.19535527e-02
     7.54920408e-02 2.37668468e-03 1.34267871e-01 1.50892025e-04
     2.82689122e-04 3.97560524e-01 1.85964298e-01 2.89790882e-02
     2.61032354e-01 4.09004267e-01 2.28426189e-04 2.69845210e-01
     6.10553550e-03 8.50242142e-01 2.03655676e-01 2.20047876e-02
     2.75311642e-02 9.77504492e-02 3.40259695e-01 3.73255885e-01
     1.52570034e-03 2.83882103e-01 1.47553228e-03 1.59946985e-02
     3.12402875e-03 9.51981172e-03 1.26780125e-01 1.38245003e-03
     1.45565399e-01 2.11117520e-01 4.74854726e-04 6.81187545e-02
     3.17382874e-01 1.45887411e-01 3.72396824e-03 1.00399080e-04
     4.13313282e-03 1.17479805e-03 6.78963271e-01 1.77858266e-02
     3.18035514e-01 1.04417578e-01 3.76900434e-02 3.05594629e-01
     8.03794773e-05 6.33186245e-01 4.86832863e-03 2.00066529e-02
     4.04799919e-03 2.19082430e-01 3.34528481e-01 6.23729351e-03
     1.66303929e-02 8.95526616e-03 3.98065567e-02 1.98639470e-01
     4.04561745e-02 5.82954816e-02 3.39805011e-03 2.09663974e-02
     2.53093383e-02 2.21555326e-02 1.09365256e-02 3.74422495e-02
     3.43872900e-04 1.89273542e-02 1.38456942e-03 4.80891278e-02
     1.07933445e-01 3.05213147e-01 6.69535889e-02 1.44887853e-03
     2.20980438e-03 3.40678910e-02 1.56474474e-02 4.55360912e-04
     1.52599230e-04 3.38423666e-02 5.30316981e-02 3.95061749e-03
     1.66579442e-03 2.99502034e-01 4.89077220e-01 7.24637743e-02
     3.72126273e-03 9.97495673e-01 5.81488703e-02 2.71824520e-03
     3.31314879e-03 1.87599392e-01 2.84798275e-02 2.23632018e-01
     3.60488048e-01 2.38428084e-01 6.82397142e-02 1.70207819e-04
     1.65429076e-02 8.97641020e-02 1.47437064e-02 1.02971798e-02
     8.66017352e-04 2.90444112e-01 2.73443451e-03 3.12627059e-02
     4.67172669e-02 1.67625382e-02 3.70097236e-02 1.45051358e-03
     3.23307861e-01 3.18419139e-03 2.22863238e-02 1.22510786e-03
     9.82459480e-02 1.11912747e-02 1.20060399e-01 1.62403349e-03
     2.90996052e-01 5.85842940e-02 7.63932976e-03 2.20318786e-02
     1.71423565e-02 1.87681205e-02 3.41083199e-02 2.05334553e-03
     4.36902743e-01 3.31158281e-03 1.42633539e-04 9.32139342e-04
     1.73175620e-02 2.04639977e-03 2.40672704e-05 2.58175562e-03
     2.00897353e-01 7.09373320e-02 1.93387721e-02 3.43676984e-02
     6.39837963e-03 2.08496946e-02 5.02641687e-02 4.40223776e-02
     9.23455857e-01 8.50272970e-01 1.20956546e-01 1.36344422e-01
     1.65764476e-01 6.13372600e-02 5.06173937e-03 3.54062021e-02
     1.70972598e-03 3.92652867e-01 1.50124067e-01 2.12859692e-02
     1.16263615e-02 4.21556783e-02 7.13342266e-02 2.05523669e-02
     6.42687173e-02 1.95607635e-01 2.36928616e-01 9.28496158e-02
     9.92363237e-01 9.99347763e-01 9.91116855e-01 9.87181820e-01
     9.57392190e-01 8.68372052e-01 9.97952198e-01 9.70984948e-01
     9.28307824e-01 9.85646977e-01 9.84346244e-01 9.96345508e-01
     9.65157702e-01 9.97200756e-01 9.59341537e-01 9.90321766e-01
     9.97160385e-01 9.99963510e-01 9.99954006e-01 8.65862808e-01
     9.72073336e-01 9.99887907e-01 7.29077611e-01 9.95690771e-01
     9.55895128e-01 9.94207279e-01 9.86176885e-01 9.97203715e-01
     9.72794523e-01 9.95845219e-01 9.31445242e-01 9.33333563e-01
     9.99562104e-01 9.99866670e-01 9.99837689e-01 9.75130335e-01
     9.97995441e-01 9.99417007e-01 9.96072060e-01 9.32357076e-01
     9.55016044e-01 6.53713472e-01 6.78432054e-01 9.84099988e-01
     1.38612115e-01 9.86935547e-01 9.30047978e-01 9.98871069e-01
     9.73709329e-01 9.88313439e-01 9.96964639e-01 9.99237123e-01
     9.92546248e-01 9.88113710e-01 9.99551783e-01 1.73016802e-01
     9.99651727e-01 9.99983129e-01 9.99789445e-01 9.78577405e-01
     9.55891996e-01 1.46452492e-01 9.98577642e-01 9.21704460e-01
     9.99953932e-01 9.99837455e-01 9.47543861e-01 9.03746805e-01
     9.99103334e-01 9.99792672e-01 9.99956825e-01 9.99997796e-01
     8.57309795e-01 6.71868292e-01 5.29623647e-01 4.96240922e-01
     9.99619180e-01 8.82698551e-01 7.67662542e-01 9.95413843e-01
     9.99991943e-01 9.17265098e-01 9.99470345e-01 8.39243490e-01
     9.99937582e-01 9.99482108e-01 9.97125578e-01 6.24979920e-01
     9.34161919e-01 9.95496765e-01 9.94315135e-01 8.83557684e-01
     8.97947696e-01 5.10179180e-01 9.94743249e-01 9.61605183e-01
     9.99644538e-01 9.88320148e-01 8.76415088e-01 9.59577794e-01
     9.50858263e-01 9.78410036e-01 9.99913326e-01 9.70535075e-01
     9.99969452e-01 9.75022751e-01 8.64891438e-01 9.82382548e-01
     9.96648735e-01 9.99061819e-01 3.99177158e-01 9.98788237e-01
     9.73770146e-01 9.99608663e-01 8.59156079e-01 9.99982757e-01
     9.99654456e-01 8.38934579e-01 8.74529610e-01 9.99903513e-01
     9.99044151e-01 7.21647474e-01 9.96229908e-01 4.32779327e-02
     9.99652362e-01 9.96643859e-01 5.28982658e-01 9.99655356e-01
     7.02083645e-01 9.99986917e-01 5.33882446e-01 3.51556956e-01
     2.58009589e-01 9.93309344e-01 9.99923436e-01 9.99929086e-01
     9.94525560e-01 9.89293529e-01 9.95676851e-01 9.98902457e-01
     6.56435543e-01 9.99881910e-01 9.20536849e-01 9.96802987e-01
     9.99722059e-01 9.99930401e-01 9.99997011e-01 8.70526167e-01
     9.63482657e-01 9.81299644e-01 9.99988951e-01 9.99666706e-01
     9.86843060e-01 9.86592838e-01 9.99742465e-01 1.54492741e-01
     3.62478287e-01 9.99781860e-01 1.29421022e-02 5.74462107e-01
     2.94746510e-01 3.23258171e-01 9.91575605e-01 9.57284443e-01
     9.99659660e-01 9.92712669e-01 9.71220091e-01 9.99939050e-01
     9.90235282e-01 6.12219544e-01 3.16340570e-01 5.57344204e-01
     9.99994937e-01 9.94087895e-01 9.02534982e-01 2.28010496e-01
     9.96876362e-01 9.99703341e-01 7.93473528e-01 3.83625876e-01
     9.99988105e-01 9.76297207e-01 9.84924536e-01 9.99999680e-01
     9.99962153e-01 8.96720802e-01 9.98621979e-01 9.98555326e-01
     9.95912839e-01 9.51539421e-01 3.72359423e-01 9.84742990e-01
     9.94566778e-01 7.34086111e-01 1.14850155e-01 9.87775139e-01
     9.99788052e-01 9.45727676e-01 8.42356808e-01 9.95216255e-01]
    Moyenne classe 0: [2.04616058 1.91278979]
    Moyenne classe 1: [-0.07510028 -0.14203197]
    Variance classe 0: [0.97936836 0.97850395]
    Variance classe 1: [0.89293017 2.05696981]
    Erreur empirique 0.0675


## Erreur en apprentissage et test pour le classifieur de Bayes naïf


```python


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=3)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB();
gnbmod=gnb.fit(X_train, y_train);
y_pred_test = gnbmod.predict(X_test)
y_pred_train = gnbmod.predict(X_train)

E_test=(y_test != y_pred_test).sum()/len(y_test)
E_train=(y_train != y_pred_train).sum()/len(y_train)

print("Error on the test set=",E_test)
print("Error on the train set=",E_train)

```

    Error on the test set= 0.05970149253731343
    Error on the train set= 0.07142857142857142


## Courbe ROC et AUC pour une validation en 5 plis


```python
#Etude MM 2022
y=Y
#############################################
# Cross-validation sur la courbe ROC
############################################
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

n_samples, n_features = X.shape

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = GaussianNB();

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    y_train=[y[i] for i in train];
    y_test=[y[i] for i in test];
    classifier.fit(X[train], y_train)
    viz = RocCurveDisplay.from_estimator(
        classifier,             # Le modèle entraîné
        X[test],                # Les données de test
        y_test,                 # Les étiquettes de test
        name='ROC fold {}'.format(i),  # Nom de la courbe pour chaque fold
        alpha=0.3,              # Transparence
        lw=1,                   # Épaisseur de la ligne
        ax=ax                   # L'axe de la figure pour tracer la courbe
        )       
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()
    

```


    
![png](./index_10_0.png)
    


## Analyse discriminante linéaire


```python
# Example usage
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Generate data
n1, n2 = 200, 200
mu1 = [2, 2]
cov1 = [[1, 0], [0, 1]]
mu2 = [0, 0]
cov2 = [[1, 0], [0, 2]]

X, Y = generate_data(n1, n2, mu1, cov1, mu2, cov2)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)

# Predict on test set
Y_pred = lda.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"LDA Classification Accuracy: {accuracy:.2f}")

# Display posterior density
display_posterior_density(X, Y, lda)
```

    LDA Classification Accuracy: 0.93



    
![png](./index_12_1.png)
    


Nous remarquons la frontière linéaire liée à l'égalité des matrices de covariance


## Analyse discriminante quadratique
Les matrices de variances-covariances sont laissées libres et la frontière est une hyperbole (proche de la frontière obtenu par le classifieur de Bayes naïf)


```python
# Example usage with QDA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Generate data
n1, n2 = 200, 200
mu1 = [2, 2]
cov1 = [[1, 0], [0, 1]]
mu2 = [0, 0]
cov2 = [[1, 0], [0, 2]]

X, Y = generate_data(n1, n2, mu1, cov1, mu2, cov2)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train QDA model
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, Y_train)

# Predict on test set
Y_pred = qda.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"QDA Classification Accuracy: {accuracy:.2f}")

# Display posterior density
display_posterior_density(X, Y, qda)

```

    QDA Classification Accuracy: 0.93



    
![png](./index_14_1.png)
    


## Les plus proches voisins


```python
# Example usage with KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generate data
n1, n2 = 200, 200
mu1 = [2, 2]
cov1 = [[1, 0], [0, 1]]
mu2 = [0, 0]
cov2 = [[1, 0], [0, 2]]

X, Y = generate_data(n1, n2, mu1, cov1, mu2, cov2)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k)
knn.fit(X_train, Y_train)

# Predict on test set
Y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"KNN Classification Accuracy: {accuracy:.2f}")

# Display posterior density
display_posterior_density(X, Y, knn)

```

    KNN Classification Accuracy: 0.85



    
![png](./index_16_1.png)
    


## La régression logistique


```python
# Example usage with Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate data
n1, n2 = 200, 200
mu1 = [2, 2]
cov1 = [[1, 0], [0, 1]]
mu2 = [0, 0]
cov2 = [[1, 0], [0, 2]]

X, Y = generate_data(n1, n2, mu1, cov1, mu2, cov2)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

# Predict on test set
Y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Logistic Regression Classification Accuracy: {accuracy:.2f}")

# Display posterior density
display_posterior_density(X, Y, log_reg)

```

    Logistic Regression Classification Accuracy: 0.88



    
![png](./index_18_1.png)
    


## Comparaison des méthodes par AUC et erreur de classification

Remarquons les grandes différences entre ensemble d'apprentissage et test. 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.datasets import make_classification

# Générer des données simulées
n1, n2 = 200, 200
X, Y = make_classification(n_samples=n1 + n2, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split les données en train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Fonction pour calculer les AUC avec validation croisée (10-fold)
def calculate_auc_cv(model, X_train, Y_train, X_test, Y_test):
    scorer = make_scorer(roc_auc_score)
    train_scores = cross_val_score(model, X_train, Y_train, cv=10, scoring=scorer)
    test_scores = cross_val_score(model, X_test, Y_test, cv=10, scoring=scorer)
    return train_scores, test_scores

# Stocker les résultats des AUC
results_train = []
results_test = []
model_names = []

# Calculer l'AUC pour chaque modèle
for name, model in models.items():
    train_auc, test_auc = calculate_auc_cv(model, X_train, Y_train, X_test, Y_test)
    results_train.append(train_auc)
    results_test.append(test_auc)
    model_names.append(name)

# Créer les boxplots pour visualiser les AUC en train et en test
fig, ax = plt.subplots(figsize=(12, 6))

# Préparer les données pour afficher les boxplots côte à côte
positions_train = np.arange(1, len(models) * 2, 2)  # Positions pour les boxplots en train
positions_test = np.arange(2, len(models) * 2 + 1, 2)  # Positions pour les boxplots en test

# Boxplot pour l'AUC sur les données de train (en bleu)
bp_train = ax.boxplot(results_train, positions=positions_train, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"))

# Boxplot pour l'AUC sur les données de test (en orange)
bp_test = ax.boxplot(results_test, positions=positions_test, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor="orange"), medianprops=dict(color="red"))

# Ajouter une légende pour les boxplots
ax.legend([bp_train["boxes"][0], bp_test["boxes"][0]], ['Train', 'Test'], loc='upper right')

# Ajuster les labels et le titre
ax.set_xticks(np.arange(1.5, len(models) * 2, 2))
ax.set_xticklabels(model_names)
ax.set_title("Comparaison des modèles - AUC en Train et Test (10-fold CV)")
ax.set_ylabel("AUC Score")

plt.tight_layout()
plt.show()

```


    
![png](./index_20_0.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from sklearn.datasets import make_classification

# Générer des données simulées
n1, n2 = 200, 200
X, Y = make_classification(n_samples=n1 + n2, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split les données en train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Fonction pour calculer les erreurs avec validation croisée (10-fold)
def calculate_error_cv(model, X_train, Y_train, X_test, Y_test):
    scorer = make_scorer(lambda y_true, y_pred: 1 - np.mean(y_true == y_pred))  # Calculer l'erreur
    train_errors = cross_val_score(model, X_train, Y_train, cv=10, scoring=scorer)
    test_errors = cross_val_score(model, X_test, Y_test, cv=10, scoring=scorer)
    return train_errors, test_errors

# Stocker les résultats des erreurs
results_train = []
results_test = []
model_names = []

# Calculer l'erreur pour chaque modèle
for name, model in models.items():
    train_error, test_error = calculate_error_cv(model, X_train, Y_train, X_test, Y_test)
    results_train.append(train_error)
    results_test.append(test_error)
    model_names.append(name)

# Créer les boxplots pour visualiser les erreurs en train et en test
fig, ax = plt.subplots(figsize=(12, 6))

# Préparer les données pour afficher les boxplots côte à côte
positions_train = np.arange(1, len(models) * 2, 2)  # Positions pour les boxplots en train
positions_test = np.arange(2, len(models) * 2 + 1, 2)  # Positions pour les boxplots en test

# Boxplot pour l'erreur sur les données de train (en bleu)
bp_train = ax.boxplot(results_train, positions=positions_train, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"))

# Boxplot pour l'erreur sur les données de test (en orange)
bp_test = ax.boxplot(results_test, positions=positions_test, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor="orange"), medianprops=dict(color="red"))

# Ajouter une légende pour les boxplots
ax.legend([bp_train["boxes"][0], bp_test["boxes"][0]], ['Train', 'Test'], loc='upper right')

# Ajuster les labels et le titre
ax.set_xticks(np.arange(1.5, len(models) * 2, 2))
ax.set_xticklabels(model_names)
ax.set_title("Comparaison des modèles - Erreur en Train et Test (10-fold CV)")
ax.set_ylabel("Erreur de Classification")

plt.tight_layout()
plt.show()

```


    
![png](./index_21_0.png)
    



```python

```
