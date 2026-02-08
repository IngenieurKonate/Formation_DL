import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

####### Fonctions for train model from 01.reseau_neurone_basic.ipynb ###############################

#Initialisation
def initialisation(x):
    w = np.random.rand(x.shape[1], 1)
    b = np.random.rand(1)
    return w, b

#STEP 1: activation function
def activation(x,w,b):
    z =np.dot(x,w)+b
    a = 1/(1+np.exp(-z)) # sigmoid
    return a

#STEP 2: loss function
def log_loss(y, a):
    m = y.shape[0]
    epsilon = 1e-15 # pour éviter les problèmes de log(0)
    cost = -1/m * np.sum(y * np.log(a + epsilon) + (1 - y) * np.log(1 - a + epsilon))
    return cost

#STEP 3,4 gradient descente & optimisation fonction 
def optimisation(x, y, a, w, b, learning_rate):
    m = y.shape[0]
    dw = 1/m * np.dot(x.T, (a-y))
    db = 1/m * np.sum(a-y)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

# Fonction de prediction
def predict(x, w, b):
    a = activation(x, w, b)
    proba = a
    return a >= 0.5, proba

#Boucle d'entrainement
def neurone_network(x, y,learning_rate, epochs): #epoches =nbre d'iteration
    w, b= initialisation(x)
    costs = []

    for i in range(epochs):
        a = activation(x, w, b)
        costs.append(log_loss(y, a))
        w, b = optimisation(x, y, a, w, b, learning_rate)

    y_pred, proba = predict(x, w, b)
    print("accuracy:", accuracy_score(y, y_pred))

    plt.plot(costs) # visualisation de la courbe d'apprentissage
    plt.show()
    return w, b, costs


# boucle d'entraînement améliorée
def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate = 0.1, epochs = 100):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(epochs)):  #tqdm permet d'afficher une barre de progression pendant l'entraînement
        A = activation(X_train, W, b)

        if i % 10 == 0:
            # Train
            train_loss.append(log_loss(y_train, A))
            y_pred_train, _ = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred_train))

            # Test
            A_test = activation(X_test, W, b)
            test_loss.append(log_loss(y_test, A_test))
            y_pred_test, _ = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred_test))

        # mise a jour
        W, b = optimisation(X_train, y_train, A, W, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.title('courbe d\'apprentissage')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.title('variations d\'accuracy')
    plt.legend()
    plt.show()

    return (W, b)