
import numpy as np

import random
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

if __name__ == "__main__":
    
    from lda import *
    from pca import *
    from bayes import *
    from knn import *


    print("TASK 2 - neutral vs. facial expression classification")

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy*100

    # Dataset
    data_folder = './Data/' 

    # Test Ratio
    test_ratio = 0.2

    # Random Seed
    np.random.seed(13)



    Ns = 200 
    face = loadmat(data_folder+'data.mat')['face']
    face_n = [face[:,:,3*n] for n in range(Ns)] # neutral
    face_x = [face[:,:,3*n+1] for n in range(Ns)] # expression
    face_il = [face[:,:,3*n+2] for n in range(Ns)] # illumination variation
    data = []
    labels = []
    for subject in range(Ns):
        
        data.append(face_n[subject].reshape(-1))
        labels.append(0)
        
        data.append(face_x[subject].reshape(-1))
        labels.append(1)

    # Split to train and test data        
    N = int( (1-test_ratio)*len(data) )
    idx = np.arange(len(data))
    random.shuffle(idx)

    tmp = np.asarray(data)
    l =np.asarray(labels)

    train_data = tmp[idx[:N]]
    train_labels = l[idx[:N]]
    test_data = tmp[idx[N:]]
    test_labels = l[idx[N:]]


    # Bayes

    X_train, X_test, y_train, y_test = train_data,test_data,train_labels,test_labels




    nb = BAYES()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))


    n_components = 11
    pca=PCA(n_components, whiten=False)
    pca.fit(train_data)
    X_train_pca=pca.transform(train_data)
    #pca.fit(test_data)
    X_test_pca=pca.transform(test_data)

    nb =BAYES()
    nb.fit(X_train_pca, train_labels)
    predictions = nb.predict(X_test_pca)

    print("Naive Bayes with PCA classification accuracy", accuracy(test_labels, predictions))


    lda = LDA(2)
    lda.fit(X_train_pca, train_labels)
    X_projected = lda.transform(X_train_pca)
    #lda.fit(X_test_pca, test_labels)
    X_test_projected = lda.transform(X_test_pca)


    nb = BAYES()
    nb.fit(X_projected, train_labels)
    predictions = nb.predict(X_test_projected)

    print("Naive Bayes with LDA classification accuracy", accuracy(test_labels, predictions))

    # KNN

    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))


   
    clf = KNN(k=k)
    clf.fit(X_train_pca, train_labels)
    predictions = clf.predict(X_test_pca)
    print("KNN with PCA classification accuracy", accuracy(test_labels, predictions))



    clf = KNN(k=k)
    clf.fit(X_projected, train_labels)
    predictions = clf.predict(X_test_projected)
    print("KNN with LDA classification accuracy", accuracy(test_labels, predictions))

