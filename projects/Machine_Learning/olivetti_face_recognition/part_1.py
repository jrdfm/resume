
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


    print("TASK 1 - identifying subject label")

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy*100

    # Dataset
    data_folder = './Data/' 

    # Test Ratio
    test_ratio = 0.2

    # Random Seed
    np.random.seed(13)


    illum = loadmat(data_folder+'illumination.mat')['illum']


    # Convert the dataset in data vectors and labels for subject identification
    data = []
    labels = []
    for subject in range(illum.shape[2]):
        for image in range(illum.shape[1]):
            data.append(illum[:,image,subject])
            labels.append(subject)




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


    

    X_train, X_test, y_train, y_test = train_data,test_data,train_labels,test_labels



    # KNN

    k = 4
    clf = KNN(k)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    print("KNN classification accuracy", accuracy(test_labels, predictions))

    nb = BAYES()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))


    n_components = 35

    pca=PCA(n_components, whiten=True)
    pca.fit(train_data)
    X_train_pca=pca.transform(train_data)
    #pca.fit(test_data)
    X_test_pca=pca.transform(test_data)

    lda = LDA(11)
    lda.fit(X_train_pca, train_labels)
    X_projected = lda.transform(X_train_pca)
    #lda.fit(X_test_pca, test_labels)
    X_test_projected = lda.transform(X_test_pca)



    k = 4
    clf = KNN(k)
    clf.fit(X_train_pca, train_labels)
    predictions = clf.predict(X_test_pca)
    print("KNN with PCA classification accuracy", accuracy(test_labels, predictions))


    k = 4
    clf = KNN(k)
    clf.fit(X_projected, train_labels)
    predictions = clf.predict(X_test_projected)
    print("KNN with PCA & LDA classification accuracy", accuracy(test_labels, predictions))

    # Bayes


    nb = BAYES()
    nb.fit(X_train_pca, train_labels)
    predictions = nb.predict(X_test_pca)

    print("Naive Bayes with PCA classification accuracy", accuracy(test_labels, predictions))


    n =  BAYES()
    n.fit(X_projected, train_labels)
    predictions = n.predict(X_test_projected)

    print("Naive Bayes with PCA & LDA classification accuracy", accuracy(test_labels, predictions))





