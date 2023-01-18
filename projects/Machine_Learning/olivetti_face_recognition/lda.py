import numpy as np



class LDA:
    def __init__ (self,n=None):
        self.n = n
        self.l_disc = None
        self.eps  = 1e-9
        
    def fit(self,X,lables):
        n_ft = X.shape[1]
        
        c_lab = np.unique(lables)    # unique class lables

        self.classes_ = c_lab 

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n == None:
            self.n = max_components

        m = np.mean(X,axis=0)        # total mean vector
        SW = np.zeros((n_ft, n_ft))    # within class scatter matrix
        SB = np.zeros((n_ft, n_ft))    # between class scatter matrix

        for c in c_lab:
            x_c = X[lables == c]
            n_i = x_c.shape[0]        # number of samples
            m_i = np.mean(x_c, axis=0)
            SW += (x_c - m_i).T.dot((x_c - m_i)) # (n_ft x n_i).(n_i x n_ft) = n_ft x n_ft
            
            m_df = (m_i - m).reshape(n_ft,1)

            SB += n_i * m_df.dot(m_df.T)


        if not(self.is_invertible(SW)):
            I = np.identity(SW.shape[0])
            SW = np.add(SW,self.eps*I)
        

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))

        eigenvectors = eigenvectors.T
        i = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[i], eigenvectors[i]

        self.l_disc = eigenvectors[0 : self.n]

    def is_invertible(self,a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    def transform (self,X):
        return np.dot(X, self.l_disc.T)
        