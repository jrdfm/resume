import numpy as np
from scipy.stats import norm



class BAYES:
    def fit(self,X,lables):
        n_i , n_ft = X.shape # ni = number of samples ,n_ft number of features
        self.classes = np.unique(lables)
        self.eps = None
        self.var_smoothing  = 1e-9
        n_cl = len(self.classes)


        self.mean = np.zeros((n_cl,n_ft),dtype = np.float64)
        self.var = np.zeros((n_cl,n_ft),dtype = np.float64)
        self.priors = np.zeros(n_cl,dtype = np.float64)


        for i,c in enumerate(self.classes):
            x_c = X[c == lables]
            self.mean[i,:] = x_c.mean(axis=0)
            self.var[i,:] = x_c.var(axis=0)
            self.priors[i] = x_c.shape[0]/ float(n_i)




    def predict(self,X):
        return np.array([self.prd(x) for x in X])

    def prd(self,x):
        
        posteriors = []
        for i,c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self.pdf(i,x)  + self.var_smoothing))
            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]


    def pdf(self,c_id,x):
        mean = self.mean[c_id]
        var = self.var[c_id]
        std = np.sqrt(var)
        return norm.pdf(x,mean,np.sqrt(var))
        #return (1/(std * np.sqrt(2*np.pi)))*(np.exp((-1/2)*((x-mean)/std)**2))
        

