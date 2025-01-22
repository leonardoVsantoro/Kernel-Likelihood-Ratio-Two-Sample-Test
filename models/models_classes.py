import os
os.chdir('..')
from modules import *
from functions.tools import *
os.chdir('./CKE')

# ------------------------ # ------------------------ #


def isotropic_different_means(mu, numDiffLocs = None):
    class themodel:
        def __init__(self, d, numDiffLocs = numDiffLocs):

            if numDiffLocs is None or numDiffLocs > d:
                numDiffLocs = d 
            
            self.name = 'isotropic_different_means'
            self.params = {'mu': mu, 'numDiffLocs': numDiffLocs}

            self.d = d

            self.mX = np.ones(d); 
            self.mY = np.ones(d); 
            self.mY[:int(numDiffLocs)]= mu*np.ones(int(numDiffLocs))

            
            self.covX = np.eye(d)
            self.covY =  np.eye(d)

            self.m_null = .5*(self.mX + self.mY)
            self.cov_null = np.eye(d)
        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)
        def sample_null(self,n):
            return np.random.multivariate_normal(self.m_null, self.cov_null, n)
    return themodel


def isotropic_vs_DiagSpiked(spike_value, num_spikes):
    class themodel:
        def __init__(self, d, spike_value=spike_value, num_spikes=num_spikes): 
            self.name = 'isotropic_vs_DiagSpiked'
            self.params = {'spike_value': spike_value, 'num_spikes': num_spikes}

            self.d = d
            self.mX = np.zeros(d); 
            self.mY = np.zeros(d)
            if num_spikes is None:
                num_spikes = d//2
            if num_spikes > d:
                num_spikes = d
            num_spikes = int(num_spikes)
            self.covX = np.eye(d)
            self.covY = spiked_covariance(d, num_spikes=num_spikes, spike_value=spike_value)

            self.m_null = .5*(self.mX + self.mY)
            self.cov_null =  .5* (self.covX +   self.covY )
        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)
        def sample_null(self,n):
            return np.random.multivariate_normal(self.m_null, self.cov_null, n)
    return themodel



def isotropic_vs_DiagSpiked_different_means(mu, spike_value, numDiffLocs, num_spikes):
    class themodel:
        def __init__(self, d, numDiffLocs = numDiffLocs):

            if numDiffLocs is None:
                numDiffLocs = d //2
            if numDiffLocs > d:
                numDiffLocs = d 

            if num_spikes is None:
                num_spikes = d //2
            if num_spikes > d:
                num_spikes = d 
            
            self.name = 'isotropic_vs_DiagSpiked_different_means'
            self.params = {'mu': mu, 'numDiffLocs': numDiffLocs, 'spike_value': spike_value, 'num_spikes': num_spikes}

            self.d = d
            self.mX = np.ones(d); 
            self.mY = np.ones(d); 
            self.mY[:int(numDiffLocs)]= mu*np.ones(int(numDiffLocs))

            self.covX = np.eye(d)
            self.covY = spiked_covariance(d, num_spikes=num_spikes, spike_value=spike_value)

            self.m_null = .5*(self.mX + self.mY)
            self.cov_null = np.eye(d)
        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)
        def sample_null(self,n):
            return np.random.multivariate_normal(self.m_null, self.cov_null, n)
    return themodel
    