import os
os.chdir('..')
from modules import *
from functions.tools import *
os.chdir('./KerCovEmb')

# ------------------------ # ------------------------ #

def geodesic_midpoint(A,B):
    return (A+B)/2


def uni_vs_bimodal(width=1):
    class themodel:
        def __init__(self,d,width=width):
            
            self.mu0 = np.zeros(d)
            self.mu1_1 = np.ones(d)*width
            self.mu1_2 = -np.ones(d)*width

            self.cov0 = np.eye(d)
            self.cov1_1 = np.eye(d)*(.5)**.5
            self.cov1_2 = np.eye(d)*(.5)**.5

            self.name = 'uni_vs_bimodal'
            self.params = {'width': width}

        
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mu1_1, self.cov1_1, n) + np.random.multivariate_normal(self.mu1_2, self.cov1_2, n)
        def sample_X(self,n):
            return np.random.multivariate_normal(self.mu0, self.cov0, n)
        def sample_null(self,n):
            return np.random.multivariate_normal(self.mu0, self.cov0, n)
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
            self.cov_null =  geodesic_midpoint( self.covX ,  self.covY)
        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)
        def sample_null(self,n):
            return np.random.multivariate_normal(self.m_null, self.cov_null, n)
    return themodel



def isotropic_vs_scaledIsotropic(sigma):
    class themodel:
        def __init__(self, d): 
            self.name = 'isotropic_vs_scaledIsotropic'
            self.params = {'sigma': sigma}
            
            self.d = d
            self.mX = np.zeros(d); 
            self.mY = np.zeros(d)

            
            self.covX = np.eye(d)
            self.covY = (sigma**2)*np.eye(d)

            self.m_null = .5*(self.mX + self.mY)
            self.cov_null =  geodesic_midpoint( self.covX ,  self.covY)
        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)
        def sample_null(self,n):
            return np.random.multivariate_normal(self.m_null, self.cov_null, n)
    return themodel


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


        
# class isotropic_vs_offDiagSpiked:
#     def __init__(self, d): 
#         self.name = 'diagSpiked_vs_offDiagSpiked'
#         self.d = d
#         self.mX = np.zeros(d); 
#         self.mY = np.zeros(d)

#         num_spikes = d//2
#         self.covX = np.eye(d)
#         self.covY = spiked_covariance_off_diag(d, num_spikes=num_spikes, spike_value=10)

#         self.m_null = .5*(self.mX + self.mY)
#         self.cov_null =  geodesic_midpoint( self.covX ,  self.covY)
#     def sample_X(self,n):
#         return np.random.multivariate_normal(self.mX, self.covX, n)
#     def sample_Y(self,n):
#         return np.random.multivariate_normal(self.mY, self.covY, n)
#     def sample_null(self,n):
#         return np.random.multivariate_normal(self.m_null, self.cov_null, n)
# class diagSpiked_vs_offDiagSpiked:
#     def __init__(self, d): 
#         self.name = 'diagSpiked_vs_offDiagSpiked'
#         self.d = d
#         self.mX = np.zeros(d); 
#         self.mY = np.zeros(d)

#         num_spikes = d//2
#         self.covX = spiked_covariance(d, num_spikes=num_spikes, spike_value=10)
#         self.covY = spiked_covariance_off_diag(d, num_spikes=num_spikes, spike_value=10)

#         self.m_null = .5*(self.mX + self.mY)
#         self.cov_null =  geodesic_midpoint( self.covX ,  self.covY)
#     def sample_X(self,n):
#         return np.random.multivariate_normal(self.mX, self.covX, n)
#     def sample_Y(self,n):
#         return np.random.multivariate_normal(self.mY, self.covY, n)
#     def sample_null(self,n):
#         return np.random.multivariate_normal(self.m_null, self.cov_null, n)



# class MODEL_modal_vs_bimodal_gaussian:
#     def __init__(self, d, get_params): 
#         self.d = d
#         self.mX, self.covX, self.mY_0, self.mY_1, self.covY_0, self.covY_1, self.p  = get_params(d)
#         self.m_null_0 = .5*(self.mX + self.mY_0)
#         self.m_null_1 = .5*(self.mX + self.mY_1)
#         self.cov_null_0 = geodesic_midpoint(self.covX, self.covY_0)
#         self.cov_null_1 = geodesic_midpoint(self.covX, self.covY_1)
  
#     def sample_X(self,n):
#         return np.random.multivariate_normal(self.mX, self.covX, n)
    
#     def sample_Y(self,n):
#         k = np.random.binomial(n, self.p)
#         return np.vstack((np.random.multivariate_normal(self.mY_0, self.covY_0, k), 
#                           np.random.multivariate_normal(self.mY_1, self.covY_1, n-k)))
#     def sample_null(self,n):
#         # k1 = np.random.binomial(n, .5)
#         # k2 = np.random.binomial(n-k1, self.p)
#         # return np.vstack((np.random.multivariate_normal(self.m0, self.cov0, k1), 
#         #                   np.random.multivariate_normal(self.m1_0, self.cov1_0, k2),
#         #                   np.random.multivariate_normal(self.m1_1, self.cov1_1, n-k1-k2)))
#         k = np.random.binomial(n, self.p)
#         return np.concatenate((         np.random.multivariate_normal(self.m_null_0, self.cov_null_0, k), 
#                                         np.random.multivariate_normal(self.m_null_1, self.cov_null_1, n-k)), axis=0)
    

# class MODEL_twoGaussians:
#     def __init__(self, d, get_params): 
#         mX,mY,covX,covY = get_params(d) 
#         self.d = d
#         self.mX = mX
#         self.covX = covX

#         self.mY = mY
#         self.covY = covY

#         self.m_null = .5*(mX + mY)
#         self.cov_null =  geodesic_midpoint(covX, covY)


#     def sample_X(self,n):
#         return np.random.multivariate_normal(self.mX, self.covX, n)
#     def sample_Y(self,n):
#         return np.random.multivariate_normal(self.mY, self.covY, n)
#     def sample_null(self,n):
#         return np.random.multivariate_normal(self.m_null, self.cov_null, n)

