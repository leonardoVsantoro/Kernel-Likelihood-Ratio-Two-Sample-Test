import os
os.chdir('..')
from modules import *
from functions.tools import *
os.chdir('./CKE')

# ------------------------ # ------------------------ #


def MODEL_1(mu, numDiffLocs):
    """
    Creates a model class with specified parameters for drawing samples from two isotropic Gaussian distributions with identical covariance but differing means.
    Both models have isotropic covaraince; the first is centred, whereas the second has the first `numDiffLocs` coordinates of its mean vector set to `mu`.

    Parameters:
    mu (float): The mean value to be assigned to the first `numDiffLocs` elements of mY.
    numDiffLocs (int): The number of locations in mY to be set to the value of mu.

    Returns:
    themodel (class): A class representing the model with methods to sample from different distributions.

    Class Attributes:
    name (str): The name of the model.
    params (dict): A dictionary containing the parameters 'mu' and 'numDiffLocs'.
    d (int): Dimensionality of the data.
    mX (ndarray): Mean vector for distribution X, initialized to zeros.
    mY (ndarray): Mean vector for distribution Y, initialized to zeros with the first `numDiffLocs` elements set to mu.
    covX (ndarray): Covariance matrix for distribution X, initialized to identity matrix.
    covY (ndarray): Covariance matrix for distribution Y, initialized to identity matrix.
    m_null (ndarray): Mean vector for the null distribution, initialized to the average of mX and mY.
    cov_null (ndarray): Covariance matrix for the null distribution, initialized to identity matrix.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    sample_null(n): Samples n points from the null distribution.
    """
    class themodel:
        def __init__(self, d, numDiffLocs = numDiffLocs):

            self.name = 'MODEL_1'
            self.params = {'mu': mu, 'numDiffLocs': numDiffLocs}
            self.d = d

            self.mX = np.zeros(d); 
            self.mY = np.zeros(d); self.mY[:numDiffLocs] = mu
            
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


def MODEL_2(spike_value, num_spikes):
    """
    Creates a model class with specified parameters for drawing samples from two isotropic Gaussian distributions with identical means but differing covariance matrices.
    Both models are centred; the first has an identity covariance matrix, whereas the second has a spiked covariance matrix with `num_spikes` spikes of value `spike_value`.
    
    Parameters:
    spike_value (float): The value of the spike to be added to the covariance matrix.
    num_spikes (int): The number of spikes to be added to the covariance matrix.

    Returns:
    themodel (class): A class representing the model with methods to sample from different distributions.

    Class Attributes:
    name (str): The name of the model.
    params (dict): A dictionary containing the parameters 'spike_value' and 'num_spikes'.
    d (int): Dimensionality of the data.
    mX (ndarray): Mean vector for distribution X, initialized to zeros.
    mY (ndarray): Mean vector for distribution Y, initialized to zeros.
    covX (ndarray): Covariance matrix for distribution X, initialized to identity matrix.
    covY (ndarray): Covariance matrix for distribution Y, initialized with spikes.
    m_null (ndarray): Mean vector for the null distribution, initialized to the average of mX and mY.
    cov_null (ndarray): Covariance matrix for the null distribution, initialized to the average of covX and covY.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    sample_null(n): Samples n points from the null distribution.
    """
    class themodel:
        def __init__(self, d, spike_value=spike_value, num_spikes=num_spikes): 

            self.name = 'MODEL_2'
            self.params = {'spike_value': spike_value, 'num_spikes': num_spikes}
            self.d = d

            self.mX = np.zeros(d); 
            self.mY = np.zeros(d)

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



def MODEL_3(mu, spike_value, numDiffLocs, num_spikes):
    """
    Creates a model class with specified parameters. The model draws samples from two isotropic Gaussian distributions with differing means and covariance matrices.
    The first has an identity covariance matrix and the first `numDiffLocs` coordinates of its mean vector set to `mu`.
    The second has mean zero and spiked covariance, with `num_spikes` spikes of value `spike_value`.

    Parameters:
    mu (float): Mean value for the first `numDiffLocs` dimensions of mY.
    spike_value (float): The value of the spikes in the covariance matrix of Y.
    numDiffLocs (int): Number of dimensions where mY differs from mX.
    num_spikes (int): Number of spikes in the covariance matrix of Y.

    Returns:
    themodel (class): A class with methods to sample from distributions X, Y, and null.
    
    Class Attributes:
    name (str): The name of the model.
    params (dict): A dictionary containing the parameters 'mu', 'spike_value', 'numDiffLocs', and 'num_spikes'.
    d (int): Dimensionality of the data.
    mX (ndarray): Mean vector for distribution X, initialized to zeros.
    mY (ndarray): Mean vector for distribution Y, initialized to zeros with the first `numDiffLocs` elements set to mu.
    covX (ndarray): Covariance matrix for distribution X, initialized to identity matrix.
    covY (ndarray): Covariance matrix for distribution Y, initialized with spikes.
    m_null (ndarray): Mean vector for the null distribution, initialized to the average of mX and mY.
    cov_null (ndarray): Covariance matrix for the null distribution, initialized to the average of covX and covY.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    sample_null(n): Samples n points from the null distribution.
    """

    class themodel:
        def __init__(self, d,mu=mu, spike_value= spike_value,  numDiffLocs = numDiffLocs, num_spikes = num_spikes):

            self.name = 'MODEL_3'
            self.params = {'mu': mu, 'numDiffLocs': numDiffLocs, 'spike_value': spike_value, 'num_spikes': num_spikes}
            self.d = d
            
            self.mX = np.zeros(d); 
            self.mY = np.zeros(d); self.mY[:numDiffLocs] = mu

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
    