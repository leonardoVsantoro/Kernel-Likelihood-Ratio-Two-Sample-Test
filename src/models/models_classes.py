from modules import *
from functions.utils import *

def spiked_covariance(d, num_spikes=1, spike_value=10, template_cov=None):
    if template_cov is None:
        cov_matrix = np.eye(d)
    else:
        cov_matrix = template_cov(d)
    cov_matrix[-num_spikes:, -num_spikes:] = spike_value
    return cov_matrix

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_0():
    """
    Creates a model class with methods to sample from two isotropic Gaussian distributions with identical parameters.
    The model has no parameters and samples from a standard multivariate normal distribution.

    Returns:
    themodel (class): A class representing the model with methods to sample from distributions X, Y, and null.

    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters of the model.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'MODEL_0'
        def __init__(self, d):
            self.d = d
        def sample_X(self,n):
            return np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), n)
        def sample_null(self,n):
            return np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), n)
    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_1(mu, P):
    """
    Creates a model class with specified parameters for drawing samples from two isotropic Gaussian distributions with identical covariance but differing means.
    Both models have isotropic covaraince; the first is centred, whereas the second has the first `P` coordinates of its mean vector set to `mu`.

    Parameters:
    mu (float): The mean value to be assigned to the first `P` elements of mY.
    P (int): The number of locations in mY to be set to the value of mu.

    Returns:
    themodel (class): A class representing the model with methods to sample from different distributions.

    Class Attributes:
    name (str): The name of the model.
    params (dict): A dictionary containing the parameters 'mu' and 'P'.
    d (int): Dimensionality of the data.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samplahes n points from the distribution Y.
    sample_null(n): Samples n points from the null distribution.
    """
    class themodel:
        __name__ = 'MODEL_1'
        def __init__(self, d, P = P):
            P = min(P, d)  # Ensure P does not exceed d
            self.params = {'mu': mu, 'P': P}
            self.d = d

            self.mX = np.zeros(d); 
            self.mY = np.zeros(d); self.mY[:P] = mu
            
            self.covX = np.eye(d)  # Isotropic covariance matrix
            self.covY =  np.eye(d)  # Isotropic covariance matrix

        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)

    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

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

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    sample_null(n): Samples n points from the null distribution.
    """
    class themodel:
        __name__ = 'MODEL_2'
        def __init__(self, d, spike_value=spike_value, num_spikes=num_spikes): 

            self.name = 'MODEL_2'
            self.params = {'spike_value': spike_value, 'num_spikes': num_spikes}
            self.d = d

            self.mX = np.zeros(d); 
            self.mY = np.zeros(d)

            num_spikes = int(num_spikes)
            self.covX = np.eye(d)
            self.covY = spiked_covariance(d, num_spikes=num_spikes, spike_value=spike_value)


        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)

    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_3(mu, spike_value, P, num_spikes):
    """
    Creates a model class with specified parameters. The model draws samples from two isotropic Gaussian distributions with differing means and covariance matrices.
    The first has an identity covariance matrix and the first `P` coordinates of its mean vector set to `mu`.
    The second has mean zero and spiked covariance, with `num_spikes` spikes of value `spike_value`.
    
    Parameters:
    mu (float): Mean value for the first `P` dimensions of mY.
    spike_value (float): The value of the spikes in the covariance matrix of Y.
    P (int): Number of dimensions where mY differs from mX.
    num_spikes (int): Number of spikes in the covariance matrix of Y.

    Returns:
    themodel (class): A class with methods to sample from distributions X, Y, and null.

    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'mu', 'P', 'spike_value', and 'num_spikes'.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    sample_null(n): Samples n points from the null distribution.
    """

    class themodel:
        __name__ = 'MODEL_3'
        def __init__(self, d, mu=mu, spike_value= spike_value,  P = P, num_spikes = num_spikes):

            self.name = 'MODEL_3'
            self.params = {'mu': mu, 'P': P, 'spike_value': spike_value, 'num_spikes': num_spikes}
            self.d = d
            
            self.mX = np.zeros(d); 
            self.mY = np.zeros(d); self.mY[:P] = mu

            self.covX = np.eye(d)
            self.covY = spiked_covariance(d, num_spikes=num_spikes, spike_value=spike_value)

        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)

    return themodel
    
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_4(epsilon, P):
    """
    Creates a model class for drawing samples from uniform distributions:
    - X is sampled from [0,1]^d (uniform hypercube).
    - Y is sampled from [epsilon, 1-epsilon]^P x [0,1]^{d-P}.

    Parameters:
    epsilon (float): The amount to truncate from the upper bound in the first P dimensions of Y.
    P (int): The number of coordinates in Y affected by epsilon.

    
    Returns:
    themodel (class): A class representing the model with methods to sample from X and Y.

    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'epsilon' and 'P'.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'MODEL_4'
        def __init__(self, d, P = P):
            assert 0 < epsilon < 1, "epsilon must be in (0, 1)"
            # assert 0 <= P <= d, "P must be between 0 and d"
            P = min(P,d)  # Ensure P does not exceed d
            self.name = 'MODEL_4'
            self.params = {'epsilon': epsilon, 'P': P}
            self.d = d
            self.P = P

        def sample_X(self, n):
            return np.random.uniform(0, 1, size=(n, self.d))

        def sample_Y(self, n):
            Y = np.empty((n, self.d))
            Y[:, :self.P] = np.random.uniform(epsilon, 1 - epsilon, size=(n, self.P))
            Y[:, self.P:] = np.random.uniform(0, 1, size=(n, self.d - self.P))
            return Y

    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_5(mu, P):
    """
    Mixture model for two-sample testing:
    - X ~ N(0, I_d)
    - Y ~ 0.5 N(-mu, I_d) + 0.5 N(mu, I_d), with mean shift in first P dimensions

    Parameters:
    mu (float): Mean shift magnitude in the mixture components.
    P (int): Number of coordinates affected by the shift.

    Returns:
    themodel (class): A model class with sampling methods.

    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'mu' and 'P'.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'MODEL_5'
        def __init__(self, d, P=P):
            self.params = {'mu': mu, 'P': P}
            self.d = d

            self.shift = np.zeros(d)
            self.shift[:P] = mu

        def sample_X(self, n):
            return np.random.normal(loc=0.0, scale=1.0, size=(n, self.d))

        def sample_Y(self, n):
            n_half = n // 2
            extra = n % 2
            Y1 = np.random.normal(loc=-self.shift, scale=1.0, size=(n_half, self.d))
            Y2 = np.random.normal(loc=+self.shift, scale=1.0, size=(n_half + extra, self.d))
            return np.vstack([Y1, Y2])

    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def MODEL_6(eps):
    """
    Model for two-sample testing:
    - X ~ N(0,Id)
    - Y ~ N(0,Id) with probability 1-\eps, 0 with probability \eps 

    Parameters:
    eps (float): Probability of sampling 0 in Y.

    Returns:
    themodel (class): A model class with sampling methods.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'eps' and 'P'.
    

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'MODEL_6'
        def __init__(self, d, eps=eps):
            self.params = {'eps': eps}
            self.d = d

        def sample_X(self, n):
            return np.random.normal(loc=0.0, scale=1.0, size=(n, self.d))

        def sample_Y(self, n):
            n_zeros = math.ceil(n * eps)
            n_noise = n - n_zeros
            zeros = np.zeros((n_zeros, self.d))
            noise = np.random.normal(loc=0.0, scale=1.0, size=(n_noise, self.d))
            samples = np.vstack((zeros, noise))
            np.random.shuffle(samples)
            return samples
            # rolls = np.random.binomial(1, 1 - self.params['eps'], n)
            # out = []
            # for r in rolls:
            #     if r == 1:
            #         out.append(np.random.normal(loc=0.0, scale=1.0, size=self.d))
            #     else:
            #         out.append(np.zeros(self.d))
            # return np.array(out)
    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_7(alpha, eps):
    """
    Decreasing correlation model for two-sample testing:
    - X ~ N(0,\Sigma_1), where \Sigma_1(i,j) = alpha^|i-j|
    - Y ~ N(0,\Sigma_2), where \Sigma_2(i,j) = (alpha + eps)^|i-j|

    Parameters:
    alpha (float): Base correlation coefficient for the covariance matrix of X.
    eps (float): Additional correlation coefficient for the covariance matrix of Y.

    Returns:
    themodel (class): A model class with sampling methods.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'eps' and 'P'.
    

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'MODEL_7'
        def __init__(self, d, alpha =alpha, eps=eps):
            self.params = {'alpha' : alpha, 'eps': eps}
            self.d = d

        def sample_X(self, n):
            cov_X = np.array([[alpha ** abs(i - j) for j in range(self.d)] for i in range(self.d)])
            return np.random.multivariate_normal(mean=np.zeros(self.d), cov=cov_X, size=n)
        def sample_Y(self, n):
            cov_Y = np.array([[ (alpha + eps) ** abs(i - j) for j in range(self.d)] for i in range(self.d)])
            return np.random.multivariate_normal(mean=np.zeros(self.d), cov=cov_Y, size=n)
    return themodel

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 

def MODEL_8(alpha, eps):
    """
    Equi-correlated model for two-sample testing:
    - X ~ N(0,\Sigma_1), where \Sigma_1 = (1-alpha)Id + alpha* 1@1^T
    - Y ~ N(0,\Sigma_2), where \Sigma_2 = (1-(alpha+eps))Id + (alpha+eps)* 1@1^T

    Parameters:
    alpha (float): Base correlation coefficient for the covariance matrix of X.
    eps (float): Additional correlation coefficient for the covariance matrix of Y.

    Returns:
    themodel (class): A model class with sampling methods.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'eps' and 'P'.
    

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'MODEL_8'
        def __init__(self, d, alpha =alpha, eps=eps):
            self.params = {'alpha' : alpha, 'eps': eps}
            self.d = d

        def sample_X(self, n):
            cov_X = (1 - alpha) * np.eye(self.d) + alpha * np.ones((self.d, self.d))
            return np.random.multivariate_normal(mean=np.zeros(self.d), cov=cov_X, size=n)
        def sample_Y(self, n):
            cov_Y = (1 - (alpha + eps)) * np.eye(self.d) + (alpha + eps) * np.ones((self.d, self.d))
            return np.random.multivariate_normal(mean=np.zeros(self.d), cov=cov_Y, size=n)

    return themodel


# -------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------- #

lsmodels = []
# lsmodels.append((MODEL_0, {}))  # Add MODEL_0 with empty parameters
lsmodels.append((MODEL_1 , {'mu' : 0.3, 'P' :  20}))
lsmodels.append((MODEL_2 , {'spike_value' : 4, 'num_spikes' : 8}))
lsmodels.append((MODEL_3 , {'mu' : 0.3,  'P' :  10,  'spike_value' : 4, 'num_spikes' : 5 }))
lsmodels.append((MODEL_4 , {'epsilon' : 0.1, 'P' : 30}))
lsmodels.append((MODEL_5 , {'mu' : 0.75, 'P' : 25}))
lsmodels.append((MODEL_6 , {'eps' : 0.075}))
lsmodels.append((MODEL_7 , {'alpha' : 0.5, 'eps' : 0.25})) 
lsmodels.append((MODEL_8 , {'alpha' : 0.5, 'eps' : 0.05}))



