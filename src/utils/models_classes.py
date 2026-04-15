import numpy as np
import math
from scipy.stats import vonmises_fisher

def spiked_covariance(d, num_spikes=1, spike_value=10, template_cov=None):
    if template_cov is None:
        cov_matrix = np.eye(d)
    else:
        cov_matrix = template_cov(d)
    cov_matrix[-num_spikes:, -num_spikes:] = spike_value
    return cov_matrix

# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def null():
    """
    Creates a model class with methods to sample from two isotropic Gaussian distributions with identical parameters.

    Returns:
    themodel (class): A class representing the model with methods to sample from distributions X, Y, and null.

    Class Attributes:
    __name__ (str): The name of the model.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = 'null'
        def __init__(self, d):
            self.d = d
        def sample_X(self,n):
            return np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d)/np.sqrt(self.d), n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d)/np.sqrt(self.d), n)
    return themodel
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def GaussianMeanShift(mu):
    """
    Creates a model class with specified parameters for drawing samples from two isotropic Gaussian distributions with identical covariance but differing means.
    Both models have isotropic covariance; the first is centred, whereas the second has a mean vector shifted by `mu`.
    
    Parameters:
    mu (float): The mean value to be assigned to the second model's mean vector.
    Returns:
    themodel (class): A class representing the model with methods to sample from different distributions.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameter 'mu'.

        Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """

    class themodel:
        __name__ = 'Gaussian Mean Shift'
        def __init__(self, d):
            self.params = {'mu': mu}
            self.d = d
            self.location = np.zeros(d)
            self.shift = np.ones(d) * mu
            self.scale = 1.0
        def sample_X(self, n):
            return np.random.normal(loc=self.location, scale=self.scale, size=(n, self.d))
        def sample_Y(self, n):
            return np.random.normal(loc=self.shift, scale=self.scale, size=(n, self.d))
    return themodel
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def GaussianSparseMeanShift(mu, P):
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
        __name__ = 'Gaussian Sparse Mean Shift'
        def __init__(self, d, P = P):
            P = min(P, d)  # Ensure P does not exceed d
            self.params = {'mu': mu, 'P': P}
            self.d = d

            self.mX = np.zeros(d); 
            self.mY = np.zeros(d); self.mY[:P] = mu
            
            self.covX = np.eye(d); self.covY =  np.eye(d) 

        def sample_X(self,n):
            return np.random.multivariate_normal(self.mX, self.covX, n)
        def sample_Y(self,n):
            return np.random.multivariate_normal(self.mY, self.covY, n)

    return themodel
# ------------------------ # ------------------------ # ------------------------ # ------------------------ 
def GaussianSpikedCovariance(spike_value, num_spikes):
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
        __name__ = 'Gaussian Spiked Covariance'
        def __init__(self, d, spike_value=spike_value, num_spikes=num_spikes): 

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
def UniformThinHypercube(epsilon, P):
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
        __name__ = 'Uniform Thin Hypercube'
        def __init__(self, d, P = P):
            assert 0 < epsilon < 1, "epsilon must be in (0, 1)"
            P = min(P,d)  # Ensure P does not exceed d
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
def GaussianMixture(mu, P):
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
        __name__ = 'Gaussian Mixture'
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
def DiracGaussianMixture(eps):
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
        __name__ = 'Dirac-Gaussian Mixture'
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
def DecreasingCorrelationGaussian(alpha, eps):
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
        __name__ = ',Gaussian, Decreasing Correlation'
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
def EquiCorrelationGaussian(alpha, eps):
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
        __name__ = 'Gaussian, Equi-Correlated '
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
def VMF(kappa):
    """ 
    Creates a model class for sampling from a radial distribution using the von Mises-Fisher distribution.
    This model samples points uniformly from the surface of a unit sphere in d dimensions for X,
    and samples points from a symmetric von Mises-Fisher distribution for Y.
    Parameters:
    kappa (float): Concentration parameter for the von Mises-Fisher distribution.
    Returns:
    themodel (class): A class representing the model with methods to sample from distributions X and Y.
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameter 'kappa'.
    Methods:
    sample_X(n_samples): Samples n_samples points uniformly from the surface of a unit sphere in d dimensions.
    sample_Y(n_samples): Samples n_samples points from a symmetric von Mises-Fisher distribution.
    """


    def sample_uniform_sphere(n_samples, d):
        x = np.random.normal(size=(n_samples, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def sample_symmetric_vMF_scipy(n_samples, d, kappa = kappa):
        mu1 = np.zeros(d); mu1[0] = 1.0
        mu2 = np.zeros(d); mu2[-1] = 1.0
        quartern = n_samples // 4
        return np.vstack([ vonmises_fisher(mu1, kappa).rvs(size=quartern),
                           vonmises_fisher(-mu1, kappa).rvs(size=quartern),
                           vonmises_fisher(mu2, kappa).rvs(size=quartern),
                           vonmises_fisher(-mu2, kappa).rvs(size=quartern)]
                           ).reshape(-1, d)
    class themodel:
        __name__ = 'Radial Distribution (vMF)'
        def __init__(self, d, kappa=kappa):
            self.params = {'kappa': kappa}
            self.kappa = kappa
            self.d = d

        def sample_X(self, n_samples):
            return sample_uniform_sphere(n_samples, self.d)

        def sample_Y(self, n_samples):
            return sample_symmetric_vMF_scipy(n_samples, self.d, self.kappa)
    return themodel
# -------------------------------------------------------------------------------------------------------- #
import numpy as np
import math
from scipy.stats import vonmises_fisher
lsmodels = []
lsmodels.append((null, {}))  # Add MODEL_0 with empty parameters
lsmodels.append((GaussianSpikedCovariance , {'spike_value' : 4, 'num_spikes' : 8}))
lsmodels.append((UniformThinHypercube , {'epsilon' : 0.1, 'P' : 30}))
lsmodels.append((GaussianMixture , {'mu' : 0.75, 'P' : 25}))
lsmodels.append((DiracGaussianMixture , {'eps' : 0.075}))
lsmodels.append((DecreasingCorrelationGaussian , {'alpha' : 0.5, 'eps' : 0.25})) 
lsmodels.append((EquiCorrelationGaussian , {'alpha' : 0.5, 'eps' : 0.05}))
lsmodels.append((VMF , {'kappa' : 2.5}))
lsmodels.append((GaussianMeanShift , {'mu' : 0.5}))  # Uncomment if needed




def LaplaceSparseMeanShift(mu,P):
    """
    Creates a model class for sampling from two isotropic Laplace distributions
    with identical scale but differing means.

    Parameters:
    mu (float): The mean value to be assigned to the second model's mean vector.

    Returns:
    themodel (class): A class representing the model with sampling methods.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameter 'mu'.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """

    class themodel:
        __name__ = 'Laplace Sparse Mean Shift'
        def __init__(self, d):
            self.params = {'mu': mu, 'P' : P}
            self.d = d

            self.location = np.zeros(d)
            self.shift = np.ones(d) * mu; self.shift[P:] =0
            self.scale = 1.0  # Laplace scale (b), not std deviation

        def sample_X(self, n):
            return np.random.laplace(loc=self.location, scale=self.scale, size=(n, self.d))

        def sample_Y(self, n):
            return np.random.laplace(loc=self.shift, scale=self.scale, size=(n, self.d))

    return themodel


def LaplaceGaussian():
    """
    Creates a model class for sampling from two isotropic Laplace distributions
    with identical scale but differing means.

    Parameters:
    mu (float): The mean value to be assigned to the second model's mean vector.

    Returns:
    themodel (class): A class representing the model with sampling methods.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameter 'mu'.

    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """

    class themodel:
        __name__ = 'Laplace vs Gaussian'
        def __init__(self, d):
            self.params = {}
            self.d = d

            self.location = np.zeros(d)
            self.scale = 1.0 

        def sample_X(self, n):
            return np.random.laplace(loc=self.location, scale=self.scale, size=(n, self.d))

        def sample_Y(self, n):
            return np.random.normal(loc=self.location, scale=self.scale, size=(n, self.d))

    return themodel


def ConcentricSpheres(shift, noise):
    '''
    Creates a model class for sampling from two concentric spheres with different radii.
    Returns:
    themodel (class): A class representing the model with sampling methods.
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'radius_X' and 'radius_Y'.
    Methods:
    sample_X(n): Samples n points from the sphere with radius_X.
    sample_Y(n): Samples n points from the sphere with radius_Y.
    '''
    def sample_uniform_sphere(n, d, radius=1.0):
        """
        Samples n points uniformly from the surface of a sphere in d dimensions with a given radius.
        """
        x = np.random.normal(size=(n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x * radius

    class themodel:
        __name__ = 'Concentric Spheres'
        def __init__(self, d):
            self.params = {'shift': shift, 'noise': noise}
            self.d = d

        def sample_X(self, n):
            return sample_uniform_sphere(n, self.d, np.ones(self.d)) + np.random.normal(0, self.params['noise'], (n, self.d))
        def sample_Y(self, n):
            return sample_uniform_sphere(n, self.d,  np.ones(self.d) + self.params['shift']) + np.random.normal(0, self.params['noise'], (n, self.d))
    return themodel