import numpy as np

class Prior(object):
    pass

class BetaPrior(Prior):
    """
    A class used to represent a Beta distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    alpha: double 
        positive real number corresponding to the alpha parameter of the Beta distribution

    beta: double 
        positive real number corresponding to the beta parameter of the Beta distribution

    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Beta distribution using numpy
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, size):
        return np.random.beta(a = self.alpha, b = self.beta, size = size)

class NormalPrior(Prior):
    """
    A class used to represent a Normal distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    mean: double 
        real number corresponding to the mean of the Normal distribution

    var: double 
        positive real number corresponding to the variance of the Normal distribution

    prec: double 
        positive real number corresponding to the precision of the Normal distribution, defined as 1 / variance

    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Normal distribution using numpy
    """

    def __init__(self, mean, var, prec = None):
        self.mean = mean
        if prec is not None: 
            self.prec = prec
            self.var = 1 / prec
        else:
            self.var = var

    def sample(self, size):
        return np.random.normal(loc = self.mean, scale = self.var, size = size)

class GammaPrior(Prior):
    """
    A class used to represent a Gamma distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    shape: double 
        real number corresponding to the shape parameter of the Gamma distribution (often written as alpha or kappa)

    scale: double 
        positive real number corresponding to the scale parameter of the Gamma distribution (often written as theta)

    rate: double 
        positive real number corresponding to the rate/inverse scale parameter of the Gamma distribution, defined as 1 / scale (often written as beta)

    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Gamma distribution using numpy
    """

    def __init__(self, shape, scale, rate = None):
        self.shape = shape
        if rate is not None:
            self.scale = 1 / rate
            self.rate = rate            
        else:
            self.scale = scale
            self.rate = 1 / scale

    def sample(self, size):
        return np.random.gamma(shape = self.shape, scale = self.scale, size = size)

class BetaBinomialPrior(Prior):
    """
    A class used to represent a Beta-Binomial distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    n: int 
        positive integer number corresponding to the number of trials of the Beta-Binomial distribution 

     alpha: double 
        positive real number corresponding to the alpha parameter of the Beta-Binomial distribution

    beta: double 
        positive real number corresponding to the beta parameter of the Beta-Binomial distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Beta-Binomial distribution using numpy
        by first drawing a probability of succes from a beta distribution and then a binomial 
        random variable using this probability
    """

    def __init__(self, n, alpha, beta):
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def sample(self, size):
        sample = np.zeros(size)
        for i in range(size):
            prob = np.random.beta(a = self.alpha, b = self.beta, size = 1)
            sample[i] = np.random.binomial(n = self.n, p = prob)
        return sample

class DirichletPrior(Prior):
    """
    A class used to represent a Dirichlet distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    alpha: array 
        array of positive real numbers corresponding to the alpha parameter of the Dirichlet distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Dirichlet distribution using numpy
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, size):
        return np.random.dirichlet(alpha = alpha, size = size)

class InverseGammaPrior(Prior):
    """
    A class used to represent a Inverse Gamma distribution to be used as a prior/posterior by a likelihood object.
    An Inverse Gamma random variable is the reicprocal of a Gamma Random variable.
    ...

    Attributes
    ----------
    shape: double 
        real number corresponding to the shape parameter of the Inverse Gamma distribution (often written as alpha or kappa)

    scale: double 
        positive real number corresponding to the scale parameter of the Inverse Gamma distribution (often written as theta)

    rate: double 
        positive real number corresponding to the rate/inverse scale parameter of the Inverse Gamma distribution, defined as 1 / scale (often written as beta)

    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Gamma distribution using numpy
        Inverse Gamma random variables are generated by return the reciprocal of Gamma
        random variables with the same parameters
    """

    def __init__(self, shape, scale, rate = None):
        self.shape = shape
        if rate is not None:
            self.scale = 1 / rate
            self.rate = rate            
        else:
            self.scale = scale
            self.rate = 1 / scale

    def sample(self, size):
        return 1 / np.random.gamma(shape = self.shape, scale = self.scale, size = size)

class NormalInverseGammaPrior(Prior):
    """
    A class used to represent a Normal Inverse Gamma distribution to be used as a prior/posterior by a likelihood object.
    ...

    Attributes
    ----------
    mean: double 
        real number corresponding to the mean of the Normal distribution

    prec: double 
        positive real number corresponding to the precision of the Normal distribution, defined as 1 / variance

    shape: double 
        real number corresponding to the shape parameter of the Inverse Gamma distribution (often written as alpha or kappa)

    scale: double 
        positive real number corresponding to the scale parameter of the Inverse Gamma distribution (often written as theta)


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Normal Inverse Gamma distribution using numpy
        using a two step procedure first sampling from the Inverse Gamma distribution and then from 
        the conditional Normal distribution
    """

    def __init__(self, mean, prec, scale, shape):
        self.mean = mean
        self.prec = prec
        self.scale = scale
        self.shape = shape

    def sample(self, size):
       sample_var = 1 / np.random.gamma(shape = self.shape, scale = self.scale, size = size) #  variance sample from inv gamma
       sample_normal = np.zeros(size) #  sample from conditional normal distn
       for i in range(size):
           var = sample_var[i] / self.prec
           sample_normal[i] = np.random.normal(loc = self.mean, scale = var, size = 1)
       return sample_normal, sample_var

class NormalGammaPrior(Prior):
    """
    A class used to represent a Normal Gamma distribution to be used as a prior/posterior by a likelihood object.
    ...

    Attributes
    ----------
    mean: double 
        real number corresponding to the mean of the Normal distribution

    prec: double 
        positive real number corresponding to the precision of the Normal distribution, defined as 1 / variance

    shape: double 
        real number corresponding to the shape parameter of the Gamma distribution (often written as alpha or kappa)

    scale: double 
        positive real number corresponding to the scale parameter of the Gamma distribution (often written as theta)


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Normal Gamma distribution using numpy
        using a two step procedure first sampling from the Gamma distribution and then from 
        the conditional Normal distribution
    """

    def __init__(self, mean, prec, scale, shape):
        self.mean = mean
        self.prec = prec
        self.scale = scale
        self.shape = shape

    def sample(self, size):   
        sample_prec = np.random.gamma(shape = self.shape, scale = self.scale, size = size) #  precision sample from gamma
        sample_normal = np.zeros(size) #  sample from conditional normal distn
        for i in range(size):
            var = 1 / (sample_prec[i] * self.prec)
            sample_normal[i] = np.random.normal(loc = self.mean, scale = var, size = 1)
        return sample_normal, sample_prec

class MultivariateNormalPrior(Prior):
    """
    A class used to represent a Multivariate Normal distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    mean: array 
        array of real numbers corresponding to the mean of the Multivariate Normal distribution

    cov: matrix 
        square matrix of real numbers corresponding to the covariance matrix of the Multivariate Normal distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Multivariate Normal distribution using numpy
    """

    def __init__(self, mean, cov):
        self.mean = mean 
        self.cov = cov

    def sample(self, size):
        return np.random.multivariate_normal(mean = self.mean, cov = self.cov, size = size)

class ScaledInverseChiSquaredPrior(Prior):
    """
    A class used to represent a Scaled Inverse Chi-Squared distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    df: int 
        positive integer corresponding to the degrees of freedom of the Scaled Inverse Chi-Squared distribution

    scale: double 
        positive real number corresponding to the scale parameter of the Scaled Inverse Chi-Squared distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Scaled Inverse Chi-Squared distribution Normal distribution using numpy
    """

    def __init__(self, df, scale):
        self.df = df
        self.scale = scale

    def sample(self, size):
        return 1 / np.random.gamma(shape = self.df / 2, scale = self.scale * self.df / 2, size = size)

class WishartPrior(Prior):
    """
    A class used to represent a Wishart distribution to be used as a prior/posterior by a likelihood object.
    The Wishart distribution is the multidimensional generalization of the Gamma distribution.
    ...

    Attributes
    ----------
    df: int 
        positive integer corresponding to the degrees of freedom of the Wishart distribution, must be greater than 
        p - 1 where p is the number of rows / columns of the covariance matrix

    cov: matrix 
        square matrix of real numbers corresponding to the covariance matrix of the Wishart distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Scaled Wishart distribution Normal distribution using numpy
    """

    def __init__(self, df, cov):
        # df must be greater than p-1
        if df < cov.shape[0]:
            raise Exception('Degrees of freedom must be at least number of rows/columns of covariance matrix.')
        self.df = df 
        self.cov

    def sample(self, size):
        sample = np.zeros(size)
        for i in range(size):
            x = np.random.multivariate_normal([0, 0], self.cov, self.df)
            x = np.matrix(x)
            sample[i] = np.matmul(x, x.T)
        return sample


class InverseWishartPrior(WishartPrior):
    """
    A class used to represent a Inverse Wishart distribution to be used as a prior/posterior by a likelihood object.
    The Inverse Wishart distribution is the multidimensional generalization of the Inverse Gamma distribution.
    ...

    Attributes
    ----------
    df: int 
        positive integer corresponding to the degrees of freedom of the Inverse Wishart distribution, must be greater than 
        p - 1 where p is the number of rows / columns of the inverse covariance matrix

    inv_cov: matrix 
        square matrix of real numbers corresponding to the inverse covariance matrix of the Inverse Wishart distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Scaled Inverse Wishart distribution Normal distribution using numpy
    """

    def __init__(self, df, inv_cov):
        # df must be greater than p-1
        if df < inv_cov.shape[0]:
            raise Exception('Degrees of freedom must be at least number of rows/columns of precision matrix.')
        self.df = df #  must be greater than p-1
        self.inv_cov
        self.cov = np.linalg.inv(self.inv_cov) #  invert for sampling

class ParetoPrior(Prior):
    """
    A class used to represent a Pareto distribution to be used as a prior/posterior by a likelihood object.
    
    ...

    Attributes
    ----------
    shape: double 
        real number corresponding to the shape parameter of the Pareto distribution (often written as alpha)

    scale: double 
        positive real number corresponding to the scale parameter/minimum value of the Pareto distribution (often written as x_min)


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Pareto distribution using numpy
    """

    def __init__(self, scale, shape):
        self.scale = scale
        self.shape = shape

    def sample(self, size):
        return (1 + np.random.pareto(self.shape, size)) * self.scale


class NormalWishartPrior(Prior):
    """
    A class used to represent a Normal Wishart distribution to be used as a prior/posterior by a likelihood object.
    The Normal Wishart distribution is the multidimensional generalization of the Normal Gamma distribution.
    ...

    Attributes
    ----------
     mean: array 
        array of real numbers corresponding to the mean of the Multivariate Normal distribution

    prec: matrix 
        square matrix of real numbers corresponding to the inverse covariance matrix of the Multivariate Normal distribution

    df: int 
        positive integer corresponding to the degrees of freedom of the Wishart distribution, must be greater than 
        p - 1 where p is the number of rows / columns of the covariance matrix

    cov: matrix 
        square matrix of real numbers corresponding to the covariance matrix of the Wishart distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Normal Wishart distribution using numpy
    """
    
    def __init__(self, mean, prec, df, cov):
        self.mean = mean
        self.prec = prec
        self.df = df
        self.cov = cov

    def sample(self, size):
        sample_wishart = []
        sample_normal  = []
        for i in range(size):
            sample_cov = np.random.multivariate_normal([0, 0], self.cov, self.df)
            sample_cov = np.matrix(sample_cov)
            sample_cov = np.matmul(sample_cov, sample_cov.T)
            sample_mean = np.random.multivariate_normal(self.mean, \
                np.linalg.inv(np.matmul(self.prec, sample_cov)), 1)
            sample_wishart.append(sample_cov)
            sample_normal.append(sample_mean)

        return sample_normal, sample_wishart

class NoramlInverseWishartPrior(NormalWishartPrior):
    """
    A class used to represent a Normal Inverse Wishart distribution to be used as a prior/posterior by a likelihood object.
    The Normal Inverse Wishart distribution is the multidimensional generalization of the Normal Inverse Gamma distribution.
    ...

    Attributes
    ----------
     mean: array 
        array of real numbers corresponding to the mean of the Multivariate Normal distribution

    prec: matrix 
        square matrix of real numbers corresponding to the inverse covariance matrix of the Multivariate Normal distribution

    df: int 
        positive integer corresponding to the degrees of freedom of the Inverse Wishart distribution, must be greater than 
        p - 1 where p is the number of rows / columns of the covariance matrix

    cov: matrix 
        square matrix of real numbers corresponding to the covariance matrix of the Inverse Wishart distribution


    Methods
    -------
    sample(size):
        returns a sample of size size from the defined Normal Inverse Wishart distribution using numpy
    """
    def __init__(self, mean, prec, df, inv_cov):
        self.mean = mean
        self.prec = prec
        self.df = df
        self.inv_cov = inv_cov
        self.cov = np.linalg.inv(self.inv_cov)

    def sample(self, size):
        sample_wishart = []
        sample_normal  = []
        for i in range(size):
            sample_cov = np.random.multivariate_normal([0, 0], self.cov, self.df)
            sample_cov = np.matrix(sample_cov)
            sample_cov = np.matmul(sample_cov, sample_cov.T)
            sample_mean = np.random.multivariate_normal(self.mean, \
                np.linalg.inv(np.matmul(self.prec, sample_cov)), 1)
            sample_wishart.append(sample_cov)
            sample_normal.append(sample_mean)
        return sample_normal, 1 / sample_wishart