import numpy as np
import copy
from priors import * 

class Likelihood(object):
    
    def __init__(self, prior = None):
        self._set_prior(prior)
        self.data = None

    def _set_prior(self, prior):
        pass

    def update(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate((self.data, data))

        self._posterior_update(data)

    def _posterior_update(self, obvs):
        pass


class BinomialLikelihood(Likelihood):

    def __init__(self, prior, n):
        self.n = n
        super().__init__(prior)

    def _set_prior(self, prior):
        if type(prior) != BetaPrior: 
            raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.alpha += sum(obvs)
        self.posterior.beta += len(obvs) * self.n - sum(obvs)
            

class BernoulliLikelihood(BinomialLikelihood):

    def __init__(self, prior):
        super().__init__(prior, n = 1)

class NegativeBinomialLikelihood(Likelihood):

    def __init__(self, prior, r):
        self.r = r
        super().__init__(prior)

    def _set_prior(self, prior):
        if type(prior) != BetaPrior: 
            raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.alpha += len(obvs) * self.r
        self.posterior.beta += sum(obvs) - len(obvs) * self.r
            

class PoissonLikelihood(Likelihood):

    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) != GammaPrior: 
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.shape += sum(obvs)
        self.posterior.rate += len(obvs)    
        self.posterior.scale = 1 / self.posterior.rate
            

class MultinomialPrior(Prior):
    
    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) != DirichletPrior: 
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.alpha = self.posterior.alpha + sum(obvs) #only supports list change to numpy matrix
 
        
class CategoricalLikelihood(MultinomialPrior):
    pass

class HypergeometricPrior(Prior): #not done

    def _set_prior(self, prior):
        if type(prior) != BetaBinomialPrior: 
            raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.shape += obvs
        self.posterior.rate += 1        
        self.posterior.scale = 1 / self.posterior.rate

class GeometricLikelihood(Prior):

    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) != BetaPrior: 
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.alpha += len(obvs)
        self.posterior.beta += sum(obvs) - len(obvs)

class NormalLikelihood(Prior):

    def __init__(self, prior, mean = None, var = None, prec = None):
        self.mean = mean
        self.var = var
        self.prec = prec
        super().__init__(prior)

    def _set_prior(self, prior):
        if type(prior) == NormalPrior:
            if self.prec is not None:
                self.var = 1 / self.prec
            elif self.var is not None:
                self.prec = 1 / self.var
            else:
                raise Exception('One of variance or precision must be specified when using a Normal prior for the mean.') 
            
        elif type(prior) == InverseGammaPrior:
            if self.mean is  None:
                raise Exception('Mean must be specified when using an Inverse Gamma prior for the variance')
                
        elif type(prior) == ScaledInverseChiSquaredPrior:
            if self.mean is  None:
                raise Exception('Mean must be specified when using a Scaled Inverse Chi-Squared prior for the variance')

        elif type(prior) == GammaPrior:
            if self.mean is  None:
                raise Exception('Mean must be specified when using a Gamma prior for the precision')

        elif type(prior) == NormalInverseGammaPrior:
            pass

        elif type(prior) == NormalGammaPrior:
            pass

        else:
            raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

        def _posterior_update(self, obvs):
            if type(self.posterior) == NormalPrior:
               self.posterior.mean =  (self.posterior.mean / self.posterior.var + sum(obvs) / self.var) / \
                   (1 / self.posterior.var + len(obvs) / self.var)
               self.posterior.var =  1 / (1 / self.posterior.var + len(obvs) / self.var)
               self.posterior.prec = 1 / self.posterior.var

            elif type(self.posterior) == InverseGammaPrior:
               self.posterior.scale += len(obvs) / 2
               self.posterior.rate += sum(obvs - self.mean) / 2
               self.posterior.scale = 1 / self.posterior.rate

            elif type(self.posterior) == ScaledInverseChiSquaredPrior:
               self.posterior.scale = (self.posposterior.df * self.posterior.scale + sum((obvs - self.mean)**2)) / \
                   (self.posterior.df + len(obvs))
               self.posterior.df += len(obvs)
               
            elif type(self.posterior) == GammaPrior:
               self.posterior.scale += len(obvs) / 2
               self.posterior.rate += sum(obvs - self.mean) / 2
               self.posterior.scale = 1 / self.posterior.rate
               
            elif type(self.posterior) == NormalInverseGammaPrior:
                posterior_mean = (self.posterior.prec * self.posterior.mean + sum(obvs)) / (self.posterior.prec + len(obvs))
                posterior_prec = self.posterior.prec + len(obvs)
                posterior_scale = self.posterior.scale + len(obvs) / 2
                posterior_rate = self.posterior.rate + 0.5 * sum((obvs - np.mean(obvs)**2)) + \
                    len(obvs) * self.posterior.prec / (len(obvs) + self.posterior.prec) * 0.5 * (np.mean(obvs) - self.posterior.mean)**2

                self.posterior.mean = posterior_mean
                self.posterior.prec = posterior_prec
                self.posterior.scale = posterior_scale
                self.posterior.rate = posterior_rate

            elif type(self.posterior) == NormalGammaPrior:
                posterior_mean = (self.posterior.prec * self.posterior.mean + sum(obvs)) / (self.posterior.prec + len(obvs))
                posterior_prec = self.posterior.prec + len(obvs)
                posterior_scale = self.posterior.scale + len(obvs) / 2
                posterior_rate = self.posterior.rate + 0.5 * sum((obvs - np.mean(obvs)**2)) + \
                    len(obvs) * self.posterior.prec / (len(obvs) + self.posterior.prec) * 0.5 * (np.mean(obvs) - self.posterior.mean)**2

                self.posterior.mean = posterior_mean
                self.posterior.prec = posterior_prec
                self.posterior.scale = posterior_scale
                self.posterior.rate = posterior_rate
            else:
                raise Exception('Something went horribily wrong this statement should never be reached.')

class LognormalLikelihood(NormalLikelihood):
    def _posterior_update(self, obvs):
        super()._posterior_update(self, np.log(obvs))

class MultivariateNormalLikelihood(Likelihood):
    pass

class InverseGaussianLikelihood(NormalLikelihood):
    pass

class UniformLikelihood(Likelihood):

    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) != ParetoPrior: 
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.scale = max(obvs, self.posterior.scale)
        self.posterior.shape += len(obvs)


class ParetoLikelihood(Likelihood):

    def __init__(self, prior, scale = None, shape = None):
        super().__init__(prior)

    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) == ParetoPrior: 
                if shape is None:
                    raise Exception('Must specify shape parameter when using Pareto prior for scale parameter.')
                self.shape = shape
            elif type(prior) == GammaPrior: 
                if scale is None:
                    raise Exception('Must specify scale parameter when using Gamma prior for shape parameter.')
                self.scale = scale
            else:
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):

        if type(prior) == GammaPrior: 
            self.posterior.shape += len(obvs)
            self.posterior.rate += sum(np.log(obvs / max(obvs)))
            self.posterior.scale = 1 / self.posterior.rate
        elif type(prior) != ParetoPrior: 
            self.posterior.shape -= self.shape * len(obvs)
            #  no update for scale parameter
        else:
            raise Exception('Something went horribily wrong this statement should never be reached.')


        

class WeibullLikelihood(Likelihood):
    
    def __init__(self, prior, shape):
        self.shape = shape
        super().__init__(prior)

    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) != InverseGammaPrior: 
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.shape += len(obvs)
        self.posterior.rate += sum(obvs**self.shape)
        self.posterior.scale = 1 / self.posterior.rate


class ExponentialLikelihood(Likelihood):
    def _set_prior(self, prior):
        def _set_prior(self, prior):
            if type(prior) != GammaPrior: 
                raise TypeError('Prior distribution of type {} is not supported.' .format(type(prior)))

        self.prior = prior
        self.posterior = copy.deepcopy(self.prior)

    def _posterior_update(self, obvs):
        self.posterior.scale += len(obvs)
        self.posterior.rate += sum(obvs)
        self.posterior.scale = 1 / self.posterior.rate

class GammaLikelihood(Likelihood):
    pass

class InverseGammaLikelihood(Likelihood):
    pass

