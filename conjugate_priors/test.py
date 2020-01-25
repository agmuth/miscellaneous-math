import numpy as np
import unittest

from likelihoods import *
from priors import *


class TestBinomialLikelihood(unittest.TestCase):
    def test_binomial_beta(self):
        prior = BetaPrior(alpha = 1, beta = 1)
        likelihood = BinomialLikelihood(prior = prior, n = 10)

        obvs1 = np.array([10, 10, 10])
        obvs2 = np.array([0, 0, 0])

        likelihood.update(obvs1)
        self.assertEqual(likelihood.posterior.alpha, 31, "First Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.beta, 1, "First Bayesian update unsucessful.")

        likelihood.update(obvs2)
        self.assertEqual(likelihood.posterior.alpha, 31, "Second Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.beta, 31, "Second Bayesian update unsucessful.")

        self.assertEqual(likelihood.prior.alpha, 1, "Bayesian update effected prior distribution.")
        self.assertEqual(likelihood.prior.beta, 1, "Bayesian update effected prior distribution.")

    def test_binomial_unsupported(self):
        prior = GammaPrior(shape = 1, scale = 1)
        self.assertRaises(TypeError, BinomialLikelihood, prior, "Unsupported prior not correctly caught.")

class TestNegativeBinomialLikelihood(unittest.TestCase):
    def test_negativebinomial_beta(self):
        prior = BetaPrior(alpha = 1, beta = 1)
        likelihood = NegativeBinomialLikelihood(prior = prior, r = 5)

        obvs1 = np.array([5, 5, 5])
        obvs2 = np.array([10, 10, 10])

        likelihood.update(obvs1)
        self.assertEqual(likelihood.posterior.alpha, 16, "First Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.beta, 1, "First Bayesian update unsucessful.")

        likelihood.update(obvs2)
        self.assertEqual(likelihood.posterior.alpha, 31, "Second Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.beta, 16, "Second Bayesian update unsucessful.")

        self.assertEqual(likelihood.prior.alpha, 1, "Bayesian update effected prior distribution.")
        self.assertEqual(likelihood.prior.beta, 1, "Bayesian update effected prior distribution.")

    def test_negativebinomial_unsupported(self):
        prior = GammaPrior(shape = 1, scale = 1)
        self.assertRaises(TypeError, NegativeBinomialLikelihood, prior, "Unsupported prior not correctly caught.")

class TestPoissonLikelihood(unittest.TestCase):
    def test_poisson_gamma(self):
        prior = GammaPrior(shape = 1, scale = 1)
        likelihood = PoissonLikelihood(prior = prior)
        
        obvs1 = np.array([1, 0, 2])
        obvs2 = np.array([3, 3, 3])

        likelihood.update(obvs1)
        self.assertEqual(likelihood.posterior.shape, 4, "First Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.rate, 4, "First Bayesian update unsucessful.")

        likelihood.update(obvs2)
        self.assertEqual(likelihood.posterior.shape, 13, "Second Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.rate, 7, "Second Bayesian update unsucessful.")

        self.assertEqual(likelihood.prior.shape, 1, "Bayesian update effected prior distribution.")
        self.assertEqual(likelihood.prior.rate, 1, "Bayesian update effected prior distribution.")

    def test_poisson_unsupported(self):
        prior = BetaPrior(alpha = 1, beta = 1)
        self.assertRaises(TypeError, PoissonLikelihood, prior, "Unsupported prior not correctly caught.")

class TestUniformLikelihood(unittest.TestCase):
    def test_uniform_pareto(self):
        prior = ParetoPrior(shape = 1, scale = 1)
        likelihood = UniformLikelihood(prior = prior)

        obvs1 = np.array([0.1, 0.2, 0.3])
        obvs2 = np.array([1.1, 1.2, 1.3])

        likelihood.update(obvs1)
        self.assertEqual(likelihood.posterior.scale, 1, "First Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.shape, 4, "First Bayesian update unsucessful.")

        likelihood.update(obvs2)
        self.assertEqual(likelihood.posterior.scale, 1.3, "Second Bayesian update unsucessful.")
        self.assertEqual(likelihood.posterior.shape, 7, "Second Bayesian update unsucessful.")

        self.assertEqual(likelihood.prior.scale, 1, "Bayesian update effected prior distribution.")
        self.assertEqual(likelihood.prior.shape, 1, "Bayesian update effected prior distribution.")

    def test_binomial_unsupported(self):
        prior = GammaPrior(shape = 1, scale = 1)
        self.assertRaises(TypeError, UniformLikelihood, prior, "Unsupported prior not correctly caught.")

if __name__ == '__main__':
    unittest.main()