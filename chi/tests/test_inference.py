#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import unittest

import arviz as az
import numpy as np
import pandas as pd
import pints
import xarray as xr

import chi
from chi.library import DataLibrary, ModelLibrary


class TestComputePointwiseLogLikelihood(unittest.TestCase):
    """
    Tests the chi.compute_pointwise_loglikelihood function.
    """

    @classmethod
    def setUpClass(cls):
        # Test case I: Non-hierarchical log-likelihood
        # Get test data and model
        data = DataLibrary().lung_cancer_control_group()
        individual = 40
        mask = data['ID'] == individual  # Arbitrary test id
        data = data[mask]
        mask = data['Biomarker'] == 'Tumour volume'  # Arbitrary biomarker
        times = data[mask]['Time'].to_numpy()
        observed_volumes = data[mask]['Measurement'].to_numpy()

        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()
        cls.log_likelihood = chi.LogLikelihood(
            mechanistic_model, error_model, observed_volumes, times)
        cls.log_likelihood.set_id(individual)

        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 1
        samples = np.ones(shape=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1']})
        cls.posterior_samples = xr.Dataset({
            param: samples for param
            in cls.log_likelihood.get_parameter_names()})

        # Test case II: Hierarchical Log-likelihood
        cls.log_likelihood_2 = chi.LogLikelihood(
            mechanistic_model, error_model, observed_volumes, times)
        cls.log_likelihood_2.set_id(56)
        pop_models = [
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()]
        cls.hierarch_log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods=[cls.log_likelihood, cls.log_likelihood_2],
            population_models=pop_models)

        # Create posterior
        n_chains = 2
        n_draws = 3
        n_ids = 2
        bottom_samples = np.ones(shape=(n_chains, n_draws, n_ids))
        top_samples = np.ones(shape=(n_chains, n_draws))
        bottom_samples = xr.DataArray(
            data=bottom_samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 40', 'ID 56']})
        top_samples = xr.DataArray(
            data=top_samples,
            dims=['chain', 'draw'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        parameter_names = cls.hierarch_log_likelihood.get_parameter_names()
        cls.hierarch_posterior_samples = xr.Dataset({
            parameter_names[0]: top_samples,
            parameter_names[1]: bottom_samples,
            parameter_names[3]: top_samples,
            parameter_names[4]: top_samples,
            parameter_names[5]: top_samples,
            parameter_names[6]: bottom_samples,
            parameter_names[8]: top_samples,
            parameter_names[9]: top_samples,
            parameter_names[10]: top_samples})

    def test_call(self):
        # Test case I: Non-hierarchical log-likelihood
        # Test call with defaults
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.log_likelihood, self.posterior_samples)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'observation')

        obs = pw_ll.observation
        self.assertEqual(len(obs), 10)
        self.assertEqual(obs[0], 'Output 1 Observation 1')
        self.assertEqual(obs[1], 'Output 1 Observation 2')
        self.assertEqual(obs[2], 'Output 1 Observation 3')
        self.assertEqual(obs[3], 'Output 1 Observation 4')
        self.assertEqual(obs[4], 'Output 1 Observation 5')
        self.assertEqual(obs[5], 'Output 1 Observation 6')
        self.assertEqual(obs[6], 'Output 1 Observation 7')
        self.assertEqual(obs[7], 'Output 1 Observation 8')
        self.assertEqual(obs[8], 'Output 1 Observation 9')
        self.assertEqual(obs[9], 'Output 1 Observation 10')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Test call with differently ordered posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 1
        samples = np.ones(shape=(n_draws, n_ids, n_chains))
        samples = xr.DataArray(
            data=samples,
            dims=['draw', 'individual', 'chain'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1']})
        posterior_samples = xr.Dataset({
            param: samples for param
            in self.log_likelihood.get_parameter_names()})
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.log_likelihood, posterior_samples)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'observation')

        obs = pw_ll.observation
        self.assertEqual(len(obs), 10)
        self.assertEqual(obs[0], 'Output 1 Observation 1')
        self.assertEqual(obs[1], 'Output 1 Observation 2')
        self.assertEqual(obs[2], 'Output 1 Observation 3')
        self.assertEqual(obs[3], 'Output 1 Observation 4')
        self.assertEqual(obs[4], 'Output 1 Observation 5')
        self.assertEqual(obs[5], 'Output 1 Observation 6')
        self.assertEqual(obs[6], 'Output 1 Observation 7')
        self.assertEqual(obs[7], 'Output 1 Observation 8')
        self.assertEqual(obs[8], 'Output 1 Observation 9')
        self.assertEqual(obs[9], 'Output 1 Observation 10')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Select individual
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.log_likelihood, self.posterior_samples, individual='ID 1')

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'observation')

        obs = pw_ll.observation
        self.assertEqual(len(obs), 10)
        self.assertEqual(obs[0], 'Output 1 Observation 1')
        self.assertEqual(obs[1], 'Output 1 Observation 2')
        self.assertEqual(obs[2], 'Output 1 Observation 3')
        self.assertEqual(obs[3], 'Output 1 Observation 4')
        self.assertEqual(obs[4], 'Output 1 Observation 5')
        self.assertEqual(obs[5], 'Output 1 Observation 6')
        self.assertEqual(obs[6], 'Output 1 Observation 7')
        self.assertEqual(obs[7], 'Output 1 Observation 8')
        self.assertEqual(obs[8], 'Output 1 Observation 9')
        self.assertEqual(obs[9], 'Output 1 Observation 10')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Map parameters
        param_map = {'myokit.tumour_volume': 'myokit.tumour_volume'}
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.log_likelihood, self.posterior_samples,
            param_map=param_map)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'observation')

        obs = pw_ll.observation
        self.assertEqual(len(obs), 10)
        self.assertEqual(obs[0], 'Output 1 Observation 1')
        self.assertEqual(obs[1], 'Output 1 Observation 2')
        self.assertEqual(obs[2], 'Output 1 Observation 3')
        self.assertEqual(obs[3], 'Output 1 Observation 4')
        self.assertEqual(obs[4], 'Output 1 Observation 5')
        self.assertEqual(obs[5], 'Output 1 Observation 6')
        self.assertEqual(obs[6], 'Output 1 Observation 7')
        self.assertEqual(obs[7], 'Output 1 Observation 8')
        self.assertEqual(obs[8], 'Output 1 Observation 9')
        self.assertEqual(obs[9], 'Output 1 Observation 10')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Return arviz.DataInference
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.log_likelihood, self.posterior_samples,
            return_inference_data=True)

        self.assertIsInstance(pw_ll, az.InferenceData)

        # Test case II: Hierarchical log-likelihood
        # Test call with defaults
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.hierarch_log_likelihood,
            self.hierarch_posterior_samples)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = pw_ll.individual
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 'ID 40')
        self.assertEqual(ids[1], 'ID 56')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Test call per observation
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.hierarch_log_likelihood,
            self.hierarch_posterior_samples,
            per_individual=False)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'observation')

        obs = pw_ll.observation
        self.assertEqual(len(obs), 20)
        self.assertEqual(obs[0], 'ID 40 Observation 1')
        self.assertEqual(obs[1], 'ID 40 Observation 2')
        self.assertEqual(obs[2], 'ID 40 Observation 3')
        self.assertEqual(obs[3], 'ID 40 Observation 4')
        self.assertEqual(obs[4], 'ID 40 Observation 5')
        self.assertEqual(obs[5], 'ID 40 Observation 6')
        self.assertEqual(obs[6], 'ID 40 Observation 7')
        self.assertEqual(obs[7], 'ID 40 Observation 8')
        self.assertEqual(obs[8], 'ID 40 Observation 9')
        self.assertEqual(obs[9], 'ID 40 Observation 10')
        self.assertEqual(obs[10], 'ID 56 Observation 1')
        self.assertEqual(obs[11], 'ID 56 Observation 2')
        self.assertEqual(obs[12], 'ID 56 Observation 3')
        self.assertEqual(obs[13], 'ID 56 Observation 4')
        self.assertEqual(obs[14], 'ID 56 Observation 5')
        self.assertEqual(obs[15], 'ID 56 Observation 6')
        self.assertEqual(obs[16], 'ID 56 Observation 7')
        self.assertEqual(obs[17], 'ID 56 Observation 8')
        self.assertEqual(obs[18], 'ID 56 Observation 9')
        self.assertEqual(obs[19], 'ID 56 Observation 10')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Test call with differently ordered posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 2
        bottom_samples = np.ones(shape=(n_draws, n_chains, n_ids))
        top_samples = np.ones(shape=(n_draws, n_chains))
        bottom_samples = xr.DataArray(
            data=bottom_samples,
            dims=['draw', 'chain', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 40', 'ID 56']})
        top_samples = xr.DataArray(
            data=top_samples,
            dims=['draw', 'chain'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        parameter_names = self.hierarch_log_likelihood.get_parameter_names()
        hierarch_posterior_samples = xr.Dataset({
            parameter_names[0]: top_samples,
            parameter_names[1]: bottom_samples,
            parameter_names[3]: top_samples,
            parameter_names[4]: top_samples,
            parameter_names[5]: top_samples,
            parameter_names[6]: bottom_samples,
            parameter_names[8]: top_samples,
            parameter_names[9]: top_samples,
            parameter_names[10]: top_samples})
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.hierarch_log_likelihood, hierarch_posterior_samples)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = pw_ll.individual
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 'ID 40')
        self.assertEqual(ids[1], 'ID 56')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Map parameters
        param_map = {
            'Pooled myokit.tumour_volume': 'Pooled myokit.tumour_volume'}
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.hierarch_log_likelihood,
            self.hierarch_posterior_samples,
            param_map=param_map)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = pw_ll.individual
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 'ID 40')
        self.assertEqual(ids[1], 'ID 56')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

        # Return arviz.DataInference
        pw_ll = chi.compute_pointwise_loglikelihood(
            self.hierarch_log_likelihood,
            self.hierarch_posterior_samples,
            return_inference_data=True)

        self.assertIsInstance(pw_ll, az.InferenceData)

        # Test case III: Fully pooled model
        pop_models = [chi.PooledModel()] * 7
        hierarch_log_likelihood = chi.HierarchicalLogLikelihood(
            [self.log_likelihood, self.log_likelihood_2],
            pop_models)
        n_chains = 2
        n_draws = 3
        top_samples = np.ones(shape=(n_draws, n_chains))
        top_samples = xr.DataArray(
            data=top_samples,
            dims=['draw', 'chain'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        parameter_names = hierarch_log_likelihood.get_parameter_names()
        hierarch_posterior_samples = xr.Dataset({
            parameter_names[0]: top_samples,
            parameter_names[1]: top_samples,
            parameter_names[2]: top_samples,
            parameter_names[3]: top_samples,
            parameter_names[4]: top_samples,
            parameter_names[5]: top_samples,
            parameter_names[6]: top_samples})
        pw_ll = chi.compute_pointwise_loglikelihood(
            hierarch_log_likelihood,
            hierarch_posterior_samples)

        dimensions = list(pw_ll.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = pw_ll.individual
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 'ID 40')
        self.assertEqual(ids[1], 'ID 56')

        chains = pw_ll.chain
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)

        draws = pw_ll.draw
        self.assertEqual(len(draws), 3)
        self.assertEqual(draws.loc[0], 0)
        self.assertEqual(draws.loc[1], 1)
        self.assertEqual(draws.loc[2], 2)

    def test_call_bad_input(self):
        # Wrong log-likelihood type
        log_likelihood = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The log-likelihood must be'):
            chi.compute_pointwise_loglikelihood(
                log_likelihood, self.posterior_samples)

        # Wrong posterior samples type
        posterior_samples = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The posterior samples must'):
            chi.compute_pointwise_loglikelihood(
                self.log_likelihood, posterior_samples)

        # The posterior samples have the wrong dimension
        posterior_samples = self.posterior_samples.copy()
        posterior_samples = posterior_samples.rename_dims(
            {'draw': 'something else'})
        with self.assertRaisesRegex(ValueError, 'The posterior samples must'):
            chi.compute_pointwise_loglikelihood(
                self.log_likelihood, posterior_samples)

        # Parameter map has the wrong type
        param_map = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The parameter map has'):
            chi.compute_pointwise_loglikelihood(
                self.log_likelihood, self.posterior_samples,
                param_map=param_map)

        # Not all parameters of the log-likelihood can be identified
        param_map = {'myokit.tumour_volume': 'Something else'}
        with self.assertRaisesRegex(ValueError, 'The parameter <Something'):
            chi.compute_pointwise_loglikelihood(
                self.log_likelihood, self.posterior_samples,
                param_map=param_map)

        # The individual is not in the posterior samples
        individual = 'Does not exist'
        with self.assertRaisesRegex(ValueError, 'The individual <Does not'):
            chi.compute_pointwise_loglikelihood(
                self.log_likelihood, self.posterior_samples,
                individual=individual)

        # Select individual with hierarchical model
        individual = 'Some ID'
        with self.assertRaisesRegex(ValueError, "Individual IDs cannot be"):
            chi.compute_pointwise_loglikelihood(
                self.hierarch_log_likelihood, self.hierarch_posterior_samples,
                individual=individual)

        # Posterior does not have all IDs of the hierarchical model
        n_chains = 2
        n_draws = 3
        n_ids = 2
        bottom_samples = np.ones(shape=(n_draws, n_chains, n_ids))
        top_samples = np.ones(shape=(n_draws, n_chains))
        bottom_samples = xr.DataArray(
            data=bottom_samples,
            dims=['draw', 'chain', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['Wrong', 'IDs']})
        top_samples = xr.DataArray(
            data=top_samples,
            dims=['draw', 'chain'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        parameter_names = self.hierarch_log_likelihood.get_parameter_names()
        hierarch_posterior_samples = xr.Dataset({
            parameter_names[0]: top_samples,
            parameter_names[1]: bottom_samples,
            parameter_names[3]: top_samples,
            parameter_names[4]: top_samples,
            parameter_names[5]: top_samples,
            parameter_names[6]: bottom_samples,
            parameter_names[8]: top_samples,
            parameter_names[9]: top_samples,
            parameter_names[10]: top_samples})

        with self.assertRaisesRegex(ValueError, "The ID <ID 40> does not"):
            chi.compute_pointwise_loglikelihood(
                self.hierarch_log_likelihood, hierarch_posterior_samples)


class TestInferenceController(unittest.TestCase):
    """
    Tests the chi.InferenceController base class.
    """

    @classmethod
    def setUpClass(cls):
        # Get test data and model
        data = DataLibrary().lung_cancer_control_group()
        individual = 40
        mask = data['ID'] == individual  # Arbitrary test id
        data = data[mask]
        mask = data['Biomarker'] == 'Tumour volume'  # Arbitrary biomarker
        times = data[mask]['Time'].to_numpy()
        observed_volumes = data[mask]['Measurement'].to_numpy()

        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()
        cls.log_likelihood = chi.LogLikelihood(
            mechanistic_model, error_model, observed_volumes, times)
        cls.log_likelihood.set_id(individual)

        # Create posterior
        log_prior_tumour_volume = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_drug_conc = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_kappa = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_lambda_0 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_lambda_1 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_sigma_base = pints.HalfCauchyLogPrior(location=0, scale=3)
        log_prior_sigma_rel = pints.HalfCauchyLogPrior(location=0, scale=3)
        cls.log_prior = pints.ComposedLogPrior(
            log_prior_tumour_volume,
            log_prior_drug_conc,
            log_prior_kappa,
            log_prior_lambda_0,
            log_prior_lambda_1,
            log_prior_sigma_base,
            log_prior_sigma_rel)
        log_posterior = chi.LogPosterior(cls.log_likelihood, cls.log_prior)

        # Set up optmisation controller
        cls.controller = chi.InferenceController(log_posterior)

        cls.n_ids = 1
        cls.n_params = 7

    def test_call_bad_input(self):
        # Wrong type of log-posterior
        log_posterior = 'bad log-posterior'

        with self.assertRaisesRegex(ValueError, 'The log-posterior has to be'):
            chi.InferenceController(log_posterior)

        # Log-posteriors don't have the same number of parameters
        log_posterior_1 = chi.LogPosterior(
            self.log_likelihood, self.log_prior)
        log_posterior_2 = chi.LogPosterior(
            self.log_likelihood, self.log_prior)
        log_posterior_2._n_parameters = 20

        with self.assertRaisesRegex(ValueError, 'All log-posteriors have to'):
            chi.InferenceController([log_posterior_1, log_posterior_2])

    def test_set_n_runs(self):
        n_runs = 5
        self.controller.set_n_runs(n_runs)

        self.assertEqual(self.controller._n_runs, n_runs)
        self.assertEqual(
            self.controller._initial_params.shape,
            (self.n_ids, n_runs, self.n_params))

    def test_parallel_evaluation(self):
        # Set to sequential
        self.controller.set_parallel_evaluation(False)
        self.assertFalse(self.controller._parallel_evaluation)

        # Set to parallel
        self.controller.set_parallel_evaluation(True)
        self.assertTrue(self.controller._parallel_evaluation)

    def test_parallel_evaluation_bad_input(self):
        # Non-boolean and non-integer
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` has'):
            self.controller.set_parallel_evaluation(2.2)

        # Negative input
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` cannot'):
            self.controller.set_parallel_evaluation(-2)

    def test_set_transform(self):
        # Apply transform
        transform = pints.LogTransformation(n_parameters=7)
        self.controller.set_transform(transform)

        self.assertEqual(self.controller._transform, transform)

    def test_set_transform_bad_transform(self):
        # Try to set transformation that is not a `pints.Transformation`
        transform = 'bad transform'

        with self.assertRaisesRegex(ValueError, 'Transform has to be an'):
            self.controller.set_transform(transform)

        # Try to set transformation with the wrong dimension
        transform = pints.LogTransformation(n_parameters=10)

        with self.assertRaisesRegex(ValueError, 'The dimensionality of the'):
            self.controller.set_transform(transform)


class TestOptimisationController(unittest.TestCase):
    """
    Tests the chi.OptimisationController class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up test problems
        # Model I: Individual with ID 40
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        model = chi.PharmacodynamicModel(path)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        cls.problem = chi.ProblemModellingController(model, error_models)

        data = DataLibrary().lung_cancer_control_group()
        cls.problem.set_data(data, {'myokit.tumour_volume': 'Tumour volume'})

        n_parameters = 7
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        cls.problem.set_log_prior(log_priors)
        cls.log_posterior_id_40 = cls.problem.get_log_posterior(
            individual='40')

        # Model II: Hierarchical model across all individuals
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel()]
        problem = copy.deepcopy(cls.problem)
        problem.set_population_model(pop_models)

        n_parameters = 1 + 1 + 8 + 1 + 1 + 1 + 2
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        cls.hierarchical_posterior = problem.get_log_posterior()

        # Get IDs for testing
        cls.ids = data['ID'].unique()

    def test_run(self):
        # Case I: Individual with ID 40
        optimiser = chi.OptimisationController(self.log_posterior_id_40)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        optimiser.set_parallel_evaluation(False)

        optimiser.set_n_runs(3)
        result = optimiser.run(n_max_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        ids = result['ID'].unique()
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 'ID 40')

        n_parameters = 7
        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), n_parameters)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')
        self.assertEqual(parameters[5], 'Sigma base')
        self.assertEqual(parameters[6], 'Sigma rel.')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

        # Case II: Hierarchical model
        optimiser = chi.OptimisationController(self.hierarchical_posterior)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        optimiser.set_parallel_evaluation(False)

        optimiser.set_n_runs(3)
        result = optimiser.run(n_max_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        # One ID for each individual and None for population parameters
        ids = result['ID'].unique()
        self.assertEqual(len(ids), 9)  # nids + None
        self.assertIsNone(ids[0])
        self.assertEqual(ids[1], 'ID ' + str(self.ids[0]))
        self.assertEqual(ids[2], 'ID ' + str(self.ids[1]))
        self.assertEqual(ids[3], 'ID ' + str(self.ids[2]))
        self.assertEqual(ids[4], 'ID ' + str(self.ids[3]))
        self.assertEqual(ids[5], 'ID ' + str(self.ids[4]))
        self.assertEqual(ids[6], 'ID ' + str(self.ids[5]))
        self.assertEqual(ids[7], 'ID ' + str(self.ids[6]))
        self.assertEqual(ids[8], 'ID ' + str(self.ids[7]))

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 9)
        self.assertEqual(parameters[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(parameters[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'Pooled myokit.lambda_0')
        self.assertEqual(parameters[4], 'Pooled myokit.lambda_1')
        self.assertEqual(parameters[5], 'Pooled Sigma base')
        self.assertEqual(parameters[6], 'Sigma rel.')
        self.assertEqual(parameters[7], 'Mean log Sigma rel.')
        self.assertEqual(parameters[8], 'Std. log Sigma rel.')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_run_catch_exception(self):
        # Check failure of optimisation doesn't interrupt all runs
        # (CMAES returns NAN for 1-dim problems)

        # Get test data and model
        problem = copy.deepcopy(self.problem)
        problem.fix_parameters({
            'myokit.drug_concentration': 1,
            'myokit.kappa': 1,
            'myokit.lambda_0': 1,
            'myokit.lambda_1': 1,
            'Sigma base': 1,
            'Sigma rel.': 1})
        problem.set_log_prior([pints.UniformLogPrior(1E-3, 1E1)])
        log_posteriors = problem.get_log_posterior()

        # Set up optmisation controller
        optimiser = chi.OptimisationController(log_posteriors[0])
        optimiser.set_n_runs(3)
        result = optimiser.run(n_max_iterations=10)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_optmiser(self):
        optimiser = chi.OptimisationController(self.log_posterior_id_40)
        optimiser.set_optimiser(pints.PSO)
        self.assertEqual(optimiser._optimiser, pints.PSO)

        optimiser.set_optimiser(pints.CMAES)
        self.assertEqual(optimiser._optimiser, pints.CMAES)

    def test_set_optimiser_bad_input(self):
        optimiser = chi.OptimisationController(self.log_posterior_id_40)
        with self.assertRaisesRegex(ValueError, 'Optimiser has to be'):
            optimiser.set_optimiser(str)


class TestSamplingController(unittest.TestCase):
    """
    Tests the chi.SamplingController class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up test problems
        # Model I: Individual with ID 40
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        model = chi.PharmacodynamicModel(path)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        problem = chi.ProblemModellingController(model, error_models)

        data = DataLibrary().lung_cancer_control_group()
        problem.set_data(data, {'myokit.tumour_volume': 'Tumour volume'})

        n_parameters = 7
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        cls.log_posterior_id_40 = problem.get_log_posterior(individual='40')

        # Model II: Hierarchical model across all individuals
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel()]
        problem.set_population_model(pop_models)

        n_parameters = 1 + 1 + 8 + 1 + 1 + 1 + 2
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        cls.hierarchical_posterior = problem.get_log_posterior()

        # Get IDs for testing
        cls.ids = data['ID'].unique()

    def test_run(self):
        # Case I: Individual with ID 40
        sampler = chi.SamplingController(self.log_posterior_id_40)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        sampler.set_parallel_evaluation(False)

        sampler.set_n_runs(3)
        n_parameters = self.log_posterior_id_40.n_parameters()
        sampler._initial_params = np.ones(shape=(1, 3, n_parameters))
        result = sampler.run(n_iterations=20)

        dimensions = list(result.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = result.individual
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 'ID 40')

        parameters = sorted(list(result.data_vars.keys()))
        self.assertEqual(len(parameters), 7)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')
        self.assertEqual(parameters[2], 'myokit.drug_concentration')
        self.assertEqual(parameters[3], 'myokit.kappa')
        self.assertEqual(parameters[4], 'myokit.lambda_0')
        self.assertEqual(parameters[5], 'myokit.lambda_1')
        self.assertEqual(parameters[6], 'myokit.tumour_volume')

        chains = result.chain
        self.assertEqual(len(chains), 3)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)
        self.assertEqual(chains.loc[2], 2)

        attrs = result.attrs
        divergent_iters = attrs['divergent iterations']
        self.assertEqual(divergent_iters, 'false')

        # Case II: Hierarchical model
        sampler = chi.SamplingController(self.hierarchical_posterior)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        sampler.set_parallel_evaluation(False)

        sampler.set_n_runs(3)
        n_parameters = self.hierarchical_posterior.n_parameters()
        sampler._initial_params = np.ones(shape=(1, 3, n_parameters))
        result = sampler.run(n_iterations=20)

        dimensions = list(result.dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = result.individual
        self.assertEqual(len(ids), 8)
        self.assertEqual(ids[0], 'ID 40')
        self.assertEqual(ids[1], 'ID 94')
        self.assertEqual(ids[2], 'ID 95')
        self.assertEqual(ids[3], 'ID 136')
        self.assertEqual(ids[4], 'ID 140')
        self.assertEqual(ids[5], 'ID 155')
        self.assertEqual(ids[6], 'ID 169')
        self.assertEqual(ids[7], 'ID 170')

        parameters = sorted(list(result.data_vars.keys()))
        self.assertEqual(len(parameters), 9)
        self.assertEqual(parameters[0], 'Mean log Sigma rel.')
        self.assertEqual(parameters[1], 'Pooled Sigma base')
        self.assertEqual(parameters[2], 'Pooled myokit.drug_concentration')
        self.assertEqual(parameters[3], 'Pooled myokit.lambda_0')
        self.assertEqual(parameters[4], 'Pooled myokit.lambda_1')
        self.assertEqual(parameters[5], 'Pooled myokit.tumour_volume')
        self.assertEqual(parameters[6], 'Sigma rel.')
        self.assertEqual(parameters[7], 'Std. log Sigma rel.')
        self.assertEqual(parameters[8], 'myokit.kappa')

        chains = result.chain
        self.assertEqual(len(chains), 3)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)
        self.assertEqual(chains.loc[2], 2)

        attrs = result.attrs
        divergent_iters = attrs['divergent iterations']
        self.assertEqual(divergent_iters, 'false')

        # Case III: Infer multiple independent models
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        model = chi.PharmacodynamicModel(path)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        problem = chi.ProblemModellingController(model, error_models)
        data = DataLibrary().lung_cancer_control_group()
        mask_id_40 = data['ID'] == 40
        mask_id_140 = data['ID'] == 140
        data = data[mask_id_40 | mask_id_140]
        problem.set_data(data, {'myokit.tumour_volume': 'Tumour volume'})
        n_parameters = 7
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        log_posterior = problem.get_log_posterior()
        sampler = chi.SamplingController(log_posterior)
        sampler.set_sampler(pints.HamiltonianMCMC)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        sampler.set_parallel_evaluation(False)

        sampler.set_n_runs(3)
        n_parameters = self.log_posterior_id_40.n_parameters()
        sampler._initial_params = np.ones(shape=(2, 3, n_parameters))
        hyperparameters = [1, 1]
        result = sampler.run(
            n_iterations=20, hyperparameters=hyperparameters)

        self.assertEqual(len(result), 2)

        dimensions = list(result[0].dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')
        dimensions = list(result[1].dims)
        self.assertEqual(len(dimensions), 3)
        self.assertEqual(dimensions[0], 'chain')
        self.assertEqual(dimensions[1], 'draw')
        self.assertEqual(dimensions[2], 'individual')

        ids = result[0].individual
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 'ID 40')
        ids = result[1].individual
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 'ID 140')

        parameters = sorted(list(result[0].data_vars.keys()))
        self.assertEqual(len(parameters), 7)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')
        self.assertEqual(parameters[2], 'myokit.drug_concentration')
        self.assertEqual(parameters[3], 'myokit.kappa')
        self.assertEqual(parameters[4], 'myokit.lambda_0')
        self.assertEqual(parameters[5], 'myokit.lambda_1')
        self.assertEqual(parameters[6], 'myokit.tumour_volume')
        parameters = sorted(list(result[1].data_vars.keys()))
        self.assertEqual(len(parameters), 7)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')
        self.assertEqual(parameters[2], 'myokit.drug_concentration')
        self.assertEqual(parameters[3], 'myokit.kappa')
        self.assertEqual(parameters[4], 'myokit.lambda_0')
        self.assertEqual(parameters[5], 'myokit.lambda_1')
        self.assertEqual(parameters[6], 'myokit.tumour_volume')

        chains = result[0].chain
        self.assertEqual(len(chains), 3)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)
        self.assertEqual(chains.loc[2], 2)
        chains = result[1].chain
        self.assertEqual(len(chains), 3)
        self.assertEqual(chains.loc[0], 0)
        self.assertEqual(chains.loc[1], 1)
        self.assertEqual(chains.loc[2], 2)

        attrs = result[0].attrs
        divergent_iters = attrs['divergent iterations']
        self.assertEqual(divergent_iters, 'true')
        attrs = result[1].attrs
        divergent_iters = attrs['divergent iterations']
        self.assertEqual(divergent_iters, 'true')

    def test_set_initial_parameters(self):
        # Test case I: Individual data
        n_runs = 10
        sampler = chi.SamplingController(self.log_posterior_id_40)
        sampler.set_n_runs(n_runs)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['myokit.kappa', 'myokit.lambda_1'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        data = pd.DataFrame({
            'ID': 'ID 40',
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Get initial values before setting them
        default_params = sampler._initial_params.copy()

        # Set initial values and test behaviour
        sampler.set_initial_parameters(data)
        new_params = sampler._initial_params

        n_ids = 1
        n_parameters = 7
        self.assertEqual(
            default_params.shape, (n_ids, n_runs, n_parameters))
        self.assertEqual(new_params.shape, (n_ids, n_runs, n_parameters))

        # Compare values. All but 3rd and 5th parameter should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(
            new_params[0, :, 0], default_params[0, :, 0]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 1], default_params[0, :, 1]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 2], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 3], default_params[0, :, 3]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 4], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 5], default_params[0, :, 5]))

        # Check that it works fine even if ID cannot be found
        data['ID'] = 'Some ID'
        sampler.set_initial_parameters(data)
        new_params = sampler._initial_params

        self.assertEqual(
            default_params.shape, (n_ids, n_runs, n_parameters))
        self.assertEqual(new_params.shape, (n_ids, n_runs, n_parameters))

        # Compare values. All but 3rd and 5th index should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(
            new_params[0, :, 0], default_params[0, :, 0]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 1], default_params[0, :, 1]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 2], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 3], default_params[0, :, 3]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 4], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 5], default_params[0, :, 5]))

        # Check that it works fine even if parameter cannot be found
        data['ID'] = 'ID 40'
        data['Parameter'] = ['SOME', 'PARAMETERS'] * 2
        sampler.set_initial_parameters(data)
        new_params = sampler._initial_params

        self.assertEqual(
            default_params.shape, (n_ids, n_runs, n_parameters))
        self.assertEqual(new_params.shape, (n_ids, n_runs, n_parameters))

        # Compare values. All but 3rd and 5th index should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(
            new_params[0, :, 0], default_params[0, :, 0]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 1], default_params[0, :, 1]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 2], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 3], default_params[0, :, 3]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 4], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 5], default_params[0, :, 5]))

    def test_set_initial_parameters_bad_input(self):
        sampler = chi.SamplingController(self.log_posterior_id_40)

        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            sampler.set_initial_parameters, data)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['myokit.lambda_0', 'noise param 1'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        test_data = pd.DataFrame({
            'ID': 'ID 40',
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Rename id key
        data = test_data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            sampler.set_initial_parameters, data)

        # Rename parameter key
        data = test_data.rename(columns={'Parameter': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Parameter>.',
            sampler.set_initial_parameters, data)

        # Rename estimate key
        data = test_data.rename(columns={'Estimate': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Estimate>.',
            sampler.set_initial_parameters, data)

        # Rename score key
        data = test_data.rename(columns={'Score': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Score>.',
            sampler.set_initial_parameters, data)

        # Rename run key
        data = test_data.rename(columns={'Run': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Run>.',
            sampler.set_initial_parameters, data)

    def test_set_sampler(self):
        sampler = chi.SamplingController(self.log_posterior_id_40)
        sampler.set_sampler(pints.HamiltonianMCMC)
        self.assertEqual(sampler._sampler, pints.HamiltonianMCMC)

        sampler.set_sampler(pints.HaarioACMC)
        self.assertEqual(sampler._sampler, pints.HaarioACMC)

    def test_set_sampler_bad_input(self):
        sampler = chi.SamplingController(self.log_posterior_id_40)
        with self.assertRaisesRegex(ValueError, 'Sampler has to be'):
            sampler.set_sampler(str)


if __name__ == '__main__':
    unittest.main()
