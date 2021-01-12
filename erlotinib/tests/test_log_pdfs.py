#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints
import pints.toy

import erlotinib as erlo


class TestHierarchicalLogLikelihood(unittest.TestCase):
    """
    Tests the erlotinib.HierarchicalLogLikelihood class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up toy problem
        data = erlo.DataLibrary().lung_cancer_control_group()
        mask = data['Biomarker'] == 'Tumour volume'  # Arbitrary biomarker
        data = data[mask]
        data = data[['ID', 'Time', 'Measurement']].rename(
            columns={'Measurement': 'Biomarker'})
        cls.n_ids = len(data['ID'].unique())
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)

        problem = erlo.ProblemModellingController(data)
        problem.set_mechanistic_model(model)
        problem.set_error_model([pints.GaussianLogLikelihood])

        # Instantiate HierarchicalLogLikelihood from likelihoods in problem
        cls.log_likelihoods = list(problem._log_likelihoods.values())
        cls.n_individual_params = cls.log_likelihoods[0].n_parameters()
        cls.population_models = [
            erlo.PooledModel()] * cls.n_individual_params
        cls.hierarchical_model = erlo.HierarchicalLogLikelihood(
            cls.log_likelihoods, cls.population_models)

    def test_bad_instantiation(self):
        # Log-likelihoods are not pints.LogPDF
        log_likelihoods = ['bad', 'type']
        with self.assertRaisesRegex(ValueError, 'The log-likelihoods have'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # Log-likelihoods are defined on different parameter spaces
        log_likelihoods = [pints.toy.AnnulusLogPDF(), pints.toy.ConeLogPDF(10)]
        with self.assertRaisesRegex(ValueError, 'All log-likelihoods have'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # Population models are not erlotinib.PopulationModel
        population_models = ['bad', 'type'] + ['match dimension'] * 4
        with self.assertRaisesRegex(ValueError, 'The population models have'):
            erlo.HierarchicalLogLikelihood(
                self.log_likelihoods, population_models)

        # Not all parameters of the likelihoods are assigned to a pop model
        population_models = [
            erlo.PooledModel(),
            erlo.PooledModel()]
        with self.assertRaisesRegex(ValueError, 'Wrong number of population'):
            erlo.HierarchicalLogLikelihood(
                self.log_likelihoods, population_models)

    def test_call(self):
        # Create reference model
        pooled_log_pdf = pints.PooledLogPDF(
            self.log_likelihoods, pooled=[True]*6)

        # Test case I.1
        parameters = [1, 1, 1, 1, 1, 1]
        score = pooled_log_pdf(parameters)

        self.assertEqual(self.hierarchical_model(parameters), score)

        # Test case I.2
        parameters = [10, 1, 0.1, 1, 3, 1]
        score = pooled_log_pdf(parameters)

        self.assertEqual(self.hierarchical_model(parameters), score)

        # Test case II.1: non-pooled model
        pop_models = [
            erlo.HeterogeneousModel()] \
            * self.n_individual_params
        likelihood = erlo.HierarchicalLogLikelihood(
            self.log_likelihoods, pop_models)

        # Compute score from individual likelihoods
        parameters = [1, 1, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        n_parameters = 6
        n_ids = 8
        parameters = [1] * n_parameters * n_ids
        self.assertEqual(likelihood(parameters), score)

        # Test case II.2
        # Compute score from individual likelihoods
        parameters = [10, 1, 0.1, 1, 3, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        n_ids = 8
        parameters = \
            [parameters[0]] * n_ids + \
            [parameters[1]] * n_ids + \
            [parameters[2]] * n_ids + \
            [parameters[3]] * n_ids + \
            [parameters[4]] * n_ids + \
            [parameters[5]] * n_ids
        self.assertEqual(likelihood(parameters), score)

        # Test case III.1: Non-trivial population model.
        pop_models = \
            [erlo.LogNormalModel()] + \
            [erlo.PooledModel()] \
            * (self.n_individual_params - 1)
        likelihood = erlo.HierarchicalLogLikelihood(
            self.log_likelihoods, pop_models)

        ref_likelihood_part_one = erlo.LogNormalModel()
        ref_likelihood_part_two = pints.PooledLogPDF(
            self.log_likelihoods, pooled=[False] + [True]*5)

        parameters = [10, 1, 0.1, 1, 3, 1]
        pop_params = [1, 1]

        n_ids = 8
        parameters = \
            [parameters[0]] * n_ids + \
            pop_params + \
            [parameters[1]] + \
            [parameters[2]] + \
            [parameters[3]] + \
            [parameters[4]] + \
            [parameters[5]]

        score = \
            ref_likelihood_part_one.compute_log_likelihood(
                pop_params, parameters[:n_ids]) + \
            ref_likelihood_part_two(parameters[:n_ids] + parameters[n_ids+2:])

        self.assertNotEqual(score, -np.inf)
        self.assertAlmostEqual(likelihood(parameters), score)

        # Test case III.2: Returns -np.inf if individuals are far away from
        # pop distribution
        parameters = [100000, 1, 0.1, 1, 3, 1]
        pop_params = [0, 0.00001]

        n_ids = 8
        parameters = \
            [parameters[0]] * n_ids + \
            pop_params + \
            [parameters[1]] + \
            [parameters[2]] + \
            [parameters[3]] + \
            [parameters[4]] + \
            [parameters[5]]

        self.assertEqual(likelihood(parameters), -np.inf)

    def test_get_log_likelihoods(self):
        log_likelihoods = self.hierarchical_model.get_log_likelihoods()

        self.assertEqual(len(log_likelihoods), 8)
        self.assertIsInstance(log_likelihoods[0], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[1], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[2], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[3], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[4], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[5], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[6], pints.LogPDF)
        self.assertIsInstance(log_likelihoods[7], pints.LogPDF)

    def test_get_population_models(self):
        population_models = self.hierarchical_model.get_population_models()

        self.assertEqual(len(population_models), 6)
        self.assertIsInstance(population_models[0], erlo.PopulationModel)
        self.assertIsInstance(population_models[1], erlo.PopulationModel)
        self.assertIsInstance(population_models[2], erlo.PopulationModel)
        self.assertIsInstance(population_models[3], erlo.PopulationModel)
        self.assertIsInstance(population_models[4], erlo.PopulationModel)
        self.assertIsInstance(population_models[5], erlo.PopulationModel)

    def test_n_parameters(self):
        n_parameters = self.log_likelihoods[0].n_parameters()
        self.assertEqual(
            self.hierarchical_model.n_parameters(), n_parameters)


class TestLogPosterior(unittest.TestCase):
    """
    Tests the erlotinib.LogPosterior class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test dataset
        times = [0, 1, 2, 3]
        values = [10, 11, 12, 13]

        # Create test model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem = erlo.InverseProblem(model, times, values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1))
        cls.log_posterior = erlo.LogPosterior(log_likelihood, log_prior)

    def test_set_id_bad_input(self):
        # Provide the wrong number of labels
        labels = ['wrong', 'number']

        with self.assertRaisesRegex(ValueError, 'If a list of IDs is'):
            self.log_posterior.set_id(labels)

    def test_set_get_id(self):
        # Check default
        self.assertIsNone(self.log_posterior.get_id())

        # Set some id
        index = 'Test id'
        self.log_posterior.set_id(index)

        self.assertEqual(self.log_posterior.get_id(), index)

        # Set an ID for each parameter individually
        labels = [
            'ID 1', 'ID 2', 'ID 3', 'ID 4', 'ID 5', 'ID 6']
        self.log_posterior.set_id(labels)

        self.assertEqual(self.log_posterior.get_id(), labels)

    def test_set_get_parameter_names(self):
        # Check default
        default = [
            'Param 1', 'Param 2', 'Param 3', 'Param 4', 'Param 5', 'Param 6']
        self.assertEqual(self.log_posterior.get_parameter_names(), default)

        # Set some parameter names
        names = ['A', 'B', 'C', 'D', 'E', 'F']
        self.log_posterior.set_parameter_names(names)

        self.assertEqual(self.log_posterior.get_parameter_names(), names)

    def test_set_parameter_names_bad_input(self):
        # Number of names does not match the number of parameters
        names = ['too', 'few', 'params']

        with self.assertRaisesRegex(ValueError, 'The list of param'):
            self.log_posterior.set_parameter_names(names)


class TestReducedLogPDF(unittest.TestCase):
    """
    Tests the erlotinib.ReducedLogPDF class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        times = [1, 2, 3, 4, 5]
        values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem = erlo.InverseProblem(model, times, values)
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)
        cls.mask = [True, False, False, True, False, True]
        cls.values = [11, 12, 13]
        cls.reduced_log_pdf = erlo.ReducedLogPDF(
            cls.log_likelihood, cls.mask, cls.values)

    def test_bad_input(self):
        # Wrong log-pdf
        log_pdf = 'Bad type'

        with self.assertRaisesRegex(ValueError, 'The log-pdf has to'):
            erlo.ReducedLogPDF(log_pdf, self.mask, self.values)

        # Mask is not as long as the number of parameyers
        mask = [True, True]

        with self.assertRaisesRegex(ValueError, 'Length of mask has to'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # Mask is not boolean
        mask = ['yes', 'no', 'yes', 'yes', 'yes', 'yes']

        with self.assertRaisesRegex(ValueError, 'Mask has to be a'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # There are not as many input values as fixed parameters
        values = [1]

        with self.assertRaisesRegex(ValueError, 'There have to be'):
            erlo.ReducedLogPDF(self.log_likelihood, self.mask, values)

    def test_call(self):
        parameters = np.array([11, 1, 1, 12, 1, 13])
        reduced_params = parameters[~np.array(self.mask)]

        self.assertEqual(
            self.reduced_log_pdf(reduced_params),
            self.log_likelihood(parameters))

    def test_n_parameters(self):
        before = self.log_likelihood.n_parameters()
        n_fixed = np.sum(self.mask)

        self.assertEqual(
            self.reduced_log_pdf.n_parameters(), before - n_fixed)


if __name__ == '__main__':
    unittest.main()
