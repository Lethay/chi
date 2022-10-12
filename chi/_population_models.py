#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import math
from warnings import warn

import numpy as np
import pints
from scipy.optimize import basinhopping
from scipy.special import erf
from scipy.stats import lognorm, multivariate_normal, norm, truncnorm, uniform

import chi

#TODO: in .compute_log_likelihood:
# for param_id, pop_model in enumerate(self._population_models):
#     #Check if this takes individual parameters to compare to pop params
#     if self._pop_model_is_KS[param_id]:
#         continue

#     # Get population and individual parameters
#     indiv_params = self._indiv_params[:, param_id]
#     pop_params  = self._pop_params[param_id]

#     # Add score
#     score += pop_model.compute_log_likelihood(
#         parameters=parameters[pop_params],
#         observations=parameters[indiv_params])
class PopulationModel(object):
    """
    A base class for population models.

    Population models can be multi-dimensional, but unless explicitly specfied
    in the model description, the dimensions of the model are modelled
    independently.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """
    def __init__(self, n_dim=1, dim_names=None):
        super(PopulationModel, self).__init__()
        if n_dim < 1:
            raise ValueError(
                'The dimension of the population model has to be greater or '
                'equal to 1.')
        self._n_dim = int(n_dim)
        self._n_hierarchical_dim = self._n_dim
        self._n_covariates = 0

        if dim_names:
            if len(dim_names) != self._n_dim:
                raise ValueError(
                    'The number of dimension names has to match the number of '
                    'dimensions of the population model.')
            dim_names = [str(name) for name in dim_names]
        else:
            dim_names = [
                'Dim. %d' % (id_dim + 1) for id_dim in range(self._n_dim)]
        self._dim_names = dim_names

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        raise NotImplementedError

    def compute_individual_parameters(self, parameters, eta, *args, **kwargs):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        return eta

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        raise NotImplementedError

    def compute_pointwise_ll(self, parameters, observations,  *args, **kwargs):
        """
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p, n_dim)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length (n, n_dim)
        """
        raise NotImplementedError

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None,  *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        raise NotImplementedError

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        return []

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return copy.copy(self._dim_names)

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        raise NotImplementedError

    def n_covariates(self):
        """
        Returns the number of covariates.
        """
        return self._n_covariates

    def n_dim(self):
        """
        Returns the dimensionality of the population model.
        """
        return self._n_dim

    def n_hierarchical_dim(self):
        """
        Returns the number of parameter dimensions whose samples are not
        deterministically defined by the population parameters.

        I.e. the number of dimensions minus the number of pooled and
        heterogeneously modelled dimensions.
        """
        return self._n_hierarchical_dim

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        raise NotImplementedError

    def n_ids(self):
        """
        Returns the number of modelled individuals.

        If the behaviour of the population model does not change with the
        number of modelled individuals 0 is returned.
        """
        return 0

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        raise NotImplementedError

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        raise NotImplementedError

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        raise NotImplementedError

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        raise NotImplementedError

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        If the model has no covariates, input is ignored.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List[str]
        """
        # Default is that models do not have covariates.
        return None

    def set_dim_names(self, names=None):
        """
        Sets the names of the population model dimensions.

        :param names: A list of dimension names. If ``None``, dimension names
            are reset to defaults.
        :type names: List[str], optional
        """
        if names is None:
            # Reset names to defaults
            self._dim_names = [
                'Dim. %d' % (id_dim + 1) for id_dim in range(self._n_dim)]
            return None

        if len(names) != self._n_dim:
            raise ValueError(
                'Length of names does not match the number of dimensions.')

        self._dim_names = [str(label) for label in names]

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        return None

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List[str]
        """
        raise NotImplementedError


class ComposedPopulationModel(PopulationModel):
    r"""
    A multi-dimensional population model composed of mutliple population
    models.

    A :class:`ComposedPopulationModel` assumes that its constituent population
    models are independent. The probability density function of the composed
    population model is therefore given by the product of the probability
    density functions.

    For constituent population models
    :math:`p(\psi _1 | \theta _1), \ldots , p(\psi _n | \theta _n)`, the
    probability density function of the composed population model is given by

    .. math::
        p(\psi _1, \ldots , \psi _n | \theta _1, \ldots , \theta _n) =
            \prod _{k=1}^n p(\psi _k | \theta _k) .

    Extends :class:`chi.PopulationModel`.

    :param population_models: A list of population models.
    :type population_models: List[chi.PopulationModel]
    """
    def __init__(self, population_models):
        super(ComposedPopulationModel, self).__init__()
        # Check inputs
        for pop_model in population_models:
            if not isinstance(pop_model, chi.PopulationModel):
                raise TypeError(
                    'The population models have to be instances of '
                    'chi.PopulationModel.')

        # Check that number of modelled individuals is compatible
        n_ids = 0
        for pop_model in population_models:
            if (n_ids > 0) and (pop_model.n_ids() > 0) and (
                    n_ids != pop_model.n_ids()):
                raise ValueError(
                    'All population models must model the same number of '
                    'individuals.')
            n_ids = n_ids if n_ids > 0 else pop_model.n_ids()
        self._population_models = population_models
        self._n_ids = n_ids

        # Get properties of population models
        n_dim = 0
        n_parameters = 0
        n_covariates = 0
        n_hierarchical_dim = 0
        for pop_model in self._population_models:
            n_covariates += pop_model.n_covariates()
            n_dim += pop_model.n_dim()
            n_hierarchical_dim += pop_model.n_hierarchical_dim()
            n_parameters += pop_model.n_parameters()

        self._n_dim = n_dim
        self._n_hierarchical_dim = n_hierarchical_dim
        self._n_parameters = n_parameters
        self._n_covariates = n_covariates

        # Make sure that models have unique parameter names
        # (if not enumerate dimensions to make them unique in most cases)
        names = self.get_parameter_names()
        if len(np.unique(names)) != len(names):
            dim_names = [
                'Dim. %d' % (dim_id + 1) for dim_id in range(self._n_dim)]
            self.set_dim_names(dim_names)

    def compute_individual_parameters(
            self, parameters, eta, covariates=None):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        If the population model does not use covariates, the covariate input
        is ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of the individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        eta = np.asarray(eta)
        parameters = np.asarray(parameters)

        # Compute individual parameters
        cov = None
        current_p = 0
        current_dim = 0
        current_cov = 0
        psis = np.empty(shape=eta.shape)
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            end_p = current_p + pop_model.n_parameters()
            end_dim = current_dim + pop_model.n_dim()
            psis[:, current_dim:end_dim] = \
                pop_model.compute_individual_parameters(
                    parameters=parameters[current_p:end_p],
                    eta=eta[:, current_dim:end_dim],
                    covariates=cov)
            current_p = end_p
            current_dim = end_dim

        return psis

    def compute_log_likelihood(
            self, parameters, observations, covariates=None):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of the individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: float
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        score = 0
        cov = None
        current_dim = 0
        current_param = 0
        current_cov = 0
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            score += pop_model.compute_log_likelihood(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim],
                covariates=cov
            )
            current_dim = end_dim
            current_param = end_param
            if np.isinf(score):
                return -np.inf

        return score

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p, n_dim)``
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape ``(n, n_dim)``
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length ``(n, n_dim)``
        """
        raise NotImplementedError

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, covariates=None):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        score = 0
        n_ids = len(observations)
        dpsi = np.zeros(shape=(n_ids, self._n_dim))
        dtheta = np.empty(shape=self._n_parameters)

        cov = None
        dlp_dpsi = None
        current_cov = 0
        current_dim = 0
        current_param = 0
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov
            # Get dlogp/dpsi
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            if dlogp_dpsi is not None:
                dlp_dpsi = dlogp_dpsi[:, current_dim:end_dim]

            s, dp, dt = pop_model.compute_sensitivities(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim],
                covariates=cov,
                dlogp_dpsi=dlp_dpsi)

            # Add score and sensitivities
            score += s
            dpsi[:, current_dim:end_dim] = dp
            dtheta[current_param:end_param] = dt

            current_dim = end_dim
            current_param = end_param

        return score, dpsi, dtheta

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_covariate_names()
        return names

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_dim_names()

        return names

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_parameter_names(exclude_dim_names)

        return names

    def get_population_models(self):
        """
        Returns the constituent population models.
        """
        return self._population_models

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_bottom, n_top = 0, 0
        for pop_model in self._population_models:
            n_b, n_t = pop_model.n_hierarchical_parameters(n_ids)
            n_bottom += n_b
            n_top += n_t

        return n_bottom, n_top

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(
            self, parameters, n_samples=None, seed=None, covariates=None,
            *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If the model does not depend on covariates the ``covariate`` input is
        ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Values of the model parameters.
        :type parameters: List, np.ndarray of shape (n_parameters,)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, np.random.Generator, optional
        :param covariates: Covariate values, specifying the sampled
            subpopulation.
        :type covariates: List, np.ndarray of shape ``(n_cov,)`` or
            ``(n_samples, n_cov)``, optional
        :returns: Samples from population model conditional on covariates.
        :rtype: np.ndarray of shape (n_samples, n_dim)
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if (self._n_covariates > 0):
            covariates = np.asarray(covariates)
            if covariates.ndim == 1:
                covariates = covariates[np.newaxis, :]
            if covariates.shape[1] != self._n_covariates:
                raise ValueError(
                    'Provided covariates do not match the number of '
                    'covariates.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)
        samples = np.empty(shape=sample_shape)

        # Transform seed to random number generator, so all models use the same
        # seed
        rng = np.random.default_rng(seed=seed)

        # Sample from constituent population models
        cov = None
        current_dim = 0
        current_param = 0
        current_cov = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()

            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            # Sample bottom-level parameters
            samples[:, current_dim:end_dim] = pop_model.sample(
                    parameters=parameters[current_param:end_param],
                    n_samples=n_samples,
                    seed=rng,
                    covariates=cov)
            current_dim = end_dim
            current_param = end_param

        return samples

    def set_dim_names(self, names=None):
        r"""
        Sets the names of the population model dimensions.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_dim`. If ``None``, dimension names are reset to
            defaults.
        """
        if names is None:
            # Reset dimension names
            for pop_model in self._population_models:
                pop_model.set_dim_names()
            return None

        if len(names) != self._n_dim:
            raise ValueError(
                'Length of names does not match the number of dimensions.')

        # Set dimension names
        names = [str(label) for label in names]
        current_dim = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            pop_model.set_dim_names(names[current_dim:end_dim])
            current_dim = end_dim

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        # Check cheap option first: Behaviour is not changed by input
        n_ids = int(n_ids)
        if (self._n_ids == 0) or (n_ids == self._n_ids):
            return None

        n_parameters = 0
        for pop_model in self._population_models:
            pop_model.set_n_ids(n_ids)
            n_parameters += pop_model.n_parameters()

        # Update n_ids and n_parameters
        self._n_ids = n_ids
        self._n_parameters = n_parameters

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset parameter names
            for pop_model in self._population_models:
                pop_model.set_parameter_names()
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        # Set parameter names
        names = [str(label) for label in names]
        current_param = 0
        for pop_model in self._population_models:
            end_param = current_param + pop_model.n_parameters()
            pop_model.set_parameter_names(names[current_param:end_param])
            current_param = end_param


class ComposedCorrelationPopulationModel(PopulationModel):
    r"""
    A multi-dimensional population model composed of mutliple population
    models.

    A :class:`ComposedCorrelationPopulationModel` assumes that its constituent population
    models are correlated.

    Extends :class:`chi.PopulationModel`.

    :param population_models: A list of population models.
    :type population_models: List[chi.PopulationModel]
    :param correlation_matrix: A 2D array describing the correlation between the random effects of each population model.
    :type correlation_matrix: np.ndarray
    """
    def __init__(self, population_models, correlation_matrix):
        super(ComposedCorrelationPopulationModel, self).__init__()
        # Check inputs
        for pop_model in population_models:
            if not isinstance(pop_model, chi.PopulationModel):
                raise TypeError(
                    'The population models have to be instances of '
                    'chi.PopulationModel.')
            if isinstance(pop_model, (
                ComposedPopulationModel, ComposedCorrelationPopulationModel, CovariatePopulationModel)):
                raise TypeError("The poplation models have to be single population models.")
            if hasattr(pop_model, "_centered") and not pop_model._centered:
                raise NotImplementedError("ComposedCorrelationPopulationModel doesn't support uncentered models.")
                #TODO: We could support this, and would then just generate random effects to give to modles directly,
                #instead of extracting CDFs
        self._correlation_matrix = np.array(correlation_matrix)
        if self._correlation_matrix.ndim!=2:
            raise TypeError("The correlation matrix must have two dimensions, not %d."%self._correlation_matrix.ndim)
        eigvals = np.linalg.eigvals(self._correlation_matrix)
        if not np.all(eigvals>=0) or not np.all(np.isreal(eigvals)):
            raise TypeError("The correlation matrix must be positive semi-definite (real non-neg eigenvalues).")

        # Check that number of modelled individuals is compatible
        n_ids = 0
        for pop_model in population_models:
            if (n_ids > 0) and (pop_model.n_ids() > 0) and (
                    n_ids != pop_model.n_ids()):
                raise ValueError(
                    'All population models must model the same number of '
                    'individuals.')
            n_ids = n_ids if n_ids > 0 else pop_model.n_ids()
        self._population_models = population_models
        self._n_ids = n_ids

        # Get properties of population models
        n_dim = 0
        n_parameters = 0
        n_covariates = 0
        n_hierarchical_dim = 0
        for pop_model in self._population_models:
            n_covariates += pop_model.n_covariates()
            n_dim += pop_model.n_dim()
            n_hierarchical_dim += pop_model.n_hierarchical_dim()
            n_parameters += pop_model.n_parameters()

        self._n_dim = n_dim
        self._n_hierarchical_dim = n_hierarchical_dim
        self._n_parameters = n_parameters
        self._n_covariates = n_covariates

        # Make sure that models have unique parameter names
        # (if not enumerate dimensions to make them unique in most cases)
        names = self.get_parameter_names()
        if len(np.unique(names)) != len(names):
            dim_names = [
                'Dim. %d' % (dim_id + 1) for dim_id in range(self._n_dim)]
            self.set_dim_names(dim_names)

        #Check that the length of the correlation matrix matches the number of dimensions
        if self._correlation_matrix.shape[0]!=self._n_dim or self._correlation_matrix.shape[1]!=self._n_dim:
            raise ValueError(
                f"The correlation matrix must have the same length as the number of dimensions. "
                f"n_dim: {n_dim}. Shape: {self._correlation_matrix.shape}.")

    def compute_individual_parameters(
            self, parameters, eta, covariates=None):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        If the population model does not use covariates, the covariate input
        is ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of the individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        warn(UserWarning("compute_individual_parameters is not tested."))
        eta = np.asarray(eta)
        parameters = np.asarray(parameters)

        # Compute individual parameters
        cov = None
        current_p = 0
        current_dim = 0
        current_cov = 0
        psis = np.empty(shape=eta.shape)
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            end_p = current_p + pop_model.n_parameters()
            end_dim = current_dim + pop_model.n_dim()
            psis[:, current_dim:end_dim] = \
                pop_model.compute_individual_parameters(
                    parameters=parameters[current_p:end_p],
                    eta=eta[:, current_dim:end_dim],
                    covariates=cov)
            current_p = end_p
            current_dim = end_dim

        return psis

    def compute_log_likelihood(
            self, parameters, observations, covariates=None):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of the individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: float
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        score = 0
        cdfs = np.zeros(observations.shape)
        cov = None
        current_dim = 0
        current_param = 0
        current_cov = 0
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            score += pop_model.compute_log_likelihood(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim],
                covariates=cov
            )
            if chi.is_heterogeneous_model(pop_model) or chi.is_pooled_model(pop_model):
                cdfs[:, current_dim:end_dim] = 0.5
            else:
                cdfs[:, current_dim:end_dim] = pop_model.compute_cdf(
                    parameters=parameters[current_param:end_param],
                    observations=observations[:, current_dim:end_dim]
                )
            current_dim = end_dim
            current_param = end_param
            if np.isinf(score):
                return -np.inf
        
        #Some CDFs will hit zero -- for example, a truncated normal with mean 1 and std.dev 1 will inevitable have samples that are approx 0.
        #0 CDFs will result in -infinity z-scores, however, so we need to impose a minimum.
        #The CDF of N(-5, 0, 1) is 2.86e-7. So, let's choose 1e-7 as the minimum/
        minCDF, maxCDF = 1e-7, 1-1e-7
        cdfs[cdfs<minCDF] = minCDF
        cdfs[cdfs>maxCDF] = maxCDF

        #With the CDFs from each population model, calculate the corresponding z-scores in the normal distribution using the inverse CDF function
        ppfs = [norm.ppf(c) for c in cdfs]
        if np.any(np.isinf(ppfs)):
            return -np.inf
        #With these scores, calculate the multivariate_normal likelihood
        multivariateLogLike = np.sum([
            multivariate_normal.logpdf(n, cov=self._correlation_matrix) for n in ppfs])
        #the underlying population distributions are not actually (necessarily) normally distributed, so find the contribution to this likelihood along the main diagonal, and remove it
        nonCorrelMultivariateLogLike = np.sum([
            multivariate_normal.logpdf(n) for n in ppfs])
        correlationContribution = multivariateLogLike - nonCorrelMultivariateLogLike

        score += correlationContribution
        return score

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p, n_dim)``
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape ``(n, n_dim)``
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length ``(n, n_dim)``
        """
        raise NotImplementedError

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, covariates=None):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        raise NotImplementedError

    def get_correlation_matrix(self):
        """
            Returns the matrix of correlations between bottom-level parameters.
        """
        return self._correlation_matrix

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_covariate_names()
        return names

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_dim_names()

        return names

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_parameter_names(exclude_dim_names)

        return names

    def get_population_models(self):
        """
        Returns the constituent population models.
        """
        return self._population_models

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_bottom, n_top = 0, 0
        for pop_model in self._population_models:
            n_b, n_t = pop_model.n_hierarchical_parameters(n_ids)
            n_bottom += n_b
            n_top += n_t

        return n_bottom, n_top

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(
            self, parameters, n_samples=None, seed=None, covariates=None,
            *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If the model does not depend on covariates the ``covariate`` input is
        ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Values of the model parameters.
        :type parameters: List, np.ndarray of shape (n_parameters,)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, np.random.Generator, optional
        :param covariates: Covariate values, specifying the sampled
            subpopulation.
        :type covariates: List, np.ndarray of shape ``(n_cov,)`` or
            ``(n_samples, n_cov)``, optional
        :returns: Samples from population model conditional on covariates.
        :rtype: np.ndarray of shape (n_samples, n_dim)
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if (self._n_covariates > 0):
            covariates = np.asarray(covariates)
            if covariates.ndim == 1:
                covariates = covariates[np.newaxis, :]
            if covariates.shape[1] != self._n_covariates:
                raise ValueError(
                    'Provided covariates do not match the number of '
                    'covariates.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)
        samples = np.empty(shape=sample_shape)

        # Transform seed to random number generator, so all models use the same
        # seed
        rng = np.random.default_rng(seed=seed)

        # Sample from the multivariate normal distribution
        normal_samples = np.random.multivariate_normal(
            np.zeros(self._n_dim), self._correlation_matrix, size=n_samples)

        # Use the normal CDF to find how far through a given PDF's support each parameter has covered
        CDF_samples = norm.cdf(normal_samples)

        # Sample from constituent population models
        cov = None
        current_dim = 0
        current_param = 0
        current_cov = 0
        for CDF_sample, pop_model in zip(np.transpose(CDF_samples), self._population_models):
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()

            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            # Sample bottom-level parameters
            if chi.is_heterogeneous_model(pop_model) or chi.is_pooled_model(pop_model):
                samples[:, current_dim:end_dim] = pop_model.sample(
                    parameters=parameters[current_param:end_param],
                    n_samples=n_samples,
                    seed=rng,
                    covariates=cov)
            else:
                samples[:, current_dim:end_dim] = pop_model.sample_from_cdf(
                    parameters=parameters[current_param:end_param],
                    cdf=CDF_sample,
                    covariates=cov)
            current_dim = end_dim
            current_param = end_param

        return samples

    def set_dim_names(self, names=None):
        r"""
        Sets the names of the population model dimensions.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_dim`. If ``None``, dimension names are reset to
            defaults.
        """
        if names is None:
            # Reset dimension names
            for pop_model in self._population_models:
                pop_model.set_dim_names()
            return None

        if len(names) != self._n_dim:
            raise ValueError(
                'Length of names does not match the number of dimensions.')

        # Set dimension names
        names = [str(label) for label in names]
        current_dim = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            pop_model.set_dim_names(names[current_dim:end_dim])
            current_dim = end_dim

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        # Check cheap option first: Behaviour is not changed by input
        n_ids = int(n_ids)
        if (self._n_ids == 0) or (n_ids == self._n_ids):
            return None

        n_parameters = 0
        for pop_model in self._population_models:
            pop_model.set_n_ids(n_ids)
            n_parameters += pop_model.n_parameters()

        # Update n_ids and n_parameters
        self._n_ids = n_ids
        self._n_parameters = n_parameters

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset parameter names
            for pop_model in self._population_models:
                pop_model.set_parameter_names()
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        # Set parameter names
        names = [str(label) for label in names]
        current_param = 0
        for pop_model in self._population_models:
            end_param = current_param + pop_model.n_parameters()
            pop_model.set_parameter_names(names[current_param:end_param])
            current_param = end_param


class CovariatePopulationModel(PopulationModel):
    r"""
    A population model that models the parameters across individuals
    conditional on covariates of the inter-individual variability.

    A covariate population model partitions a population into subpopulations
    which are characterised by covariates :math:`\chi`. The inter-individual
    variability within a subpopulation is modelled by a non-covariate
    population model

    .. math::
        p(\psi | \theta, \chi) = p(\psi | \vartheta (\theta, \chi)),

    where :math:`\vartheta` are the population parameters of the subpopulation
    which depend on global population parameters :math:`\theta` and the
    covariates :math:`\chi`.

    The ``population_model`` input defines the non-covariate population model
    for the subpopulations :math:`p(\psi | \vartheta )` and the
    ``covariate_model`` defines the relationship between the subpopulations and
    the covariates :math:`\vartheta (\theta, \chi)`.

    Extends :class:`PopulationModel`.

    :param population_model: Defines the distribution of the subpopulations.
    :type population_model: PopulationModel
    :param covariate_model: Defines the covariate model.
    :type covariate_model: CovariateModel
    :param dim_names: Name of dimensions.
    :type dim_names: List[str], optional
    """
    def __init__(self, population_model, covariate_model, dim_names=None):
        super(CovariatePopulationModel, self).__init__()
        # Check inputs
        if not isinstance(population_model, PopulationModel):
            raise TypeError(
                'The population model has to be an instance of a '
                'chi.PopulationModel.')
        if not isinstance(covariate_model, chi.CovariateModel):
            raise TypeError(
                'The covariate model has to be an instance of a '
                'chi.CovariateModel.')
        if chi.is_composed_population_model(population_model):
            raise TypeError(
                'The population model cannot be an instance of a '
                'chi.ComposedPopulationModel. Please compose multiple '
                'covariate models instead.')
        if isinstance(population_model, ReducedPopulationModel):
            raise TypeError(
                'The population model cannot be an instance of a '
                'chi.ReducedPopulationModel. Please define a covariate '
                'population model before fixing parameters.')

        # Remember models
        self._population_model = copy.deepcopy(population_model)
        self._covariate_model = copy.deepcopy(covariate_model)

        # Get properties
        self._n_dim = self._population_model.n_dim()
        self._n_pop = self._population_model.n_parameters()
        self._n_hierarchical_dim = self._population_model.n_hierarchical_dim()
        self._n_covariates = self._covariate_model.n_covariates()

        # Set names and all parameters to be modelled by the covariate model
        n_cov = self._covariate_model.n_covariates()
        self._population_model.set_dim_names(dim_names)
        indices = []
        for dim_id in range(self._n_dim):
            for param_id in range(self._n_pop // self._n_dim):
                indices.append([param_id, dim_id])
        self._covariate_model.set_population_parameters(indices)
        names = []
        for name in self._population_model.get_parameter_names():
            names += [name] * n_cov
        self._covariate_model.set_parameter_names(names)

    def compute_individual_parameters(self, parameters, eta, covariates):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        # Split into covariate model parameters and population parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        # TODO: Need to introduce a population model owned method that
        # transforms n_p to (n_p, n_d).
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute vartheta(theta, chi)
        parameters = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Compute psi(eta, vartheta)
        psi = self._population_model.compute_individual_parameters(
            parameters, eta)

        return psi

    def compute_log_likelihood(self, parameters, observations, covariates):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: float
        """
        # Split into covariate model parameters and population parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute vartheta(theta, chi)
        parameters = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations)

        return score

    # def compute_pointwise_ll(self, parameters, observations):
    #     r"""
    #     Returns the pointwise log-likelihood of the model parameters for
    #     each observation.

    #     :param parameters: Values of the model parameters :math:`\vartheta`.
    #     :type parameters: List, np.ndarray of length (p,)
    #     :param observations: "Observations" of the individuals :math:`\eta`.
    #         Typically refers to the inter-individual fluctuations of the
    #         mechanistic model parameter.
    #     :type observations: List, np.ndarray of length (n,)
    #     :returns: Log-likelihoods of individual parameters for population
    #         parameters.
    #     :rtype: np.ndarray of length (n,)
    #     """
    #     raise NotImplementedError
    #     # # Compute population parameters
    #     # parameters = self._covariate_model.compute_population_parameters(
    #     #     parameters)

    #     # # Compute log-likelihood
    #     # score = self._population_model.compute_pointwise_ll(
    #     #     parameters, observations)

    #     # return score

    def compute_sensitivities(
            self, parameters, observations, covariates, dlogp_dpsi=None):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        # Split into covariate model parameters and population parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute vartheta(theta, chi) and dvartheta/dtheta
        parameters = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Compute log-likelihood and sensitivities dscore/deta,
        # dscore/dvartheta
        score, dpsi, dvartheta = self._population_model.compute_sensitivities(
            parameters, observations, dlogp_dpsi=dlogp_dpsi,
            flattened=False)

        # Propagate sensitivities of score to population model parameters
        dpop, dcov = self._covariate_model.compute_sensitivities(
            cov_params, pop_params, covariates, dvartheta)
        dtheta = np.hstack([dpop, dcov])

        return score, dpsi, dtheta

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        return self._covariate_model.get_covariate_names()

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return self._population_model.get_dim_names()

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        names = self._population_model.get_parameter_names(exclude_dim_names)
        names += self._covariate_model.get_parameter_names()
        return names

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        # Get number of individual parameters
        n_ids, _ = self._population_model.n_hierarchical_parameters(n_ids)

        return (n_ids, self.n_parameters())

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        n_parameters = self._population_model.n_parameters()
        n_parameters += self._covariate_model.n_parameters()
        return n_parameters

    def sample(self, parameters, covariates, n_samples=None, seed=None):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        :param covariates: Covariate values, specifying the sampled
            subpopulation.
        :type covariates: List, np.ndarray of shape ``(n_cov,)`` or
            ``(n_samples, n_cov)``
        """
        covariates = np.array(covariates)
        if covariates.ndim == 1:
            covariates = covariates[np.newaxis, :]
        if covariates.shape[1] != self._n_covariates:
            raise ValueError(
                'Provided covariates do not match the number of covariates.')
        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)
        covariates = np.broadcast_to(
            covariates, (n_samples, self._n_covariates))

        # Split parameters into covariate model parameters and population model
        # parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute population parameters
        pop_params = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Sample parameters from population model
        seed = np.random.default_rng(seed)
        psi = np.empty(shape=(n_samples, self._n_dim))
        for ids, params in enumerate(pop_params):
            psi[ids] = self._population_model.sample(
                params, n_samples=1, seed=seed)[0]

        return psi

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        """
        self._covariate_model.set_covariate_names(names)

    def set_dim_names(self, names=None):
        """
        Sets the names of the population model dimensions.

        Setting the dimension names overwrites the parameter names of the
        covariate model.

        :param names: A list of dimension names. If ``None``, dimension names
            are reset to defaults.
        :type names: List[str], optional
        """
        self._population_model.set_dim_names(names)

        # Get names of parameters affected by the covariate model
        names = self._population_model.get_parameter_names()
        names = np.array(names).reshape(
            self._n_pop // self._n_dim, self._n_dim)
        pidx, didx = self._covariate_model.get_set_population_parameters()
        names = names[pidx, didx]

        n = []
        for name in names:
            n += [name] * self._n_covariates
        self._covariate_model.set_parameter_names(n)

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset parameter names
            self._population_model.set_parameter_names()
            self._covariate_model.set_parameter_names()
            return None

        self._population_model.set_parameter_names(names[:self._n_pop])
        self._covariate_model.set_parameter_names(names[self._n_pop:])

    def set_population_parameters(self, indices):
        """
        Sets the parameters of the population model that are transformed by the
        covariate model.

        Note that this influences the number of model parameters.

        :param indices: A list of parameter indices
            [param index, dim index].
        :type indices: List[Tuple[int, int]]
        """
        # Check that indices are in bounds
        indices = np.array(indices)
        upper = np.max(indices, axis=0)
        n_pop = self._n_pop // self._n_dim
        out_of_bounds = \
            (upper[0] >= n_pop) or (upper[1] >= self._n_dim) or \
            (np.min(indices) < 0)
        if out_of_bounds:
            raise IndexError('The provided indices are out of bounds.')
        self._covariate_model.set_population_parameters(indices)

        # Update parameter names
        names = np.array(self._population_model.get_parameter_names())
        names = names.reshape(n_pop, self._n_dim)[indices[:, 0], indices[:, 1]]
        n = []
        for name in names:
            n += [name] * self._n_covariates
        self._covariate_model.set_parameter_names(n)


class GaussianModel(PopulationModel):
    r"""
    A population model which models parameters across individuals
    with a Gaussian distribution.

    A Gaussian population model assumes that a model parameter
    :math:`\psi` varies across individuals such that :math:`\psi` is
    normally distributed in the population

    .. math::
        p(\psi |\mu, \sigma) =
        \frac{1}{\sqrt{2\pi} \sigma}
        \exp\left(-\frac{(\psi - \mu )^2}
        {2 \sigma ^2}\right).

    Here, :math:`\mu` and :math:`\sigma ^2` are the
    mean and variance of the Gaussian distribution.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    If ``centered = False`` the parametrisation is non-centered, i.e.

    .. math::
        \psi = \mu + \sigma \eta ,

    where :math:`\eta` models the inter-individual variability and is
    standard normally distributed.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    :param centered: Boolean flag indicating whether parametrisation is
        centered or non-centered.
    :type centered: bool, optional
    """
    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(GaussianModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = ['Mean'] * self._n_dim + ['Std.'] * self._n_dim

        self._centered = bool(centered)

    def _compute_dpsi(self, sigma, observations):
        """
        Computes the partial derivatives of psi = mu + sigma eta w.r.t.
        eta, mu and sigma.

        sigma: (n_ids, n_dim)
        observations: (n_ids, n_dim)

        rtype: np.ndarray of shape (n_ids, n_dim),
            np.ndarray of shape (n_ids, 2, n_dim)
        """
        n_ids, n_dim = observations.shape
        dpsi_deta = sigma
        dpsi_dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dpsi_dtheta[:, 0] = np.ones(shape=(n_ids, n_dim))
        dpsi_dtheta[:, 1] = observations
        return dpsi_deta, dpsi_dtheta

    @staticmethod
    def _compute_log_likelihood(mus, vars, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood.

        mus shape: (n_ids, n_dim)
        vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_likelihood = - np.sum(
                np.log(2 * np.pi * vars) / 2 + (observations - mus) ** 2
                / (2 * vars))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    def _compute_non_centered_sensitivities(
            self, sigmas, observations, dlogp_dpsi, flattened):
        """
        Returns the log-likelihood and the sensitivities with respect to
        eta and theta.
        """
        # Copmute score
        zeros = np.zeros(shape=(1, self._n_dim))
        ones = np.ones(shape=(1, self._n_dim))
        score = self._compute_log_likelihood(zeros, ones, observations)

        # Compute sensitivities
        if dlogp_dpsi is None:
            dlogp_dpsi = np.zeros((1, self._n_dim))
        deta = self._compute_sensitivities(zeros, ones, observations)
        dpsi_deta, dpsi_dtheta = self._compute_dpsi(sigmas, observations)
        dlogp_deta = dlogp_dpsi * dpsi_deta + deta
        dlogp_dtheta = dlogp_dpsi[:, np.newaxis, :] * dpsi_dtheta

        if not flattened:
            return score, dlogp_deta, dlogp_dtheta

        # Sum contributions across individuals and flatten
        dlogp_dtheta = np.sum(dlogp_dtheta, axis=0).flatten()

        return score, dlogp_deta, dlogp_dtheta

    @staticmethod
    def _compute_pointwise_ll(mean, var, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * var) / 2 \
            - (observations - mean) ** 2 / (2 * var)

        # If score evaluates to NaN, return -infinity
        mask = np.isnan(log_likelihood)
        if np.any(mask):
            log_likelihood[mask] = -np.inf
            return log_likelihood

        return log_likelihood

    def _compute_sensitivities(self, mus, vars, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mus shape: (n_ids, n_dim)
        vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)

        Returns:
        deta for non-centered of shape (n_ids, n_dim)

        and
        deta and dtheta for centered
        dtheta: np.ndarray of shape (n_ids, n_parameters, n_dim)
        """
        # Compute sensitivities w.r.t. observations (psi)
        with np.errstate(divide='ignore'):
            dpsi = (mus - psi) / vars
        if not self._centered:
            # Despite the naming, this is really deta
            return dpsi

        # Compute sensitivities w.r.t. parameters
        n_ids = len(psi)
        with np.errstate(divide='ignore'):
            dmus = (psi - mus) / vars
            dstd = (-1 + (psi - mus)**2 / vars) / np.sqrt(vars)

        # Collect sensitivities
        n_ids, n_dim = psi.shape
        dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dtheta[:, 0] = dmus
        dtheta[:, 1] = dstd

        return dpsi, dtheta

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Get parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        return norm.cdf(observations, loc=mus, scale=sigmas)

    def compute_individual_parameters(self, parameters, eta, *args, **kwargs):
        r"""
        Returns the individual parameters.

        If ``centered = True``, the model does not transform the parameters
        and ``eta`` is returned.

        If ``centered = False``, the individual parameters are defined as

        .. math::
            \psi = \mu + \sigma \eta,

        where :math:`\mu` and :math:`\sigma` are the model parameters and
        :math:`\eta` are the inter-individual fluctuations.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        eta = np.asarray(eta)
        if eta.ndim == 1:
            eta = eta[:, np.newaxis]
        if self._centered:
            return eta

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        mu = parameters[:, 0]
        sigma = parameters[:, 1]

        if np.any(sigma < 0):
            return np.full(shape=eta.shape, fill_value=np.nan)

        psi = mu + sigma * eta

        return psi

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        If ``centered = False``, the log-likelihood of the standard normal
        is returned. The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        if not self._centered:
            mus = np.zeros(shape=(1, self._n_dim))
            vars = np.ones(shape=(1, self._n_dim))
            score = self._compute_log_likelihood(mus, vars, observations)
            return score

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]
        vars = sigmas**2

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mus, vars, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivities to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]
        vars = sigmas**2

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution is strictly positive
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((len(observations), 2, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        if not self._centered:
            return self._compute_non_centered_sensitivities(
                sigmas, observations, dlogp_dpsi, flattened)

        # Compute for centered parametrisation
        score = self._compute_log_likelihood(mus, vars, observations)
        dpsi, dtheta = self._compute_sensitivities(mus, vars, observations)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        if not flattened:
            return score, dpsi, dtheta

        # Sum contributions across individuals and flatten
        dtheta = np.sum(dtheta, axis=0).flatten()
        return score, dpsi, dtheta

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        return (n_ids * self._n_dim, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        mean, std = norm.fit(sample)
        return (mean, std)

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Get parameters
        mus = parameters[0]
        sigmas = parameters[1]
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(
            loc=mus, scale=sigmas, size=sample_shape)

        return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Get parameters
        mus = parameters[0]
        sigmas = parameters[1]
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        return norm.ppf(cdf, loc=mus, scale=sigmas).reshape(sample_shape)

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a GaussianModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Mean'] * self._n_dim + ['Std.'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class GaussianModelRelativeSigma(GaussianModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution.

    The difference between this model and GaussianModel is that the standard
    deviation :math:`\sigma` is calculated as the product of the two input params,
    :math:`\mu` and :math:`\frac{\sigma}{\mu}`

    Extends :class:`GaussianModel`.
    """

    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(GaussianModelRelativeSigma, self).__init__()

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = ['Mean'] * self._n_dim + ['Rel. Std.'] * self._n_dim

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Get parameters
        means = parameters[:, 0]
        stdRatios = parameters[:, 1]
        stds = stdRatios*np.abs(means)

        if np.any(stdRatios < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        return norm.cdf(observations, loc=means, scale=stds)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        If ``centered = False``, the log-likelihood of the standard normal
        is returned. The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        if not self._centered:
            mus = np.zeros(shape=(1, self._n_dim))
            vars = np.ones(shape=(1, self._n_dim))
            score = self._compute_log_likelihood(mus, vars, observations)
            return score

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        means = parameters[:, 0]
        stdRatios = parameters[:, 1]
        vars = (stdRatios*means)**2

        if np.any(stdRatios < 0):
            # The std. of the Gaussian distribution is strictly positive
            return -np.inf

        return self._compute_log_likelihood(means, vars, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivities to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        stdRatios = parameters[:, 1]
        sigmas = mus*stdRatios
        vars = (mus*stdRatios)**2

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution is strictly positive
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((len(observations), 2, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        if not self._centered:
            return self._compute_non_centered_sensitivities(
                sigmas, observations, dlogp_dpsi, flattened)

        # Compute for centered parametrisation
        score = self._compute_log_likelihood(mus, vars, observations)
        dpsi, dtheta = self._compute_sensitivities(mus, vars, observations)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        if not flattened:
            return score, dpsi, dtheta

        # Sum contributions across individuals and flatten
        dtheta = np.sum(dtheta, axis=0).flatten()
        return score, dpsi, dtheta

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        mean, std = norm.fit(sample)
        stdRatio = std/mean
        return (mean, stdRatio)

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Get parameters
        mus = parameters[0]
        stdRatios = parameters[1]
        sigmas = stdRatios*np.abs(mus)
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(
            loc=mus, scale=sigmas, size=sample_shape)

        return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Get parameters
        mus = parameters[0]
        stdRatios = parameters[1]
        sigmas = stdRatios*np.abs(mus)
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        return norm.ppf(cdf, loc=mus, scale=sigmas).reshape(sample_shape)

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a GaussianModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Mean'] * self._n_dim + ['Rel. Std.'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class HeterogeneousModel(PopulationModel):
    """
    A population model which imposes no relationship on the model parameters
    across individuals.

    A heterogeneous model assumes that the parameters across individuals are
    independent.

    Extends :class:`PopulationModel`.

    .. note::
        Heterogeneous population models are special: the number of parameters
        depends on the number of modelled individuals.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    :param n_ids: Number of modelled individuals.
    :type n_ids: int, optional
    """
    def __init__(self, n_dim=1, dim_names=None, n_ids=1):
        super(HeterogeneousModel, self).__init__(n_dim, dim_names)
        self._n_ids = 0  # This is a temporary dummy value
        self._n_hierarchical_dim = 0
        self.set_n_ids(n_ids)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        r"""
        Returns the log-likelihood of the population model parameters.

        A heterogenous population model is equivalent to a
        multi-dimensional delta-distribution, where each bottom-level parameter
        is determined by a separate delta-distribution.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            # Heterogenous model is special, because n_param_per_dim = n_ids.
            # But after covariate transformation, the covariate information is
            # in the n_ids dimension.
            parameters = parameters[:, 0, :]

        # Return -inf if any of the observations do not equal the heterogenous
        # parameters
        mask = np.not_equal(observations, parameters)
        if np.any(mask):
            return -np.inf

        # Otherwise return 0
        return 0

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the parameters and the observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            # Heterogenous model is special, because n_param_per_dim = n_ids.
            # But after covariate transformation, the covariate information is
            # in the n_ids dimension.
            parameters = parameters[:, 0, :]

        # Return -inf if any of the observations does not equal the
        # heterogenous parameters
        n_ids = len(observations)
        mask = observations != parameters
        if np.any(mask):
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((n_ids, self._n_ids, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        # Otherwise return
        dpsi = np.zeros(observations.shape)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        dtheta = np.zeros(self._n_parameters)
        if not flattened:
            dtheta = np.zeros((n_ids, self._n_ids, self._n_dim))
        return 0, dpsi, dtheta

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        return (0, n_ids * self._n_dim)

    def n_ids(self):
        """
        Returns the number of modelled individuals.

        If the behaviour of the population model does not change with the
        number of modelled individuals 0 is returned.
        """
        return self._n_ids

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        For ``n_samples > 1`` the samples are randomly drawn from the ``n_ids``
        individuals.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(self._n_ids, self._n_dim)

        # Randomly sample from n_ids
        ids = np.arange(self._n_ids)
        rng = np.random.default_rng(seed=seed)
        n_samples = n_samples if n_samples else 1
        parameters = parameters[
            rng.choice(ids, size=n_samples, replace=True)]

        return parameters

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        if n_ids < 1:
            raise ValueError(
                'The number of modelled individuals needs to be greater or '
                'equal to 1.')

        if n_ids == self._n_ids:
            return None

        self._n_ids = n_ids
        self._n_parameters = self._n_ids * self._n_dim
        self._parameter_names = []
        for _id in range(self._n_ids):
            self._parameter_names += ['ID %d' % (_id + 1)] * self._n_dim

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = []
            for _id in range(self._n_ids):
                self._parameter_names += ['ID %d' % (_id + 1)] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class LogNormalModel(PopulationModel):
    r"""
    A population model which models parameters across individuals
    with a lognormal distribution.

    A lognormal population model assumes that a model parameter :math:`\psi`
    varies across individuals such that :math:`\psi` is log-normally
    distributed in the population

    .. math::
        p(\psi |\mu _{\text{log}}, \sigma _{\text{log}}) =
        \frac{1}{\psi} \frac{1}{\sqrt{2\pi} \sigma _{\text{log}}}
        \exp\left(-\frac{(\log \psi - \mu _{\text{log}})^2}
        {2 \sigma ^2_{\text{log}}}\right).

    Here, :math:`\mu _{\text{log}}` and :math:`\sigma ^2_{\text{log}}` are the
    mean and variance of :math:`\log \psi` in the population, respectively.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    If ``centered = False`` the parametrisation is non-centered, i.e.

    .. math::
        \log \psi = \mu _{\text{log}} + \sigma _{\text{log}} \eta ,

    where :math:`\eta` models the inter-individual variability and is
    standard normally distributed.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    :param centered: Boolean flag indicating whether parametrisation is
        centered or non-centered.
    :type centered: bool, optional
    """

    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(LogNormalModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = [
            'Log mean'] * self._n_dim + ['Log std.'] * self._n_dim

        self._centered = bool(centered)

    def _compute_dpsi(self, mu, sigma, etas):
        """
        Computes the partial derivatives of psi = exp(mu + sigma eta) w.r.t.
        eta, mu and sigma.

        mu: (n_ids, n_dim)
        sigma: (n_ids, n_dim)
        etas: (n_ids, n_dim)

        rtype: np.ndarray of shape (n_ids, n_dim),
            np.ndarray of shape (n_ids, 2, n_dim)
        """
        n_ids, n_dim = etas.shape
        psi = np.exp(mu + sigma * etas)
        dpsi_deta = sigma * psi
        dpsi_dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dpsi_dtheta[:, 0] = psi
        dpsi_dtheta[:, 1] = etas * psi
        return dpsi_deta, dpsi_dtheta

    @staticmethod
    def _compute_log_likelihood(log_mus, log_variance, observations):
        r"""
        Calculates the log-likelihood using.

        log_mus shape: (n_ids, n_dim)
        log_variance shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_likelihood = - np.sum(
                np.log(2 * np.pi * log_variance) / 2 + np.log(observations)
                + (np.log(observations) - log_mus)**2 / 2 / log_variance)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_non_centered_log_likelihood(observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood.

        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        log_likelihood = - np.sum(
            np.log(2 * np.pi) / 2 + observations ** 2 / 2)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    def _compute_non_centered_sensitivities(
            self, log_mus, log_sigmas, observations, dlogp_dpsi, flattened):
        """
        Returns the log-likelihood and the sensitivities with respect to
        eta and theta.
        """
        # Copmute score
        score = self._compute_non_centered_log_likelihood(observations)

        # Compute sensitivities
        if dlogp_dpsi is None:
            dlogp_dpsi = np.zeros((1, self._n_dim))
        deta = -observations
        dpsi_deta, dpsi_dtheta = self._compute_dpsi(log_mus, log_sigmas, observations)
        dlogp_deta = dlogp_dpsi * dpsi_deta + deta
        dlogp_dtheta = dlogp_dpsi[:, np.newaxis, :] * dpsi_dtheta

        if not flattened:
            return score, dlogp_deta, dlogp_dtheta

        # Sum contributions across individuals and flatten
        dlogp_dtheta = np.sum(dlogp_dtheta, axis=0).flatten()

        return score, dlogp_deta, dlogp_dtheta

    @staticmethod
    def _compute_pointwise_ll(log_mean, log_variance, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_psi = np.log(observations)
            log_likelihood = \
                - np.log(2 * np.pi * log_variance) / 2 \
                - log_psi \
                - (log_psi - log_mean) ** 2 / (2 * log_variance)

        # If score evaluates to NaN, return -infinity
        mask = np.isnan(log_likelihood)
        if np.any(mask):
            log_likelihood[mask] = -np.inf
            return log_likelihood

        return log_likelihood

    def _compute_sensitivities(self, log_mus, log_vars, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        log_mus shape: (n_ids, n_dim)
        log_vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)

        Returns:
        deta for non-centered of shape (n_ids, n_dim)

        and
        deta and dtheta for centered
        dtheta: np.ndarray of shape (n_ids, n_parameters, n_dim)
        """
        # Compute sensitivities
        n_ids = len(psi)
        with np.errstate(divide='ignore'):
            dpsi = - ((np.log(psi) - log_mus) / log_vars + 1) / psi
            dmus = (np.log(psi) - log_mus) / log_vars
            dstd = (-1 + (np.log(psi) - log_mus)**2 / log_vars) / np.sqrt(log_vars)

        # Collect sensitivities
        n_ids, n_dim = psi.shape
        dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dtheta[:, 0] = dmus
        dtheta[:, 1] = dstd

        return dpsi, dtheta

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        log_sigmas = parameters[:, 1]

        if np.any((mus<=0) | (log_sigmas<0)):
            # lognormal distributons cannot produce negative values, and
            # the scale of the lognormal distribution is strictly positive
            return -np.inf

        #np.random.lognorm takes the mean and sigma of the underlying normal dist. 
        #scipy.stats.lognorm takes scale=(the exponential of that mean), and s=(that sigma).
        #n.b.:
            #np.median(np.random.lognormal(np.log(100), 1, 10000))               -> 102
            #np.median(lognorm.rvs(scale=100, s=1, loc=0, size=10000))           -> 100
            #np.percentile(np.random.lognormal(np.log(100), 10, 100000), 75)     -> 8e4
            #np.percentile(lognorm.rvs(scale=100, s=10, loc=0, size=100000), 75) -> 9e4
            #lognorm.cdf(100, scale=100, s=1, loc=0)                             -> 0.5
        return lognorm.cdf(observations, scale=mus, s=log_sigmas, loc=0)

    def compute_individual_parameters(self, parameters, eta, *args, **kwargs):
        r"""
        Returns the individual parameters.

        If ``centered = True``, the model does not transform the parameters
        and ``eta`` is returned.

        If ``centered = False``, the individual parameters are computed using

        .. math::
            \psi = \mathrm{e}^{
                \mu _{\mathrm{log}} + \sigma _{\mathrm{log}} \eta},

        where :math:`\mu _{\mathrm{log}}` and :math:`\sigma _{\mathrm{log}}`
        are the model parameters and :math:`\eta` are the inter-individual
        fluctuations.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        eta = np.asarray(eta)
        if eta.ndim == 1:
            eta = eta[:, np.newaxis]
        if self._centered:
            return eta

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        #Get parameters
        mu = parameters[:, 0]
        log_sigma = parameters[:, 1]
        if np.any((mu<=0) | (log_sigma<0)):
            # lognormal distributons cannot produce negative values, and
            # The scale of the lognormal distribution is strictly positive
            return np.full(shape=eta.shape, fill_value=np.nan)

        #Transform to log parameters
        log_mu = np.log(mu)

        psi = np.exp(log_mu + log_sigma*eta)

        return psi

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        If ``centered = False``, the log-likelihood of the standard normal
        is returned. The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        if not self._centered:
            return self._compute_non_centered_log_likelihood(observations)

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any((mus<=0) | (sigmas<0)):
            # lognormal distributons cannot produce negative values, and
            # the scale of the lognormal distribution is strictly positive
            return -np.inf

        #Transform to log parameters
        log_mus = np.log(mus)
        log_vars = sigmas**2

        return self._compute_log_likelihood(log_mus, log_vars, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        log_sigmas = parameters[:, 1]

        if np.any((mus<=0) | (log_sigmas<0)):
            # lognormal distributons cannot produce negative values, and
            # the scale of the lognormal distribution is strictly positive
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((len(observations), 2, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        #Transform to log parameters
        log_mus = np.log(mus)
        log_vars = log_sigmas**2

        if not self._centered:
            return self._compute_non_centered_sensitivities(
                log_mus, log_sigmas, observations, dlogp_dpsi, flattened)

        # Compute for centered parametrisation
        score = self._compute_log_likelihood(log_mus, log_vars, observations)
        dpsi, dtheta = self._compute_sensitivities(log_mus, log_vars, observations)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        if not flattened:
            return (score, dpsi, dtheta)

        # Sum contributions across individuals and flatten
        dtheta = np.sum(dtheta, axis=0).flatten()
        return score, dpsi, dtheta

    def get_mean_and_std(self, parameters):
        r"""
        Returns the mean and the standard deviation of the population
        for given :math:`\mu _{\text{log}}` and :math:`\sigma _{\text{log}}`.

        The mean and variance of the parameter :math:`\psi`,
        :math:`\mu = \mathbb{E}\left[ \psi \right]` and
        :math:`\sigma ^2 = \text{Var}\left[ \psi \right]`, are given by

        .. math::
            \mu = \mathrm{e}^{\mu _{\text{log}} + \sigma ^2_{\text{log}} / 2}
            \quad \text{and} \quad
            \sigma ^2 =
            \mu ^2 \left( \mathrm{e}^{\sigma ^2_{\text{log}}} - 1\right) .

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        """
        # Check input
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        #Get parameters
        mus = parameters[0]
        log_sigmas = parameters[1]
        if np.any(mus <= 0):
            raise ValueError('The mean before logging cannot be negative.')
        if np.any(log_sigmas < 0):
            raise ValueError('The standard deviation cannot be negative.')

        #Transform to log parameters
        log_mus = np.log(mus)

        # Compute mean and standard deviation
        mean = np.exp(log_mus + log_sigmas**2 / 2)
        std = np.sqrt(
            np.exp(2 * log_mus + log_sigmas**2) * (np.exp(log_sigmas**2) - 1))

        return np.vstack([mean, std])

    def get_parameters_from_mean_and_std(self, mean, std):
        r"""
        Returns :math:`\mu _{\text{log}}` and :math:`\sigma _{\text{log}}` from the desired mean and standard deviation
        of the population. In other words, takes normal parameters and returns lognormal parameters. This is the corresponding function to get_mean_and_std.

        :param mean: mean of the population.
        :type mean: float
        :param std: standard deviation of the population.
        :type std: float
        """
        log_std  = np.sqrt(np.log((std/mean)**2 + 1))
        log_mean = np.exp(np.log(mean) - log_std**2 / 2)

        return np.vstack([log_mean, log_std])

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_ids = int(n_ids)

        return (n_ids * self._n_dim, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def reverse_sample(self, sample, fast=True):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        if fast:
            #Fixing floc makes the calculation analytical
            s, loc, scale = lognorm.fit(sample, floc=0)
        else:
            s, loc, scale = lognorm.fit(sample)

        """
        Suppose a normally distributed random variable ``X`` has  mean ``mu`` and
        standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
        distributed with ``shape = sigma`` and ``scale = exp(mu)``.

        For a log-normal distribution whose underyling normal has mean np.log(100) and standard deviation 1,
        normMean, normStd = np.log(100), 100/100

        lognorm.fit(np.random.lognormal(np.log(100), 1, 10000))          ->  (1.00518, -0.01436, 99.2 )
        lognorm.fit(np.random.lognormal(np.log(100), 1, 10000), floc=0)  ->  (1.0008,  0.0,      99.47)

        np.median(np.random.lognormal(np.log(100), 1, 10000))               -> 102
        np.median(lognorm.rvs(scale=100, s=1, loc=0, size=10000))           -> 100
        np.percentile(np.random.lognormal(np.log(100), 10, 100000), 75)     -> 8e4
        np.percentile(lognorm.rvs(scale=100, s=10, loc=0, size=100000), 75) -> 9e4

        The scale is exp(normally distributed mean).
        s         is the normally distributed sigma.
        """
        #Get log parameters
        mean  = scale  #and log_mean = np.log(mean)
        log_sigma = s

        return mean, log_sigma

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Instantiate random number generator
        rng = np.random.default_rng(seed=seed)

        # Get parameters
        mus = parameters[0]
        log_sigmas = parameters[1]

        if np.any(mus <= 0):
            raise ValueError(
                'A log-normal distribution only accepts the log of strictly positive '
                'means.')
        if np.any(log_sigmas <= 0):
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)
            return rng.normal(loc=mus, scale=sigmas, size=sample_shape)
        else:
            # Sample from population distribution
            # (Mean and sigma are the mean and standard deviation of
            # the log samples)
            log_mus = np.log(mus)
            samples = rng.lognormal(
                mean=log_mus, sigma=log_sigmas, size=sample_shape)
            return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Get parameters
        mus = parameters[0]
        log_sigmas = parameters[1]

        #Error checking
        if np.any(mus <= 0):
            raise ValueError(
                'A log-normal distribution only accepts the log of strictly positive '
                'means.')
        if np.any(log_sigmas <= 0):
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        # Sample from population distribution
        if not self._centered:
            mus = np.zeros(mus.shape)
            log_sigmas = np.ones(log_sigmas.shape)
            return norm.ppf(cdf, loc=mus, scale=log_sigmas)
        else:
            #np.random.lognorm takes the mean and sigma of the underlying normal dist. 
            #scipy.stats.lognorm takes scale=(the exponential of that mean), and s=(that sigma).
            #n.b.:
                #np.median(np.random.lognormal(np.log(100), 1, 10000))               -> 102
                #np.median(lognorm.rvs(scale=100, s=1, loc=0, size=10000))           -> 100
                #np.percentile(np.random.lognormal(np.log(100), 10, 100000), 75)     -> 8e4
                #np.percentile(lognorm.rvs(scale=100, s=10, loc=0, size=100000), 75) -> 9e4
                #lognorm.ppf(0.5, scale=100, s=1, loc=0)                             -> 100
            return lognorm.ppf(cdf, scale=mus, s=log_sigmas, loc=0).reshape(sample_shape)

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Log mean'] * self._n_dim + ['Log std.'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class LogNormalModelRelativeSigma(LogNormalModel):
    r"""
    A population model which assumes that model parameters across individuals
    are log-normally distributed.

    The difference between this model and LogNormalModel is that the standard
    deviation of the log :math:`\sigma` is calculated as the product of the two input params,
    :math:`\mu` and :math:`\frac{\sigma}{\mu}`

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(LogNormalModelRelativeSigma, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mean log', 'Rel. Std. log']

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        stdRatios = parameters[:, 1]
        log_sigmas = stdRatios*np.abs(mus)

        if any ((mus<=0) | (stdRatios<0)):
            # lognormal distributons cannot produce negative values, and
            # the scale of the lognormal distribution is strictly positive
            return -np.inf

        #Transform to log parameters

        #np.random.lognorm takes the mean and sigma of the underlying normal dist. 
        #scipy.stats.lognorm takes scale=(the exponential of that mean), and s=(that sigma).
        #n.b.:
            #np.median(np.random.lognormal(np.log(100), 1, 10000))               -> 102
            #np.median(lognorm.rvs(scale=100, s=1, loc=0, size=10000))           -> 100
            #np.percentile(np.random.lognormal(np.log(100), 10, 100000), 75)     -> 8e4
            #np.percentile(lognorm.rvs(scale=100, s=10, loc=0, size=100000), 75) -> 9e4
            #lognorm.cdf(100, scale=100, s=1, loc=0)                             -> 0.5
        return lognorm.cdf(observations, scale=mus, s=log_sigmas, loc=0)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        If ``centered = False``, the log-likelihood of the standard normal
        is returned. The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        if not self._centered:
            return self._compute_non_centered_log_likelihood(observations)

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        stdRatios = parameters[:, 1]

        if any ((mus<=0) | (stdRatios<0)):
            # lognormal distributons cannot produce negative values, and
            # the scale of the lognormal distribution is strictly positive
            return -np.inf

        #Transform to log parameters
        log_mus = np.log(mus)
        log_sigmas = stdRatios * np.abs(mus)
        log_vars = log_sigmas**2

        return self._compute_log_likelihood(log_mus, log_vars, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        stdRatios = parameters[:, 1]

        if np.any((mus<=0 ) | (stdRatios<0)):
            # lognormal distributons cannot produce negative values, and
            # the scale of the lognormal distribution is strictly positive
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((len(observations), 2, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        #Transform to log parameters
        log_mus = np.log(mus)
        log_sigmas = stdRatios*np.abs(mus)
        log_vars = log_sigmas**2

        if not self._centered:
            return self._compute_non_centered_sensitivities(
                log_mus, log_sigmas, observations, dlogp_dpsi, flattened)

        # Compute for centered parametrisation
        score = self._compute_log_likelihood(log_mus, log_vars, observations)
        dpsi, dtheta = self._compute_sensitivities(log_mus, log_vars, observations)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        if not flattened:
            return (score, dpsi, dtheta)

        # Sum contributions across individuals and flatten
        dtheta = np.sum(dtheta, axis=0).flatten()
        return score, dpsi, dtheta

    def get_mean_and_std(self, parameters):
        r"""
        Returns the mean and the standard deviation of the population
        for given :math:`\mu _{\text{log}}` and :math:`\sigma _{\text{log}}`.

        The mean and variance of the parameter :math:`\psi`,
        :math:`\mu = \mathbb{E}\left[ \psi \right]` and
        :math:`\sigma ^2 = \text{Var}\left[ \psi \right]`, are given by

        .. math::
            \mu = \mathrm{e}^{\mu _{\text{log}} + \sigma ^2_{\text{log}} / 2}
            \quad \text{and} \quad
            \sigma ^2 =
            \mu ^2 \left( \mathrm{e}^{\sigma ^2_{\text{log}}} - 1\right) .

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        """
        # Check input
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        #Get parameters
        mus = parameters[0]
        stdRatios = parameters[1]
        log_sigmas = stdRatios*np.abs(mus)
        if np.any(stdRatios < 0):
            raise ValueError('The standard deviation cannot be negative.')

        #Transform to log parameters
        log_mus = np.log(mus)

        # Compute mean and standard deviation
        mean = np.exp(log_mus + log_sigmas**2 / 2)
        std = np.sqrt(
            np.exp(2 * log_mus + log_sigmas**2) * (np.exp(log_sigmas**2) - 1))

        return np.vstack([mean, std])

    def get_parameters_from_mean_and_std(self, mean, std):
        r"""
        Returns :math:`\mu _{\text{log}}` and :math:`\sigma _{\text{log}}` from the desired mean and standard deviation
        of the population. In other words, takes normal parameters and returns lognormal parameters. This is the corresponding function to get_mean_and_std.

        :param mean: mean of the population.
        :type mean: float
        :param std: standard deviation of the population.
        :type std: float
        """
        log_std  = np.sqrt(np.log((std/mean)**2 + 1))
        log_mean = np.exp(np.log(mean) - log_std**2 / 2)
        stdRatio = log_std / log_mean

        return np.vstack([log_mean, stdRatio])

    def reverse_sample(self, sample, fast=True):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        if fast:
            #Fixing floc makes the calculation analytical
            s, loc, scale = lognorm.fit(sample, floc=0)
        else:
            s, loc, scale = lognorm.fit(sample)

        """
        Suppose a normally distributed random variable ``X`` has  mean ``mu`` and
        standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
        distributed with ``shape = sigma`` and ``scale = exp(mu)``.

        For a log-normal distribution whose underyling normal has mean np.log(100) and standard deviation 1,
        normMean, normStd = np.log(100), 100/100

        lognorm.fit(np.random.lognormal(np.log(100), 1, 10000))          ->  (1.00518, -0.01436, 99.2 )
        lognorm.fit(np.random.lognormal(np.log(100), 1, 10000), floc=0)  ->  (1.0008,  0.0,      99.47)

        np.median(np.random.lognormal(np.log(100), 1, 10000))               -> 102
        np.median(lognorm.rvs(scale=100, s=1, loc=0, size=10000))           -> 100
        np.percentile(np.random.lognormal(np.log(100), 10, 100000), 75)     -> 8e4
        np.percentile(lognorm.rvs(scale=100, s=10, loc=0, size=100000), 75) -> 9e4

        The scale is exp(normally distributed mean).
        s         is the normally distributed sigma.
        """
        #Get log parameters
        mean  = scale  #and log_mean = np.log(mean)
        log_sigma = s
        stdRatio = log_sigma/np.abs(mean)

        return mean, stdRatio

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Instantiate random number generator
        rng = np.random.default_rng(seed=seed)

        # Get parameters
        mus = parameters[0]
        logStdRatios = parameters[1]

        if any(mus <= 0):
            raise ValueError(
                'A log-normal distribution only accepts the log of strictly positive '
                'means.')
        if any(logStdRatios <= 0):
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        #Non-centered sample
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)
            return rng.normal(loc=mus, scale=sigmas, size=sample_shape)
        else:
            #Transform to log parameters
            log_mus = np.log(mus)
            log_sigmas = logStdRatios*np.abs(mus)

            # Sample from population distribution
            # (log_mean and sigma are the mean and standard deviation of
            # the log samples)
            samples = rng.lognormal(
                mean=log_mus, sigma=log_sigmas, size=sample_shape) #this'll return entries that look like e^log_mean

            return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Get parameters
        mus = parameters[0]
        logStdRatios = parameters[1]

        # Error checking
        if any(mus <= 0):
            raise ValueError(
                'A log-normal distribution only accepts the log of strictly positive '
                'means.')
        if any(logStdRatios <= 0):
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)
            return norm.ppf(cdf, loc=mus, scale=sigmas)
        else:
            #np.random.lognorm takes the mean and sigma of the underlying normal dist. 
            #scipy.stats.lognorm takes scale=(the exponential of that mean), and s=(that sigma).
            #n.b.:
                #np.median(np.random.lognormal(np.log(100), 1, 10000))               -> 102
                #np.median(lognorm.rvs(scale=100, s=1, loc=0, size=10000))           -> 100
                #np.percentile(np.random.lognormal(np.log(100), 10, 100000), 75)     -> 8e4
                #np.percentile(lognorm.rvs(scale=100, s=10, loc=0, size=100000), 75) -> 9e4
                #lognorm.ppf(0.5, scale=100, s=1, loc=0)                             -> 100
            log_sigmas = logStdRatios * np.abs(mus)
            return lognorm.ppf(cdf, scale=mus, s=log_sigmas, loc=0).reshape(sample_shape)

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Log mean'] * self._n_dim + ['Rel. Log std.'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class PooledModel(PopulationModel):
    """
    A population model which pools the model parameters across individuals.

    A pooled model assumes that the parameters across individuals do not vary.
    As a result, all individual parameters are set to the same value.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """

    def __init__(self, n_dim=1, dim_names=None):
        super(PooledModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = self._n_dim

        # Set number of hierarchical dimensions
        self._n_hierarchical_dim = 0

        # Set default parameter names
        self._parameter_names = ['Pooled'] * self._n_dim

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            parameters = parameters[:, 0]

        # Return -inf if any of the observations do not equal the pooled
        # parameter
        mask = observations != parameters
        if np.any(mask):
            return -np.inf

        # Otherwise return 0
        return 0

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        A pooled population model is a delta-distribution centred at the
        population model parameter. As a result the log-likelihood score
        is 0, if all individual parameters are equal to the population
        parameter, and :math:`-\infty` otherwise.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual, i.e. [:math:`\psi _1, \ldots , \psi _N`].
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length (n, n_dim)
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(1, self._n_dim)

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        log_likelihood = np.zeros_like(observations, dtype=float)
        mask = observations != parameters
        log_likelihood[mask] = -np.inf

        return log_likelihood

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            parameters = parameters[:, 0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        mask = observations != parameters
        if np.any(mask):
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((len(observations), 1, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        # Otherwise return
        dpsi = np.zeros(observations.shape)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        dtheta = np.zeros(self._n_parameters)
        if not flattened:
            dtheta = np.zeros((len(observations), 1, self._n_dim))
        return 0, dpsi, dtheta

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        return (0, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        return np.asarray(np.mean(sample))

    def sample(self, parameters, n_samples=None, *args, **kwargs):
        r"""
        Returns random samples from the underlying population
        distribution.

        For a PooledModel the input top-level parameters are copied
        ``n_samples`` times and are returned.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # If only one sample is requested, return input parameters
        if n_samples is None:
            return parameters

        # If more samples are wanted, broadcast input parameter to shape
        # (n_samples, n_dim)
        sample_shape = (int(n_samples), self._n_dim)
        samples = np.broadcast_to(parameters, shape=sample_shape)
        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Pooled'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class ReducedPopulationModel(PopulationModel):
    """
    A class that can be used to permanently fix model parameters of a
    :class:`PopulationModel` instance.

    This may be useful to explore simplified versions of a model.

    Extends :class:`chi.PopulationModel`.

    :param population_model: A population model.
    :type population_model: chi.PopulationModel
    """

    def __init__(self, population_model):
        super(ReducedPopulationModel, self).__init__()

        # Check inputs
        if not isinstance(population_model, PopulationModel):
            raise TypeError(
                'The population model has to be an instance of a '
                'chi.PopulationModel.')

        self._population_model = population_model

        # Set defaults
        self._fixed_params_mask = None
        self._fixed_params_values = None
        self._n_parameters = population_model.n_parameters()
        self._n_dim = population_model.n_dim()
        self._n_covariates = population_model.n_covariates()
        self._n_hierarchical_dim = population_model.n_hierarchical_dim()

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood
        cdf = self._population_model.compute_cdf(
            parameters, observations, *args, **kwargs)

        return cdf
        
    def compute_individual_parameters(self, parameters, eta, *args, **kwargs):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        return self._population_model.compute_individual_parameters(
            parameters, eta, *args, **kwargs)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        parameters = np.asarray(parameters).flatten()
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations, *args, **kwargs)

        return score

    # def compute_pointwise_ll(self, parameters, observations):
    #     """
    #     Returns the pointwise log-likelihood of the population model
    #     parameters
    #     for each observation.

    #     Parameters
    #     ----------
    #     parameters
    #         An array-like object with the parameters of the population model.
    #     observations
    #         An array-like object with the observations of the individuals.
    #         Each
    #         entry is assumed to belong to one individual.
    #     """
    #     # TODO: Needs proper research to establish which pointwise
    #     # log-likelihood makes sense for hierarchical models.
    #     # Also needs to be adapted to match multi-dimensional API.
    #     raise NotImplementedError
    #     # # Get fixed parameter values
    #     # if self._fixed_params_mask is not None:
    #     #     self._fixed_params_values[~self._fixed_params_mask] = \
    #               parameters
    #     #     parameters = self._fixed_params_values

    #     # # Compute log-likelihood
    #     # scores = self._population_model.compute_pointwise_ll(
    #     #     parameters, observations)

    #     # return scores

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        parameters = np.asarray(parameters).flatten()
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood and sensitivities
        kwargs['flattened'] = True
        score, dpsi, dtheta = self._population_model.compute_sensitivities(
            parameters, observations, dlogp_dpsi, *args, **kwargs)

        if self._fixed_params_mask is None:
            return score, dpsi, dtheta

        return score, dpsi, dtheta[~self._fixed_params_mask]

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        :param name_value_dict: A dictionary with model parameter names as
            keys, and parameter values as values.
        :type name_value_dict: Dict[str:float]
        """
        # Check type
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        # If no model parameters have been fixed before, instantiate a mask
        # and values
        if self._fixed_params_mask is None:
            self._fixed_params_mask = np.zeros(
                shape=self._n_parameters, dtype=bool)

        if self._fixed_params_values is None:
            self._fixed_params_values = np.empty(shape=self._n_parameters)

        # Update the mask and values
        for index, name in enumerate(
                self._population_model.get_parameter_names()):
            try:
                value = name_value_dict[name]
            except KeyError:
                # KeyError indicates that parameter name is not being fixed
                continue

            # Fix parameter if value is not None, else unfix it
            self._fixed_params_mask[index] = value is not None
            self._fixed_params_values[index] = value

        # If all parameters are free, set mask and values to None again
        if np.alltrue(~self._fixed_params_mask):
            self._fixed_params_mask = None
            self._fixed_params_values = None

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        return self._population_model.get_covariate_names()

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return self._population_model.get_dim_names()

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        names = self._population_model.get_parameter_names(exclude_dim_names)

        # Remove fixed model parameters
        if self._fixed_params_mask is not None:
            names = np.array(names)
            names = names[~self._fixed_params_mask]
            names = list(names)

        return copy.copy(names)

    def get_population_model(self):
        """
        Returns the original population model.
        """
        return self._population_model

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        # Get individual parameters
        n_indiv, n_pop = self._population_model.n_hierarchical_parameters(
            n_ids)

        # If parameters have been fixed, updated number of population
        # parameters
        if self._fixed_params_mask is not None:
            n_pop = int(np.sum(self._fixed_params_mask))

        return (n_indiv, n_pop)

    def n_fixed_parameters(self):
        """
        Returns the number of fixed model parameters.
        """
        if self._fixed_params_mask is None:
            return 0

        n_fixed = int(np.sum(self._fixed_params_mask))

        return n_fixed

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        # Get number of fixed parameters
        n_fixed = 0
        if self._fixed_params_mask is not None:
            n_fixed = int(np.sum(self._fixed_params_mask))

        # Subtract fixed parameters from total number
        n_parameters = self._n_parameters - n_fixed

        return n_parameters

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        # Sample from population model
        parameters = self._population_model.sample(sample)

        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            return_parameters = np.copy(self._fixed_params_values)
            return_parameters[self._fixed_params_mask] = self._fixed_params_values
            return_parameters[~self._fixed_params_mask] = parameters #pylint:disable=invalid-unary-operand-type
        else:
            return_parameters = parameters

        return return_parameters

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Sample from population model
        sample = self._population_model.sample(
            parameters=parameters, n_samples=n_samples, seed=seed, *args,
            **kwargs)

        return sample

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Sample from population model
        sample = self._population_model.sample_from_cdf(
            parameters=parameters, cdf=cdf, *args,
            **kwargs)

        return sample

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        If the model has no covariates, input is ignored.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        """
        self._population_model.set_covariate_names(names)

    def set_dim_names(self, names=None):
        """
        Sets the names of the population model dimensions.

        :param names: A list of dimension names. If ``None``, dimension names
            are reset to defaults.
        :type names: List[str], optional
        """
        self._population_model.set_dim_names(names)

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        self._population_model.set_n_ids(n_ids)

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
            If ``None``, parameter names are reset to defaults.
        """
        if names is None:
            # Reset names to defaults
            self._population_model.set_parameter_names()
            return None

        # Check input
        if len(names) != self.n_parameters():
            raise ValueError(
                'Length of names does not match n_parameters.')

        # Limit the length of parameter names
        for name in names:
            if len(name) > 50:
                raise ValueError(
                    'Parameter names cannot exceed 50 characters.')

        parameter_names = [str(label) for label in names]

        # Reconstruct full list of error model parameters
        if self._fixed_params_mask is not None:
            names = np.array(
                self._population_model.get_parameter_names(), dtype='U50')
            names[~self._fixed_params_mask] = parameter_names
            parameter_names = names

        # Set parameter names
        self._population_model.set_parameter_names(parameter_names)


class TruncatedGaussianModel(PopulationModel):
    r"""
    A population model which models model parameters across individuals
    as Gaussian random variables which are truncated at zero.

    A truncated Gaussian population model assumes that a model parameter
    :math:`\psi` varies across individuals such that :math:`\psi` is
    Gaussian distributed in the population for :math:`\psi` greater 0

    .. math::
        p(\psi |\mu, \sigma) =
        \frac{1}{1 - \Phi (-\mu / \sigma )} \frac{1}{\sqrt{2\pi} \sigma}
        \exp\left(-\frac{(\psi - \mu )^2}
        {2 \sigma ^2}\right)\quad \text{for} \quad \psi > 0

    and :math:`p(\psi |\mu, \sigma) = 0` for :math:`\psi \leq 0`.
    :math:`\Phi (\psi )` denotes the cumulative distribution function of
    the Gaussian distribution.

    Here, :math:`\mu` and :math:`\sigma ^2` are the
    mean and variance of the untruncated Gaussian distribution.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """
    def __init__(self, n_dim=1, dim_names=None):
        super(TruncatedGaussianModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = ['Mu'] * self._n_dim + ['Sigma'] * self._n_dim

    @staticmethod
    def _compute_log_likelihood(mus, sigmas, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.

        We are using the relationship between the Gaussian CDF and the
        error function

        ..math::
            Phi(x) = (1 + erf(x/sqrt(2))) / 2

        mus shape: (n_ids, n_dim)
        sigmas shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)
        """
        # Return infinity if any psis are negative
        if np.any(observations < 0):
            return -np.inf

        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_likelihood = - np.sum(
                np.log(2 * np.pi * sigmas**2) / 2
                + (observations - mus) ** 2 / (2 * sigmas**2)
                + np.log(1 - _norm_cdf(-mus/sigmas)))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        """
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2) \
            - np.log(1 - math.erf(-mean/std/math.sqrt(2))) + np.log(2)

        # If score evaluates to NaN, return -infinity
        mask = np.isnan(log_likelihood)
        if np.any(mask):
            log_likelihood[mask] = -np.inf
            return log_likelihood

        return log_likelihood

    def _compute_sensitivities(self, mus, sigmas, psi):  # pragma: no cover
        """
        Calculates the log-likelihood and its sensitivities.

        Expects:
        mus shape: (n_ids, n_dim)
        sigmas shape: (n_ids, n_dim)
        psi: (n_ids, n_dim)

        Returns:
        log_likelihood: float
        dpsi: (n_ids, n_dim)
        dtheta: (n_ids, n_parameters, n_dim)
        """
        # Compute log-likelihood score
        log_likelihood = self._compute_log_likelihood(mus, sigmas, psi)

        n_ids = len(psi)
        dtheta = np.empty(shape=(n_ids, self._n_parameters, self._n_dim))
        if np.isinf(log_likelihood):
            return (-np.inf, np.empty(shape=psi.shape), dtheta)

        # Compute sensitivities
        with np.errstate(divide='ignore'):
            dpsi = (mus - psi) / sigmas**2
            dtheta[:, 0] = (
                (psi - mus) / sigmas
                - _norm_pdf(mus/sigmas) / (1 - _norm_cdf(-mus/sigmas))
                ) / sigmas
            dtheta[:, 1] = (
                -1 + (psi - mus)**2 / sigmas**2
                + _norm_pdf(mus/sigmas) * mus / sigmas
                / (1 - _norm_cdf(-mus/sigmas))
                ) / sigmas

        return log_likelihood, dpsi, dtheta

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        #Calculate limits, because a and b aren't simply absolute boundaries
        a = (0 - mus) / sigmas
        b = np.inf
        
        return truncnorm.cdf(observations, loc=mus, scale=sigmas, a=a, b=b)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any(sigmas <= 0):
            # Gaussians are only defined for positive sigmas.
            return -np.inf

        return self._compute_log_likelihood(mus, sigmas, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any(sigmas <= 0):
            # Gaussians are only defined for positive sigmas.
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                n_ids, n_dim = observations.shape
                dtheta = np.empty((n_ids, self._n_parameters, n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        score, dpsi, dtheta = self._compute_sensitivities(
            mus, sigmas, observations)

        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        if not flattened:
            return score, dpsi, dtheta

        # Sum contributions across individuals and flatten
        dtheta = np.sum(dtheta, axis=0).flatten()
        return score, dpsi, dtheta

    def get_mean_and_std(self, parameters):
        r"""
        Returns the mean and the standard deviation of the population
        for given :math:`\mu` and :math:`\sigma`.

        The mean and variance of the parameter :math:`\psi` are given
        by

        .. math::
            \mathbb{E}\left[ \psi \right] =
                \mu + \sigma F(\mu/\sigma)
            \quad \text{and} \quad
            \text{Var}\left[ \psi \right] =
                \sigma ^2 \left[
                    1 - \frac{\mu}{\sigma}F(\mu/\sigma)
                    - F(\mu/\sigma) ^2
                \right],

        where :math:`F(\mu/\sigma) = \phi(\mu/\sigma )/(1-\Phi(-\mu/\sigma))`
        is a function given by the Gaussian probability density function
        :math:`\phi(\psi)` and the Gaussian cumulative distribution function
        :math:`\Phi(\psi)`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :rtype: np.ndarray of shape ``(p_per_dim, n_dim)``
        """
        # Check input
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        mus = parameters[0]
        sigmas = parameters[1]
        if np.any(sigmas < 0):
            raise ValueError('The standard deviation cannot be negative.')

        # Compute mean and standard deviation
        output = np.empty((self._n_parameters, self._n_dim))
        output[0] = \
            mus + sigmas * norm.pdf(mus/sigmas) / (1 - norm.cdf(-mus/sigmas))
        output[1] = np.sqrt(
            sigmas**2 * (
                1 -
                mus / sigmas * norm.pdf(mus/sigmas)
                / (1 - norm.cdf(-mus/sigmas))
                - (norm.pdf(mus/sigmas) / (1 - norm.cdf(-mus/sigmas)))**2)
            )

        return output

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        return (n_ids * self._n_dim, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def reverse_sample(self, sample, fa=None, fb=np.inf, fast=True):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        fa
            A float for the lower bound of the truncated Gaussian. Beware that this is relative to the mean.
        fb
            A float for the upper bound of the truncated Gaussian. Beware that this is relative to the mean.
        """
        mean, std = np.mean(sample), np.std(sample)
        if fast:
            return mean, std
        else:
            #Use a function that allows us to constrain a and b to actually sensible values
            func = lambda loc, scale: truncnorm.nnlf([(fa-loc)/scale, fb, loc, scale], sample)
            func2 = lambda params: func(*params)
            mean, std = basinhopping(
                func2, x0=(mean, std), niter=1,
                minimizer_kwargs={"bounds": [[0.1*mean, 10*mean], [0.1*std, 10*std]]})
            return mean, std

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Get parameters
        mus = parameters[0]
        sigmas = parameters[1]

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A truncated Gaussian distribution only accepts strictly '
                'positive standard deviations.')

        # Convert seed to int if seed is a rng
        # (Unfortunately truncated normal is not yet available with numpys
        # random number generator API)
        if isinstance(seed, np.random.Generator):
            # Draw new seed such that rng is propagated, but truncated normal
            # samples can also be seeded.
            seed = seed.integers(low=0, high=1E6)
        np.random.seed(seed)

        #Calculate limits, because a and b aren't simply absolute boundaries
        a = (0 - mus) / sigmas
        b = np.inf
        
        # Sample from population distribution
        samples = truncnorm.rvs(
            a=a, b=b, loc=mus, scale=sigmas, size=sample_shape)

        return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Get parameters
        mus = parameters[0]
        sigmas = parameters[1]

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A truncated Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        #Calculate limits, because a and b aren't simply absolute boundaries
        a = (0 - mus) / sigmas
        b = np.inf
        
        return truncnorm.ppf(cdf, loc=mus, scale=sigmas, a=a, b=b).reshape(sample_shape)

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Mu'] * self._n_dim + ['Sigma'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class TruncatedGaussianModelRelativeSigma(TruncatedGaussianModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution.

    The difference between this model and GaussianModel is that the standard
    deviation :math:`\sigma` is calculated as the product of the two input params,
    :math:`\mu` and :math:`\frac{\sigma}{\mu}`

    Extends :class:`GaussianModel`.
    """

    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(TruncatedGaussianModelRelativeSigma, self).__init__()

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = ['Mu'] * self._n_dim + ['Rel. Sigma'] * self._n_dim

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        means = parameters[:, 0]
        stdRatios = parameters[:, 1]
        sigmas = (stdRatios*np.abs(means))

        if np.any(stdRatios < 0):
            # The std. of the Gaussian distribution is strictly positive
            return -np.inf

        #Calculate limits, because a and b aren't simply absolute boundaries
        a = (0 - means) / sigmas
        b = np.inf
        
        return truncnorm.cdf(observations, loc=means, scale=sigmas, a=a, b=b)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        means = parameters[:, 0]
        stdRatios = parameters[:, 1]
        sigmas = (stdRatios*np.abs(means))

        if np.any(stdRatios < 0):
            # The std. of the Gaussian distribution is strictly positive
            return -np.inf

        return self._compute_log_likelihood(means, sigmas, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, flattened=True,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivities to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        stdRatios = parameters[:, 1]
        sigmas = stdRatios*np.abs(mus)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution is strictly positive
            dtheta = np.empty(self._n_parameters)
            if not flattened:
                dtheta = np.empty((len(observations), 2, self._n_dim))
            return -np.inf, np.empty(observations.shape), dtheta

        return self._compute_non_centered_sensitivities(
            sigmas, observations, dlogp_dpsi, flattened)

    def reverse_sample(self, sample, fa=None, fb=np.inf, fast=True):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        fa
            A float for the lower bound of the truncated Gaussian. Beware that this is relative to the mean.
        fb
            A float for the upper bound of the truncated Gaussian. Beware that this is relative to the mean.
        """
        mean, std = np.mean(sample), np.std(sample)
        if fast:
            return mean, std
        else:
            #Use a function that allows us to constrain a and b to actually sensible values
            func = lambda loc, scale: truncnorm.nnlf([(fa-loc)/scale, fb, loc, scale], sample)
            func2 = lambda params: func(*params)
            mean, std = basinhopping(
                func2, x0=(mean, std), niter=1,
                minimizer_kwargs={"bounds": [[0.1*mean, 10*mean], [0.1*std, 10*std]]})
            return mean, std

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Get parameters
        mus = parameters[0]
        stdRatios = parameters[1]
        sigmas = stdRatios*np.abs(mus)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Convert seed to int if seed is a rng
        # (Unfortunately truncated normal is not yet available with numpys
        # random number generator API)
        if isinstance(seed, np.random.Generator):
            # Draw new seed such that rng is propagated, but truncated normal
            # samples can also be seeded.
            seed = seed.integers(low=0, high=1E6)
        np.random.seed(seed)

        #Calculate limits, because a and b aren't simply absolute boundaries
        a = (0 - mus) / sigmas
        b = np.inf
        
        # Sample from population distribution
        samples = truncnorm.rvs(a=a, b=b, loc=mus, scale=sigmas, size=sample_shape)

        return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Get parameters
        mus = parameters[0]
        stdRatios = parameters[1]
        sigmas = stdRatios*np.abs(mus)

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A truncated Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        #Calculate limits, because a and b aren't simply absolute boundaries
        a = (0 - mus) / sigmas
        b = np.inf
        
        return truncnorm.ppf(cdf, loc=mus, scale=sigmas, a=a, b=b).reshape(sample_shape)

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a GaussianModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Mu'] * self._n_dim + ['Rel. Sigma'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class UniformModel(PopulationModel):
    """
    A population model which imposes no relationship on the model parameters
    across individuals, but which returns uniform samples of parameters.

    A uniform model assumes that the parameters across individuals are
    independent, but constrained within a region.

    Extends :class:`PopulationModel`.
    """

    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(UniformModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 0

        # Set default parameter names
        self._parameter_names = None

        #Uninitialised boundaries
        self._boundaries, self._value = None, None
        self._centered = bool(centered)

    def compute_cdf(self, parameters, observations, *args, **kwargs):
        """
        Calculated the cumulative distribution function from the underlying likelihood function, given population parameters and observations (individual parameters).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        if self._boundaries is None:
            raise ValueError("Uniform model was not initialised.")

        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        return uniform.cdf(observations, low=self._boundaries.lower(), high=self._boundaries.upper())

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        A uniform population model imposes no restrictions on the
        individuals, as a result the log-likelihood score is zero irrespective
        of the model parameters, as long as they are within bounds.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        return self._value if self._boundaries.check(observations) else -np.inf

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        A uniform population model imposes no restrictions on the
        individuals, as a result the log-likelihood score is zero irrespective
        of the model parameters, as long as they are within bounds.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        n_observations = len(observations)
        scalar = self._value if self._boundaries.check(observations) else -np.inf
        return np.ones(shape=n_observations)*scalar/n_observations

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the parameters and the observations.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        #Compute log-likelihood
        scalar = self._value if self._boundaries.check(observations) else -np.inf

        #Derivative w.r.t. observations
        d_obs = 0

        #Derivative w.r.t parameters is 0 above and below the boundaries,
        # and infinity at the boundaries. Strictly we should give NaN, but that'd be messy.
        d_params = 0 if self._boundaries.check(observations) else -np.inf

        #sensitivities
        sensitivites = np.array([d_obs, d_params])

        return scalar, sensitivites

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        return copy.copy(self._parameter_names)

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_ids = int(n_ids)

        return (n_ids, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        low, high = np.min(sample), np.max(sample)
        return (low, high)

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            Parameter values of the top-level parameters that are used for the
            simulation.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        if self._boundaries is None:
            raise ValueError("Uniform model was not initialised.")

        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Sample from population distribution
        rng = np.random.default_rng(seed=seed)
        samples = rng.uniform(
            low=self._boundaries.lower(), high=self._boundaries.upper(), size=sample_shape)

        return samples

    def sample_from_cdf(self, parameters, cdf, *args, **kwargs):
        """
            Returns samples from the population distribution given values of the CDF, using the inverse-CDF (ppf) function, instead of generating them randomly. This function can be used to map from a different distribution to this one. 

            :param parameters: Parameters of the population model.
            :type parameters: np.ndarray of shape ``(p,)`` or
                ``(p_per_dim, n_dim)``
            :param cdf: Values of the cumulative distribution function, which must lie between 0 and 1. One value must be given for each sample.
            :type cdf: np.ndarray of shape (n_samples).
        """
        if self._boundaries is None:
            raise ValueError("Uniform model was not initialised.")

        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        sample_shape = (len(cdf), self._n_dim)

        # Sample from population distribution
        if np.any((cdf<0)|(cdf>1)):
            raise ValueError("Values of the CDF must lie between 0 and 1.")

        return uniform.ppf(cdf, low=self._boundaries.lower(), high=self._boundaries.upper()).reshape(sample_shape)

    def set_boundaries(self, lower_or_boundaries, upper=None):
        """Sets the pints.RectangularBoundaries object within the class.
        
            Parameters
            ----------
            lower
                A 1d array of lower boundaries.
            upper
                The corresponding upper boundaries
        """
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError(
                    'UniformModel requires a lower and an upper bound, or a'
                    ' single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.RectangularBoundaries(
                lower_or_boundaries, upper)

        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._value = -np.log(np.product(self._boundaries.range()))
        else:
            self._value = 1

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        A uniform population model has no population parameters.
        However, a name may nevertheless be assigned for convience.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = None
            return None

        if len(names) != 1:
            raise ValueError(
                'Length of names has to be 1.')

        self._parameter_names = [str(label) for label in names]


def _norm_cdf(x):  # pragma: no cover
    """
    Returns the cumulative distribution function value of a standard normal
    Gaussian distribtion.
    """
    return 0.5 * (1 + erf(x/np.sqrt(2)))


def _norm_pdf(x):  # pragma: no cover
    """
    Returns the probability density function value of a standard normal
    Gaussian distribtion.
    """
    return np.exp(-x**2/2) / np.sqrt(2 * np.pi)


def is_composed_population_model(population_model):
    return isinstance(population_model, (chi.ComposedPopulationModel, chi.ComposedCorrelationPopulationModel))

def is_heterogeneous_or_uniform_model(population_model):
    """Returns true if the given population_model is one of the Heterogeneous or Uniform population models."""
    return is_heterogeneous_model(population_model) or is_uniform_model(population_model)


def is_heterogeneous_model(population_model):
    """Returns true if the given population_model is one of the Heterogeneous population models."""
    return isinstance(population_model, (chi.HeterogeneousModel))


def is_pooled_model(population_model):
    """Returns true if the given population_model is one of the pooled population models."""
    return isinstance(population_model, (chi.PooledModel))


def is_uniform_model(population_model):
    """Returns true if the given population_model is one of the uniform population models."""
    return isinstance(population_model, (chi.UniformModel))


