#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import math

import numpy as np
import pints
from scipy.stats import norm, lognorm, truncnorm

class PopulationModel(object):
    """
    A base class for population models.
    """

    def __init__(self):
        super(PopulationModel, self).__init__()

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        raise NotImplementedError

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        raise NotImplementedError

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        raise NotImplementedError

    def get_parameter_names(self):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        raise NotImplementedError

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        raise NotImplementedError


class KolmogorovSmirnovPopulationModel(PopulationModel):
    """
    A base class for population models based on the Kolmogorov Smirnov statistic.
    These models compare the CDF of the outputs of the mechanistic model over
    all individuals to the CDF of the observations over all individuals.
    """
    _biomarker = None
    _cdf_param_index = None
    _cdfInLogBiomarker = None
    _n_bins = None
    _n_noise_parameters = None
    _n_parameters = None
    _n_pop_parameters = None
    _n_times = None
    _obs_cdf_edges = None
    _obs_cdf_heights = None
    _obs_cdf_times = None
    _parameter_names = None
    _pooled_param_index = None

    def __init__(self, cdf_in_log_biomarker, biomarker):
        super(KolmogorovSmirnovPopulationModel, self).__init__()

    def _calculate_PDF(self, values, time_key="time", biom_key="biomarker", meas_key="bm_value", bins=None):
        '''
            Given a pandas dataframe `values', calculate the CDF over individuals, for the biomarker of interest,
            as a function of time. Returns cdfTimes, cdfHeights, cdfEdges.

            Parameters:
            - values: pandas.DataFrame. the values over which the CDF should be calculated.
            - time_key: string. The column in which values of the times at which observations were taken is recorded.
            - biom_key: string. The column in which the name of the biomarker for each observation is recorded.
            - meas_key: string. The column in which the value of each observation is recorded.
            - bins: numpy.Array or None. If given, these are the edges for the histogram, given to numpy.histogram.

            Return value:
            - cdfTimes: numpy.Array. The times at which the CDF of the observation over individuals was calculated.
            - pdfHeights: numpy.Array. The probability density of the CDF in each histogram bin.
            - cdfEdges: numpy.Array. The edges of each histogram bin.
        '''
        # Mask data for biomarker
        if biom_key in values.columns:
            mask  = values[biom_key] == self._biomarker
            temp_df = values[mask]
        else:
            temp_df = copy.copy(values)

        # Filter observations for non-NaN entries
        mask  = temp_df[meas_key].notnull()
        temp_df = temp_df[mask]

        # If we already have the observation CDF, mask for those times
        if self._obs_cdf_times is not None:
            mask = np.any(
                [temp_df[time_key]==t for t in self._obs_cdf_times], axis=0)
            temp_df = temp_df[mask]

        # Otherwise, filter times for non-NaN entries
        else:
            mask  = temp_df[time_key].notnull()
            temp_df = temp_df[mask]

        #Find unique times
        cdfTimes = np.unique(temp_df[time_key])

        #Check this is equal to the times we already have
        if self._obs_cdf_times is not None:
            if len(cdfTimes) != self._n_times or not all(cdfTimes == self._obs_cdf_times):
                raise ValueError("Estimates are not taken at the same times as the observations.")

        #Log data, if necessary
        if self._cdfInLogBiomarker:
            temp_df[meas_key] = np.log(temp_df[meas_key].astype(float))
            if bins is not None:
                bins = np.log(bins)

        #If the number of bins is not given, choose an argument for np.histogram
        givenBins = (bins is not None)
        if not givenBins:
            #Freedman-Diaconis rule:
            # bin width = 2*IQR*n^(-1/3)
            # num bins = (max-min)/width

            #Get IQR
            allValues = temp_df[meas_key].to_numpy(float)
            q = np.nanpercentile(allValues, [25, 75])
            iqr = q[1] - q[0]

            #Get median count per time point
            counts = [np.sum(temp_df[time_key]==t) for t in cdfTimes]
            medianCount = int(np.round(np.median(counts)))

            #Get optimal width and number of bins
            h = 2*iqr * medianCount**(-1/3)
            bins = int((np.nanmax(allValues) - np.nanmin(allValues))/h)+1

            #Deal with silly numbers of bins: 3 <= n_bins <= 25
            bins = np.min([bins, 25])
            bins = np.max([bins, 3])

        elif len(bins) != len(self._obs_cdf_edges):
            raise ValueError("Input number of bins must be the same as calculated for the observations.")

        #Initialise storage
        pdfHeights = [None]*len(cdfTimes)
        cdfEdges   = [None]*len(cdfTimes)

        #Find CDF at each time point
        for tind, time in enumerate(cdfTimes):
            #Reset
            skip = False

            #Mask data for time
            mask = temp_df[time_key] == time
            temp_t_df = temp_df[mask]

            #Find input data for histogram
            histData = temp_t_df[meas_key].to_numpy(float)

            #Get input of `bins' for numpy.histogram
            _bins = bins[tind] if hasattr(bins, "__len__") else bins

            # If we have been given the bins, see if any data lies outside the range of the bins.
            # For example, if we have 2 data points, and one is above the largest bin, we want the
            # CDF to add up to 0.5, not 1.0. So, we multiply density by 0.5.
            # We perform the same trick for data below the smallest bin, too.
            if givenBins:
                binEdges = bins[tind]
                dataBelow = sum(histData<binEdges[0])
                dataAbove = sum(histData>binEdges[-1])
                densityReduction = (len(histData)-dataAbove-dataBelow)/len(histData)

                if densityReduction == 0:
                    binHeights = np.zeros(len(_bins)-1)
                    skip = True
            else:
                densityReduction = 1

            #Find histogram
            if not skip:
                binHeights, binEdges = np.histogram(histData, bins=_bins, density=True)
                binHeights *= densityReduction

            pdfHeights[tind] = binHeights
            cdfEdges[tind]   = binEdges

        if self._cdfInLogBiomarker:
            cdfEdges = np.exp(cdfEdges)

        return np.array(cdfTimes), np.array(pdfHeights), np.array(cdfEdges)

    def _calculate_CDF(self, values, time_key="time", biom_key="biomarker", meas_key="bm_value", bins=None):
        '''
            Given a pandas dataframe `values', calculate the CDF over individuals, for the biomarker of interest,
            as a function of time. Returns cdfTimes, cdfHeights, cdfEdges.

            Parameters:
            - values: pandas.DataFrame. the values over which the CDF should be calculated.
            - time_key: string. The column in which values of the times at which observations were taken is recorded.
            - biom_key: string. The column in which the name of the biomarker for each observation is recorded.
            - meas_key: string. The column in which the value of each observation is recorded.
            - bins: numpy.Array or None. If given, these are the edges for the histogram, given to numpy.histogram.

            Return value:
            - cdfTimes: numpy.Array. The times at which the CDF of the observation over individuals was calculated.
            - pdfHeights: numpy.Array. The probability density of the CDF in each histogram bin.
            - cdfEdges: numpy.Array. The edges of each histogram bin.
        '''
        #Here, we multiply all cdfHeights by dx = (cdfEdges[:, 1:] - cdfEdges[:, :-1]).
        #Both data and estimate will be multiplied by the same dx, so we could compare PDFs
        #without muliplitcation to find the difference between them. But, we still need to do
        #this to make different time points contribute equally.

        #Get PDF
        cdfTimes, pdfHeights, cdfEdges = self._calculate_PDF(
            values, time_key, biom_key, meas_key, bins)

        #Find widths
        if self._cdfInLogBiomarker:
            _e = np.log(cdfEdges)
            dx = (_e[:, 1:] - _e[:, :-1])
        else:
            dx = (cdfEdges[:, 1:] - cdfEdges[:, :-1])

        #Find CDFs
        cdfHeights = np.array([
            [
                np.sum((_heights*_dx)[:i+1]) for i in range(len(_dx))
            ] for _heights, _dx in zip(pdfHeights, dx)
        ])
        return cdfTimes, cdfHeights, cdfEdges

    @staticmethod
    def _compute_log_likelihood(
            sigma, model_cdf, observation_cdf):  # pragma: no cover
        """
        Calculates the log-likelihood using numba speed up.
        """
        if sigma <= 0:
            # sigma is strictly positive
            return -np.inf

        # #Check shape
        # if len(model_cdf) != self._n_times:
        #     raise ValueError("Model CDF must have same size _n_times."
        #         f"Sizes: {len(model_cdf)}, {self._n_times}."
        #     )

        # if len(model_cdf) != len(observation_cdf):
        #     raise ValueError("Model CDF must have same dimensionality as observation CDF."
        #         f"Sizes: {len(model_cdf)}, {len(observation_cdf)}."
        #     )

        # Calculate the K-S statistic:
        # the maximum distance between the two CDFs, as a function of time
        ksStat = np.max(np.abs(model_cdf - observation_cdf), axis=1)

        # Compute log-likelihood
        n_times = len(model_cdf)
        log_likelihood = \
            - n_times * (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - np.sum(ksStat**2) / sigma**2 / 2

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(
            sigma, model_cdf, observation_cdf):  # pragma: no cover
        """
        Calculates the pointwise log-likelihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        if sigma <= 0:
            # sigma is strictly positive
            n_times = len(model_cdf)
            return np.full(n_times, -np.inf)

        # Calculate the K-S statistic:
        # the maximum distance between the two CDFs, as a function of time
        ksStat = np.max(np.abs(model_cdf - observation_cdf), axis=1)

        # Compute log-likelihood
        n_times = len(model_cdf)
        pointwise_ll = \
            - (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - (ksStat**2) / sigma**2 / 2

        return pointwise_ll

    @staticmethod
    def _compute_sensitivities(
            sigma, model_cdf, model_sensitivities,
            observation_cdf):  # pragma: no cover
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        """
        if sigma <= 0:
            # sigma is strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        # Calculate the K-S statistic:
        # the maximum distance between the two CDFs, as a function of time
        ksStat = np.max(np.abs(model_cdf - observation_cdf), axis=2)
        sumSquareKs = np.sum(ksStat**2, axis=0)

        # Compute log-likelihood
        n_times = len(model_cdf)
        log_likelihood = \
            - n_times * (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - sumSquareKs / sigma**2 / 2

        # Compute sensitivities
        dpsi = \
            np.sum(ksStat * model_sensitivities, axis=0) / sigma**2
        dsigma = \
            sumSquareKs / sigma**3 - n_times / sigma
        sensitivities = np.concatenate((dpsi, dsigma))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_outputs,
    time_key="time", biom_key="biomarker", meas_key="bm_value"):
        """
        Returns the log-likelihood of the population model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        model_estimates
            A pandas dataframe with the outputs from the mechanistic model
            for each patient.
        """
        #Error checking
        if self._obs_cdf_heights is None:
            raise ValueError('The data has not been set.')
        parameters = np.asarray(parameters)

        #Calculate CDF over outputs
        times, est_cdf_heights = self._calculate_CDF(
            model_outputs, time_key, biom_key, meas_key, bins=self._obs_cdf_edges)[:2]

        #Check for compatibility
        if not all(times == self._obs_cdf_times):
            raise ValueError("Times at which CDF was calculated are "
                f"not the same as for the data, for biomarker {self._biomarker}"
            )

        #We could calculate the K-S statistic here, via
        # np.max(np.abs(est_cdf_heights - self._obs_cdf_heights), axis=1)
        #But instead, we'll calculate a likelihood function.
        sigma = parameters[self._cdf_param_index]
        return self._compute_log_likelihood(
            sigma, est_cdf_heights, self._obs_cdf_heights)

    def compute_pointwise_ll(self, parameters, model_outputs,
    time_key="time", biom_key="biomarker", meas_key="bm_value"):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        model_estimates
            A pandas dataframe with the outputs from the mechanistic model
            for each patient.
        """
        #Error checking
        parameters = np.asarray(parameters)
        if self._obs_cdf_heights is None:
            raise ValueError('The data has not been set.')

        #Calculate CDF over outputs
        times, est_cdf_heights = self._calculate_CDF(
            model_outputs, time_key, biom_key, meas_key, bins=self._obs_cdf_edges)[:2]

        #Check for compatibility
        if not all(times == self._obs_cdf_times):
            raise ValueError("Times at which CDF was calculated are "
                f"not the same as for the data, for biomarker {self._biomarker}"
            )

        sigma = parameters[self._cdf_param_index]
        return self._compute_pointwise_ll(
            sigma, est_cdf_heights, self._obs_cdf_heights)

    def compute_sensitivities(self, parameters, model_outputs, model_sensitivities,
    time_key="time", biom_key="biomarker", meas_key="bm_value"):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        model_estimates
            A pandas dataframe with the outputs from the mechanistic model
            for each patient.
        model_sensitivities
            A pandas dataframe with the outputs from compute_sensitivites for 
            the mechanistic model, for each patient.
        """
        #Error checking
        parameters = np.asarray(parameters)
        sens = np.asarray(model_sensitivities)
        if self._obs_cdf_heights is None:
            raise ValueError('The data has not been set.')

        #Calculate CDF over outputs
        times, est_cdf_heights = self._calculate_CDF(
            model_outputs, time_key, biom_key, meas_key, bins=self._obs_cdf_edges)[:2]

        #Check for compatibility
        if not all(times == self._obs_cdf_times):
            raise ValueError("Times at which CDF was calculated are "
                f"not the same as for the data, for biomarker {self._biomarker}"
            )

        model = np.asarray(est_cdf_heights).reshape((self._n_times, self._n_bins, 1))
        obs = np.asarray(self._obs_cdf_heights).reshape((self._n_times, self._n_bins, 1))

        sigma = parameters[self._cdf_param_index]
        return self._compute_sensitivities(
            sigma, model, sens, obs)

    def create_observation_CDF(self, data, time_key="time", biom_key="biomarker", meas_key="bm_value"):
        """
            Given a pandas dataframe `data', calculate the CDF over individuals, for the biomarker of interest,
            as a function of time.

            Parameters:
            - values: pandas.DataFrame. the values over which the CDF should be calculated.
            - time_key: string. The column in which values of the times at which observations were taken is recorded.
            - biom_key: string. The column in which the name of the biomarker for each observation is recorded.
            - meas_key: string. The column in which the value of each observation is recorded.
        """
        #Check that the user has chosen an actual population model (heterogeneous or pooled)
        if self._biomarker is None:
            raise NotImplementedError("self._biomarker not set,"
                "was a correct child class of KolmogorovSmirnovPopulationModel chosen?")

        #Reset any known values
        self._obs_cdf_times = None
        self._obs_cdf_heights = None
        self._obs_cdf_edges = None
        self._n_times = None
        self._n_bins  = None

        #Calculate CDF
        times, heights, edges = self._calculate_CDF(data, time_key, biom_key, meas_key)

        #Record data
        self._obs_cdf_times   = times
        self._obs_cdf_heights = heights
        self._obs_cdf_edges   = edges

        #Calculate shape required for model outputs
        self._n_times = len(self._obs_cdf_times)
        self._n_bins = len(heights[0])

        return times, heights, edges

    def get_parameter_names(self):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.
        """
        return copy.copy(self._parameter_names)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample_noise(self, parameters, model_outputs, n_samples=None, seed=None,
    time_key="time", biom_key="biomarker", meas_key="bm_value"):
        """
        Returns samples from the mechanistic model-population model pair in form
        of a NumPy array of shape ``(len(model_output), n_samples)``.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_outputs
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`.
        n_samples
            Number of samples from the error model for each entry in
            ``model_output``. If ``None``, one sample is assumed.
        seed
            Seed for the pseudo-random number generator. If ``None``, the
            pseudo-random number generator is not seeded.
        """
        #Error checking
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of population model parameters.')

        if self._obs_cdf_heights is None:
            raise ValueError('The data has not been set.')

        # ─── CDF ─────────────────────────────────────────────────────────
        #Calculate CDF over outputs
        times, est_cdf_heights = self._calculate_CDF(
            model_outputs, time_key, biom_key, meas_key, bins=self._obs_cdf_edges)[:2]

        #Check for compatibility
        if not all(times == self._obs_cdf_times):
            raise ValueError("Times at which CDF was calculated are "
                f"not the same as for the data, for biomarker {self._biomarker}"
            )

        # ─── ERROR MODEL ─────────────────────────────────────────────────
        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (self._n_times, self._n_bins, int(n_samples))

        # Get parameters
        sigma = parameters[0]

        # Sample from Gaussian distributions
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(loc=0, scale=sigma, size=sample_shape)

        # Construct final samples, by addoing noise to the obtained model CDFs
        est_cdf_heights = np.expand_dims(est_cdf_heights, axis=2)
        samples = est_cdf_heights + samples

        return samples


def is_heterogeneous_or_uniform_model(population_model):
    """Returns true if the given population_model is one of the Heterogeneous or Uniform population models."""
    return is_heterogeneous_model(population_model) or is_uniform_model(population_model)


def is_heterogeneous_model(population_model):
    """Returns true if the given population_model is one of the Heterogeneous population models."""
    return isinstance(population_model, (HeterogeneousModel, KolmogorovSmirnovHeterogeneousModel))


def is_pooled_model(population_model):
    """Returns true if the given population_model is one of the pooled population models."""
    return isinstance(population_model, (PooledModel, KolmogorovSmirnovPooledModel))


def is_uniform_model(population_model):
    """Returns true if the given population_model is one of the uniform population models."""
    return isinstance(population_model, (UniformModel, KolmogorovSmirnovUniformModel))


class GaussianModel(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution.

    A Gaussian population model assumes that a model parameter
    :math:`\psi` varies across individuals such that :math:`\psi` is
    Gaussian distributed in the population

    .. math::
        p(\psi |\mu, \sigma) =
        \frac{1}{\sqrt{2\pi} \sigma}
        \exp\left(-\frac{(\psi - \mu )^2}
        {2 \sigma ^2}\right).

    Here, :math:`\mu` and :math:`\sigma ^2` are the
    mean and variance of the Gaussian distribution.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(GaussianModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mean', 'Std.']

    @staticmethod
    def _compute_log_likelihood(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.
        """
        # Compute log-likelihood score
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum((observations - mean) ** 2) / (2 * std**2)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2)

        return log_likelihood

    @staticmethod
    def _compute_sensitivities(mean, std, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        std = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * (np.log(2 * np.pi) / 2 + np.log(std)) \
            - np.sum((psi - mean)**2) / (2 * std**2)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mean - psi) / std**2

        # Copmute sensitivities w.r.t. parameters
        dmean = np.sum(psi - mean) / std**2
        dstd = -n_ids / std + np.sum((psi - mean)**2) / std**3

        sensitivities = np.concatenate((dpsi, np.array([dmean, dstd])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a truncated Gaussian distribution is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu , \sigma | \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return -np.inf

        return self._compute_log_likelihood(mean, std, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a truncated Gaussian distribution is
        the log-pdf evaluated at the observations

        .. math::
            L(\mu , \sigma | \psi _i) =
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, std, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, std, observations)

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
        mean, std = norm.fit(sample)
        return (mean, std)

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mu, sigma = parameters

        if sigma < 0:
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(
            loc=mu, scale=sigma, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a GaussianModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mean', 'Std.']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class GaussianModelRelativeSigma(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution.

    The difference between this model and GaussianModel is that the standard
    deviation :math:`\sigma` is calculated as the product of the two input params,
    :math:`\mu` and :math:`\frac{\sigma}{\mu}`

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(GaussianModelRelativeSigma, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mean', 'Rel. Std.']

    @staticmethod
    def _compute_log_likelihood(mean, stdRatio, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.
        """
        # Compute log-likelihood score
        std = mean*stdRatio
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum((observations - mean) ** 2) / (2 * std**2)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, stdRatio, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        std = mean*stdRatio
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2)

        return log_likelihood

    @staticmethod
    def _compute_sensitivities(mean, stdRatio, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        stdRatio = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        std = mean*stdRatio
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * (np.log(2 * np.pi) / 2 + np.log(std)) \
            - np.sum((psi - mean)**2) / (2 * std**2)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mean - psi) / std**2

        # Copmute sensitivities w.r.t. parameters
        dmean  = -n_ids/mean       +   1/std**2      * np.sum(psi*(psi/mean-1))
        dRatio = -n_ids/stdRatio   +   1/stdRatio**3 * np.sum((psi/mean-1)**2)

        sensitivities = np.concatenate((dpsi, np.array([dmean, dRatio])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a truncated Gaussian distribution is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu , \sigma | \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, stdRatio = parameters

        if stdRatio <= 0:
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return -np.inf

        return self._compute_log_likelihood(mean, stdRatio, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a truncated Gaussian distribution is
        the log-pdf evaluated at the observations

        .. math::
            L(\mu , \sigma | \psi _i) =
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, stdRatio = parameters

        if stdRatio <= 0:
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, stdRatio, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, stdRatio = parameters

        if stdRatio <= 0:
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, stdRatio, observations)

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
        mean, std = norm.fit(sample)
        stdRatio = std/mean
        return (mean, stdRatio)

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mu, stdRatio = parameters

        if stdRatio < 0:
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        sigma = mu*stdRatio
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(
            loc=mu, scale=sigma, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a GaussianModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mean', 'Rel. Std.']
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
    """

    def __init__(self):
        super(HeterogeneousModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 0

        # Set default parameter names
        self._parameter_names = None

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        A heterogenous population model imposes no restrictions on the
        individuals, as a result the log-likelihood score is zero irrespective
        of the model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        return 0

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        A heterogenous population model imposes no restrictions on the
        individuals, as a result the log-likelihood score is zero irrespective
        of the model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        return np.zeros(shape=len(observations))

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
        n_observations = len(observations)
        return 0, np.zeros(shape=n_observations)

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

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        A heterogeneous population model has no population parameters.
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


class KolmogorovSmirnovHeterogeneousModel(KolmogorovSmirnovPopulationModel):
    """
    A Kolmogorov Smirnov model that assumes underlying individual parameters are heterogeneous.
    These models compare the CDF of the outputs of the mechanistic model over
    all individuals to the CDF of the observations over all individuals.
    """

    def __init__(self, cdf_in_log_biomarker, biomarker):
        super(KolmogorovSmirnovHeterogeneousModel, self).__init__(cdf_in_log_biomarker, biomarker)
        # super(KolmogorovSmirnovPopulationModel, self).__init__()

        #Determine whether the likelihood function requires logged data
        self._cdfInLogBiomarker = cdf_in_log_biomarker

        #Set number of parameters and parametr names
        self._parameter_names = ['CDF_Sigma']
        self._n_parameters = 1
        self._n_noise_parameters, self._n_pop_parameters = 1, 0
        self._pooled_param_index, self._cdf_param_index  = None, 0

        #Set biomarker name
        self._biomarker = biomarker

        #Set default values
        self._obs_cdf_times = None
        self._obs_cdf_heights = None
        self._obs_cdf_edges = None
        self._n_times = None
        self._n_bins = None

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

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['CDF_Sigma']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class KolmogorovSmirnovPooledModel(KolmogorovSmirnovPopulationModel):
    """
    A Kolmogorov Smirnov model that assumes underlying individual parameters are identical.
    These models compare the CDF of the outputs of the mechanistic model over
    all individuals to the CDF of the observations over all individuals.
    """

    def __init__(self, cdf_in_log_biomarker, biomarker):
        super(KolmogorovSmirnovPooledModel, self).__init__(cdf_in_log_biomarker, biomarker)
        # super(KolmogorovSmirnovPopulationModel, self).__init__()

        #Determine whether the likelihood function requires logged data
        self._cdfInLogBiomarker = cdf_in_log_biomarker

        #Set number of parameters and parameter names:
        # one for noise, one for population params.
        # The pooled one must be first, because the logic of individual
        # likelihood functions for pooled models relies on it being there
        self._parameter_names = ['Pooled', 'CDF_Sigma']
        self._n_parameters = 2
        self._n_noise_parameters, self._n_pop_parameters = 1, 1
        self._pooled_param_index, self._cdf_param_index  = 0, 1

        #Set biomarker name
        self._biomarker = biomarker

        #Set default values
        self._obs_cdf_times = None
        self._obs_cdf_heights = None
        self._obs_cdf_edges = None
        self._n_times = None
        self._n_bins = None

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
        return (0, self._n_parameters)

    def reverse_sample(self, sample):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        """
        meanOfSamples = np.mean(sample)
        #Just return the same estimate for both CDF_noise and for the actual pooled parameter
        return np.asarray([meanOfSamples, meanOfSamples])

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the underlying population
        distribution.

        For a PooledModel the input top-level parameters are copied
        ``n_samples`` and are returned.

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        #Grab only the population parameter
        samples = np.asarray(parameters[1])

        # If only one sample is wanted, return input parameter
        if n_samples is None:
            return samples

        # If more samples are wanted, broadcast input parameter to shape
        # (n_samples,)
        samples = np.broadcast_to(samples, shape=(n_samples,))
        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Pooled', 'CDF_Sigma']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class KolmogorovSmirnovUniformModel(KolmogorovSmirnovPopulationModel):
    """
    A Kolmogorov Smirnov model that assumes underlying individual parameters are uniformly distributed.
    These models compare the CDF of the outputs of the mechanistic model over
    all individuals to the CDF of the observations over all individuals.
    """

    def __init__(self, cdf_in_log_biomarker, biomarker):
        super(KolmogorovSmirnovUniformModel, self).__init__(cdf_in_log_biomarker, biomarker)
        # super(KolmogorovSmirnovPopulationModel, self).__init__()

        #Determine whether the likelihood function requires logged data
        self._cdfInLogBiomarker = cdf_in_log_biomarker

        #Set number of parameters and parameter names:
        # one for noise, one for population params.
        # The pooled one must be first, because the logic of individual
        # likelihood functions for pooled models relies on it being there
        self._parameter_names = ['CDF_Sigma']
        self._n_parameters = 1
        self._n_noise_parameters, self._n_pop_parameters = 1, 0
        self._pooled_param_index, self._cdf_param_index  = None, 0

        #Set biomarker name
        self._biomarker = biomarker

        #Set default values
        self._obs_cdf_times = None
        self._obs_cdf_heights = None
        self._obs_cdf_edges = None
        self._n_times = None
        self._n_bins = None
        self._boundaries, self._value = None, None

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
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['CDF_Sigma']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class LogNormalModel(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are log-normally distributed.

    A log-normal population model assumes that a model parameter :math:`\psi`
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

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(LogNormalModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mean log', 'Std. log']

    @staticmethod
    def _compute_log_likelihood(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.
        """
        # Compute log-likelihood score
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum(np.log(observations)) \
            - np.sum((np.log(observations) - mean)**2) / 2 / std**2

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Transform observations
        log_psi = np.log(observations)

        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - log_psi \
            - (log_psi - mean) ** 2 / (2 * std**2)

        return log_likelihood

    @staticmethod
    def _compute_sensitivities(mean, std, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        std = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum(np.log(psi)) \
            - np.sum((np.log(psi) - mean)**2) / 2 / std**2

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = - ((np.log(psi) - mean) / std**2 + 1) / psi

        # Copmute sensitivities w.r.t. parameters
        dmean = np.sum(np.log(psi) - mean) / std**2
        dstd = (np.sum((np.log(psi) - mean)**2) / std**2 - n_ids) / std

        sensitivities = np.concatenate((dpsi, np.array([dmean, dstd])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a LogNormalModel is the log-pdf evaluated
        at the observations

        .. math::
            L(\mu _{\text{log}}, \sigma _{\text{log}}| \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu _{\text{log}}, \sigma _{\text{log}}) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The standard deviation of log psi is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mean, std, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a LogNormalModel is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu _{\text{log}}, \sigma _{\text{log}}| \psi _i) =
            \log p(\psi _i |
            \mu _{\text{log}}, \sigma _{\text{log}}) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The standard deviation of log psi is strictly positive
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, std, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The standard deviation of log psi is strictly positive
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, std, observations)

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

        Parameters
        ----------
        mean_log
            Mean of :math:`\log \psi` in the population.
        std_log
            Standard deviation of :math:`\log \psi` in the population.
        """
        # Check input
        mean_log, std_log = parameters
        if std_log < 0:
            raise ValueError('The standard deviation cannot be negative.')

        # Compute mean and standard deviation
        mean = np.exp(mean_log + std_log**2 / 2)
        std = np.sqrt(
            np.exp(2 * mean_log + std_log**2) * (np.exp(std_log**2) - 1))

        return [mean, std]

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
            shape, _, scale = lognorm.fit(sample, floc=0)
        else:
            shape, _, scale = lognorm.fit(sample)
        sigma = shape
        mean  = np.log(scale)
        return mean, sigma

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mean, std = parameters

        if std <= 0:
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        # (Mean and sigma are the mean and standard deviation of
        # the log samples)
        rng = np.random.default_rng(seed=seed)
        samples = rng.lognormal(
            mean=mean, sigma=std, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mean log', 'Std. log']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class LogNormalModelRelativeSigma(PopulationModel):
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

    @staticmethod
    def _compute_log_likelihood(log_mean, stdRatio, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.
        """
        # Compute log-likelihood score
        log_std = log_mean*stdRatio
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * log_std**2) / 2 \
            - np.sum(np.log(observations)) \
            - np.sum((np.log(observations) - log_mean)**2) / 2 / log_std**2

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(log_mean, stdRatio, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Transform observations
        log_std = log_mean*stdRatio
        log_psi = np.log(observations)

        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * log_std**2) / 2 \
            - log_psi \
            - (log_psi - log_mean) ** 2 / (2 * log_std**2)

        return log_likelihood

    @staticmethod
    def _compute_sensitivities(log_mean, stdRatio, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        log_mean = float
        stdRatio = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        log_std = log_mean*stdRatio
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * log_std**2) / 2 \
            - np.sum(np.log(psi)) \
            - np.sum((np.log(psi) - log_mean)**2) / 2 / log_std**2

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = - ((np.log(psi) - log_mean) / log_std**2 + 1) / psi

        # Copmute sensitivities w.r.t. parameters
        dmean      = -n_obs/log_mean       +   np.sum(np.log(psi)/log_std**2 * (np.log(psi)/log_mean - 1))
        dstdRatio  = -n_obs/stdRatio   +   1/stdRatio**3 * (1 - np.log(psi)/log_mean)**2

        sensitivities = np.concatenate((dpsi, np.array([dmean, dstdRatio])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a LogNormalModel is the log-pdf evaluated
        at the observations

        .. math::
            L(\mu _{\text{log}}, \sigma _{\text{log}}| \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu _{\text{log}}, \sigma _{\text{log}}) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        log_mean, stdRatio = parameters

        if stdRatio <= 0:
            # The standard deviation of log psi is strictly positive
            return -np.inf

        return self._compute_log_likelihood(log_mean, stdRatio, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a LogNormalModel is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu _{\text{log}}, \sigma _{\text{log}}| \psi _i) =
            \log p(\psi _i |
            \mu _{\text{log}}, \sigma _{\text{log}}) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        log_mean, stdRatio = parameters

        if stdRatio <= 0:
            # The standard deviation of log psi is strictly positive
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(log_mean, stdRatio, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        log_mean, stdRatio = parameters

        if stdRatio <= 0:
            # The standard deviation of log psi is strictly positive
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(log_mean, stdRatio, observations)

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

        Parameters
        ----------
        mean_log
            Mean of :math:`\log \psi` in the population.
        std_log
            Standard deviation of :math:`\log \psi` in the population.
        """
        # Check input
        mean_log, stdRatio_log = parameters
        if stdRatio_log < 0:
            raise ValueError('The standard deviation cannot be negative.')

        # Compute mean and standard deviation
        std_log = stdRatio_log * mean_log
        mean = np.exp(mean_log + std_log**2 / 2)
        std = np.sqrt(
            np.exp(2 * mean_log + std_log**2) * (np.exp(std_log**2) - 1))

        return [mean, std]

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
            shape, _, scale = lognorm.fit(sample, floc=0)
        else:
            shape, _, scale = lognorm.fit(sample)
        log_sigma = shape
        log_mean  = np.log(scale)
        sigmaRatio = log_sigma/log_mean
        return log_mean, sigmaRatio

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        log_mean, stdRatio = parameters

        if stdRatio <= 0:
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        # (log_mean and sigma are the mean and standard deviation of
        # the log samples)
        log_std = stdRatio*log_mean
        rng = np.random.default_rng(seed=seed)
        samples = rng.lognormal(
            mean=log_mean, sigma=log_std, size=sample_shape) #this'll return entries that look like e^log_mean

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mean log', 'Rel. Std. log']
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
    """

    def __init__(self):
        super(PooledModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 1

        # Set default parameter names
        self._parameter_names = ['Pooled']

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the unnormalised log-likelihood score of the population model.

        A pooled population model is a delta-distribution centred at the
        population model parameter. As a result the log-likelihood score
        is 0, if all individual parameters are equal to the population
        parameter, and :math:`-\infty` otherwise.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get the population parameter
        parameter = parameters[0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        observations = np.array(observations)
        mask = observations != parameter
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

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get the population parameter
        parameter = parameters[0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        log_likelihood = np.zeros(shape=len(observations))
        observations = np.array(observations)
        mask = observations != parameter
        log_likelihood[mask] = -np.inf

        return log_likelihood

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get the population parameter
        parameter = parameters[0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        observations = np.array(observations)
        n_obs = len(observations)
        mask = observations != parameter
        if np.any(mask):
            return -np.inf, np.full(shape=n_obs + 1, fill_value=np.inf)

        # Otherwise return 0
        return 0, np.zeros(shape=n_obs + 1)

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

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the underlying population
        distribution.

        For a PooledModel the input top-level parameters are copied
        ``n_samples`` and are returned.

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        samples = np.asarray(parameters)

        # If only one sample is wanted, return input parameter
        if n_samples is None:
            return samples

        # If more samples are wanted, broadcast input parameter to shape
        # (n_samples,)
        samples = np.broadcast_to(samples, shape=(n_samples,))
        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Pooled']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class ReducedPopulationModel(object):
    """
    A class that can be used to permanently fix model parameters of a
    :class:`PopulationModel` instance.

    This may be useful to explore simplified versions of a model.

    Parameters
    ----------
    population_model
        An instance of a :class:`PopulationModel`.
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
        self._parameter_names = population_model.get_parameter_names()

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters #pylint:disable=invalid-unary-operand-type
            parameters = self._fixed_params_values

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations)

        return score

    def compute_pointwise_ll(self, parameters, observations):
        """
        Returns the pointwise log-likelihood of the population model parameters
        for each observation.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters #pylint:disable=invalid-unary-operand-type
            parameters = self._fixed_params_values

        # Compute log-likelihood
        scores = self._population_model.compute_pointwise_ll(
            parameters, observations)

        return scores

    def compute_sensitivities(self, parameters, observations):
        """
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters #pylint:disable=invalid-unary-operand-type
            parameters = self._fixed_params_values

        # Compute log-likelihood and sensitivities
        score, sensitivities = self._population_model.compute_sensitivities(
            parameters, observations)

        if self._fixed_params_mask is None:
            return score, sensitivities

        # Filter sensitivities for fixed parameters
        n_obs = len(observations)
        mask = np.ones(n_obs + self._n_parameters, dtype=bool)
        mask[-self._n_parameters:] = ~self._fixed_params_mask #pylint:disable=invalid-unary-operand-type

        return score, sensitivities[mask]

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        Parameters
        ----------
        name_value_dict
            A dictionary with model parameter names as keys, and parameter
            values as values.
        """
        # Check type
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError) as e:
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.') from e

        # If population model does not have model parameters, break here
        if self._n_parameters == 0:
            return None

        # If no model parameters have been fixed before, instantiate a mask
        # and values
        if self._fixed_params_mask is None:
            self._fixed_params_mask = np.zeros(
                shape=self._n_parameters, dtype=bool)

        if self._fixed_params_values is None:
            self._fixed_params_values = np.empty(shape=self._n_parameters)

        # Update the mask and values
        for index, name in enumerate(self._parameter_names):
            try:
                value = name_value_dict[name]
                if hasattr(value, "__len__"):
                    raise ValueError("Value for param %s has a length. Is this a mechanistic parameter?"%name)
            except KeyError:
                # KeyError indicates that parameter name is not being fixed
                continue

            # Fix parameter if value is not None, else unfix it
            self._fixed_params_mask[index] = value is not None
            self._fixed_params_values[index] = value

        # If all parameters are free, set mask and values to None again
        if np.alltrue(~self._fixed_params_mask): #pylint:disable=invalid-unary-operand-type
            self._fixed_params_mask = None
            self._fixed_params_values = None

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        # Remove fixed model parameters
        names = self._parameter_names
        if self._fixed_params_mask is not None:
            names = np.array(names)
            names = names[~self._fixed_params_mask] #pylint:disable=invalid-unary-operand-type
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

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the underlying population distribution.

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
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters #pylint:disable=invalid-unary-operand-type
            parameters = self._fixed_params_values

        # Sample from population model
        sample = self._population_model.sample(parameters, n_samples, seed)

        return sample

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
            self._parameter_names = \
                self._population_model.get_parameter_names()
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
            names[~self._fixed_params_mask] = parameter_names #pylint:disable=invalid-unary-operand-type
            parameter_names = names

        # Set parameter names
        self._population_model.set_parameter_names(parameter_names)
        self._parameter_names = self._population_model.get_parameter_names()


class TruncatedGaussianModel(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution which is truncated at
    zero.

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
    """

    def __init__(self):
        super(TruncatedGaussianModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mu', 'Sigma']

    @staticmethod
    def _compute_log_likelihood(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.

        We are using the relationship between the Gaussian CDF and the
        error function

        ..math::
            Phi(x) = (1 + erf(x/sqrt(2))) / 2
        """
        # Compute log-likelihood score
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum((observations - mean) ** 2) / (2 * std**2) \
            - n_ids * np.log(1 - _norm_cdf(-mean/std))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2) \
            - np.log(1 - math.erf(-mean/std/math.sqrt(2))) + np.log(2)

        return log_likelihood

    @staticmethod
    def _compute_sensitivities(mean, std, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        std = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * (np.log(2 * np.pi) / 2 + np.log(std)) \
            - np.sum((psi - mean)**2) / (2 * std**2) \
            - n_ids * np.log(1 - _norm_cdf(-mean/std))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mean - psi) / std**2

        # Copmute sensitivities w.r.t. parameters
        dmean = (
            np.sum(psi - mean) / std
            - _norm_pdf(mean/std) / (1 - _norm_cdf(-mean/std)) * n_ids
            ) / std
        dstd = (
            -n_ids + np.sum((psi - mean)**2) / std**2
            + _norm_pdf(mean/std) * mean / std / (1 - _norm_cdf(-mean/std))
            * n_ids
            ) / std

        sensitivities = np.concatenate((dpsi, np.array([dmean, dstd])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a truncated Gaussian distribution is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu , \sigma | \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if (mean <= 0) or (std <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return -np.inf

        return self._compute_log_likelihood(mean, std, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a truncated Gaussian distribution is
        the log-pdf evaluated at the observations

        .. math::
            L(\mu , \sigma | \psi _i) =
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if (mean <= 0) or (std <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, std, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if (mean <= 0) or (std <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, std, observations)

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

        Parameters
        ----------
        mu
            Mean of untruncated Gaussian distribution.
        sigma
            Standard deviation of untruncated Gaussian distribution.
        """
        # Check input
        mu, sigma = parameters
        if (mu < 0) or (sigma < 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            raise ValueError(
                'The parameters mu and sigma cannot be negative.')

        # Compute mean and standard deviation
        mean = mu + sigma * norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma))
        std = np.sqrt(
            sigma**2 * (
                1 -
                mu / sigma * norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma))
                - (norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma)))**2)
            )

        return [mean, std]

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

    def reverse_sample(self, sample, fa=0, fb=np.inf, fast=True):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        fa
            A float for the lower bound of the truncated Gaussian. Defaults to 0.
        fb
            A float for the upper bound of the truncated Gaussian. Defaults to np.inf.
        """
        if fast:
            mean, std = np.mean(sample), np.std(sample)
        else:
            mean, std = truncnorm.fit(sample, fa=fa, fb=fb)[2:]
        return mean, std

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mu, sigma = parameters

        if (mu < 0) or (sigma < 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            raise ValueError(
                'A truncated Gaussian distribution only accepts strictly '
                'positive means and standard deviations.')

        # Convert seed to int if seed is a rng
        # (Unfortunately truncated normal is not yet available with numpys
        # random number generator API)
        if isinstance(seed, np.random.Generator):
            # Draw new seed such that rng is propagated, but truncated normal
            # samples can also be seeded.
            seed = seed.integers(low=0, high=1E6)
        np.random.seed(seed)

        # Sample from population distribution
        samples = truncnorm.rvs(
            a=0, b=np.inf, loc=mu, scale=sigma, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mu', 'Sigma']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class TruncatedGaussianModelRelativeSigma(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Truncated Gaussian distribution.

    The difference between this model and TruncatedGaussianModel is that the standard
    deviation :math:`\sigma` is calculated as the product of the two input params,
    :math:`\mu` and :math:`\frac{\sigma}{\mu}`

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(TruncatedGaussianModelRelativeSigma, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mu', 'Rel. Sigma']

    @staticmethod
    def _compute_log_likelihood(mean, stdRatio, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.

        We are using the relationship between the Gaussian CDF and the
        error function

        ..math::
            Phi(x) = (1 + erf(x/sqrt(2))) / 2
        """
        # Compute log-likelihood score
        std = mean*stdRatio
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum((observations - mean) ** 2) / (2 * std**2) \
            - n_ids * np.log(1 - _norm_cdf(-mean/std))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, stdRatio, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        std = mean*stdRatio
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2) \
            - np.log(1 - math.erf(-mean/std/math.sqrt(2))) + np.log(2)

        return log_likelihood

    @staticmethod
    def _compute_sensitivities(mean, stdRatio, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        std = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        std = mean*stdRatio
        log_likelihood = \
            - n_ids * (np.log(2 * np.pi) / 2 + np.log(std)) \
            - np.sum((psi - mean)**2) / (2 * std**2) \
            - n_ids * np.log(1 - _norm_cdf(-mean/std))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mean - psi) / std**2

        # Copmute sensitivities w.r.t. parameters
        dmean = -n_ids/mean       +   1/std**2      * np.sum(psi*(psi/mean-1)) 
        dRatio = \
            -n_ids/stdRatio   +   1/stdRatio**3 * np.sum((psi/mean-1)**2) \
            - n_ids/stdRatio**2 * _norm_pdf(mean/std) / (1 - _norm_cdf(-mean/std))

        sensitivities = np.concatenate((dpsi, np.array([dmean, dRatio])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a truncated Gaussian distribution is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu , \sigma | \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, stdRatio = parameters

        if (mean <= 0) or (stdRatio <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return -np.inf

        return self._compute_log_likelihood(mean, stdRatio, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a truncated Gaussian distribution is
        the log-pdf evaluated at the observations

        .. math::
            L(\mu , \sigma | \psi _i) =
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, stdRatio = parameters

        if (mean <= 0) or (stdRatio <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, stdRatio, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, stdRatio = parameters

        if (mean <= 0) or (stdRatio <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, stdRatio, observations)

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

        Parameters
        ----------
        mu
            Mean of untruncated Gaussian distribution.
        sigma
            Standard deviation of untruncated Gaussian distribution.
        """
        # Check input
        mu, sigmaRatio = parameters
        if (mu < 0) or (sigmaRatio < 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            raise ValueError(
                'The parameters mu and sigma cannot be negative.')
        sigma = mu*sigmaRatio

        # Compute mean and standard deviation
        mean = mu + sigma * norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma))
        std = np.sqrt(
            sigma**2 * (
                1 -
                mu / sigma * norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma))
                - (norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma)))**2)
            )

        return [mean, std]

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

    def reverse_sample(self, sample, fa=0, fb=np.inf, fast=True):
        r"""
        Returns an estimate of the population model parameters given a sample.

        The returned value is a NumPy array with shape ``(n_parameters)``.

        Parameters
        ----------
        sample
            An array-like object with a sample from the population model.
        fa
            A float for the lower bound of the truncated Gaussian. Defaults to 0.
        fb
            A float for the upper bound of the truncated Gaussian. Defaults to np.inf.
        """
        if fast:
            mean, std = np.mean(sample), np.std(sample)
        else:
            mean, std = truncnorm.fit(sample, fa=fa, fb=fb)[2:]
        stdRatio = std/mean
        return mean, stdRatio

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
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mu, sigmaRatio = parameters

        if (mu < 0) or (sigmaRatio < 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            raise ValueError(
                'A truncated Gaussian distribution only accepts strictly '
                'positive means and standard deviations.')

        # Convert seed to int if seed is a rng
        # (Unfortunately truncated normal is not yet available with numpys
        # random number generator API)
        if isinstance(seed, np.random.Generator):
            # Draw new seed such that rng is propagated, but truncated normal
            # samples can also be seeded.
            seed = seed.integers(low=0, high=1E6)
        np.random.seed(seed)

        # Sample from population distribution
        sigma = mu*sigmaRatio
        samples = truncnorm.rvs(
            a=0, b=np.inf, loc=mu, scale=sigma, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mu', 'Rel. Sigma']
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

    def __init__(self):
        super(UniformModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 0

        # Set default parameter names
        self._parameter_names = None

        #Uninitialised boundaries
        self._boundaries, self._value = None, None

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
    return 0.5 * (1 + math.erf(x/math.sqrt(2)))


def _norm_pdf(x):  # pragma: no cover
    """
    Returns the probability density function value of a standard normal
    Gaussian distribtion.
    """
    return math.exp(-x**2/2) / math.sqrt(2 * math.pi)
