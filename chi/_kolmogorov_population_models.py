import copy
import numpy as np
import chi
import pints

class KolmogorovSmirnovPopulationModel(chi.PopulationModel):
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
    return isinstance(population_model, (chi.HeterogeneousModel, KolmogorovSmirnovHeterogeneousModel))


def is_pooled_model(population_model):
    """Returns true if the given population_model is one of the pooled population models."""
    return isinstance(population_model, (chi.PooledModel, KolmogorovSmirnovPooledModel))


def is_uniform_model(population_model):
    """Returns true if the given population_model is one of the uniform population models."""
    return isinstance(population_model, (chi.UniformModel, KolmogorovSmirnovUniformModel))


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

