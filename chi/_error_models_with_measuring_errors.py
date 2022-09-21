#pylint:disable=invalid-unary-operand-type

from copy import copy
import numpy as np

from ._error_models import (  # noqa
    ConstantAndMultiplicativeGaussianErrorModel,
    ErrorModel,
    GaussianErrorModel,
    LogNormalErrorModel,
    MultiplicativeGaussianErrorModel,
    NaiveLogNormalErrorModel,
    ReducedErrorModel
)

class ErrorModelWithMeasuringErrors(ErrorModel):
    '''
    A base class for error models that take an additional, minimum source of error due to provided measurement errors. The measurement errors are assumed to follow the same error model as the unknown source of error. The combined log-likelihood function is logL_measurement + logL_error_model.

    Parameters:
    - error_model: an error model in the syntax defined by Chi.
    - observation_errors: measuring errors corresponding to observations. It must have the same length as the observations and model_output given to compute_log_likelihood, and thus must already be masked (see _log_pdfs.LogLikelihood._arange_times_for_mechanistic_model).
    '''
    def __init__(self, error_model):
        super(ErrorModelWithMeasuringErrors, self).__init__()

        # Check input
        if not isinstance(error_model, ErrorModel):
            raise ValueError(
                'The error model has to be an instance of a '
                'chi.ErrorModel')

        self._error_model = error_model

        # Set values belonging to the error_model
        self._parameter_names           = self._error_model._parameter_names
        self._n_parameters              = self._error_model._n_parameters
        self._normalised_log_likelihood = self._error_model._normalised_log_likelihood

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        observationErrors
            An array-like object with the measuring errors of the observations.
        """
        raise NotImplementedError

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value. :math:`\psi` and
        :math:`\sigma` are the parameters of the mechanistic model and the
        error model, respectively.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        observationErrors
            An array-like object with the measuring errors of the observations.
        """
        raise NotImplementedError

    @staticmethod
    def _nansum(pointwise_log_likelihood): # pragma: no cover
        r"""
            Returns np.nansum of the input. Intended for use with compute_pointwise_ll().
        """
        return np.nansum(pointwise_log_likelihood, axis=0)

    def compute_normalised_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), normalised by its value
            when the model output is equal to x^{\text{obs}} + \sigma or an equivalent expression, depending on the form
            of the log likelihood.
        """
        pLL = self.compute_normalised_pointwise_ll(parameters, model_output, observations, observationErrors)
        return self._nansum(pLL)

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations.
        """
        raise NotImplementedError


    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi} \quad \text{and} \quad
            \frac{\partial L}{\partial \sigma},

        where both :math:`\psi` and :math:`\sigma` can be multi-dimensional.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        :param observationErrors:
            An array-like object with the measuring errors of the observations.
        :type observationErrors: list, numpy.ndarray of length t
        """
        raise NotImplementedError

    def get_error_model(self):
        """
        Returns the original error model.
        """
        return self._error_model

    def sample(self, parameters, model_output, n_samples=None, seed=None):
        """
        Returns samples from the mechanistic model-error model pair in form
        of a NumPy array of shape ``(len(model_output), n_samples)``.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`.
        n_samples
            Number of samples from the error model for each entry in
            ``model_output``. If ``None``, one sample is assumed.
        seed
            Seed for the pseudo-random number generator. If ``None``, the
            pseudo-random number generator is not seeded.
        """
        return self._error_model.sample(parameters, model_output, n_samples, seed)
        #TODO: Sample with measuring errors too, instead?

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        return self._error_model.set_parameter_names(names)

    def get_parameter_names(self):
        return self._error_model.get_parameter_names()

    def n_parameters(self):
        return self._error_model.n_parameters()


class ConstantAndMultiplicativeGaussianErrorModelWithMeasuringErrors(ErrorModelWithMeasuringErrors):
    r"""
    See the documentation for MultiplicativeGaussianErrorModel.
    Here, it is expected that observationErrors = error_base + obsrevations*error_rel, which must be calculated externally.
    """
    def __init__(self, error_model):
        super(ConstantAndMultiplicativeGaussianErrorModelWithMeasuringErrors, self).__init__(error_model)
        assert isinstance(error_model, ConstantAndMultiplicativeGaussianErrorModel)

    @staticmethod
    def _compute_log_likelihood(parameters, model_output, observations, observationErrors):
        """
        Calculates the log-likelihood using numba speed up.
        """
        # Get parameters
        sigma_p_base, sigma_p_rel = parameters

        if sigma_p_base <= 0 or sigma_p_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation from parameters
        sigma_params = sigma_p_base + np.abs(sigma_p_rel * model_output)
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot)) \
            - np.sum(((model_output - observations) / sigma_tot)**2) / 2

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(parameters, model_output, observations, observationErrors):
        """
        Calculates the pointwise log-likelihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_p_base, sigma_p_rel = parameters

        if sigma_p_base <= 0 or sigma_p_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation from parameters
        sigma_params = sigma_p_base + np.abs(sigma_p_rel * model_output)
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 \
            - np.log(sigma_tot) \
            - ((model_output - observations) / sigma_tot)**2 / 2

        return pointwise_ll

    @staticmethod
    def _compute_sensitivities(parameters, model_output, model_sensitivities, observations, observationErrors):
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        Shape observationErrors =  (n_obs, 1)
        """

        # Get parameters
        sigma_p_base, sigma_p_rel = parameters

        if sigma_p_base <= 0 or sigma_p_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            n_parameters = model_sensitivities.shape[1] + 2
            return -np.inf, np.full(n_parameters, np.inf)


        # Compute total standard deviation from parameters
        sigma_params = sigma_p_base + np.abs(sigma_p_rel * model_output)
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute error and squared error
        yobsMinusYest = (observations - model_output)
        sqDiffOverErr = (yobsMinusYest/sigma_tot)**2

        # Compute log-likelihood due to measuring errors
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot)) \
            - np.nansum(sqDiffOverErr, axis=0) / 2
        log_likelihood = log_likelihood[0]

        # Compute sensitivities
        dpsi = \
            np.sum(sqDiffOverErr * model_sensitivities, axis=0) \
            - sigma_p_rel * np.sum(model_sensitivities / sigma_tot, axis=0) \
            + sigma_p_rel * np.sum(
                sqDiffOverErr / sigma_tot * model_sensitivities, axis=0) #n.b. (yest-yobs)^2/sigma^3
        dsigma_base = \
            np.sum(yobsMinusYest**2 * model_output    * sigma_params / sigma_tot_sq**2, axis=0) \
            - np.sum(sigma_params / sigma_tot_sq, axis=0)
        dsigma_rel = \
            np.sum(yobsMinusYest**2 * model_output**2 * sigma_params / sigma_tot_sq**2, axis=0) \
            - np.sum(model_output * sigma_params / sigma_tot_sq, axis=0)
        sensitivities = np.concatenate((dpsi, dsigma_base, dsigma_rel))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        observationErrors
            An array-like object with the measuring error of the observations.
        """
        if self._normalised_log_likelihood:
            return self.compute_normalised_log_likelihood(parameters, model_output, observations, observationErrors)

        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs, obsErr)

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs, obsErr)

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations. In this case,

            .. math::
                p(x | \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
                \frac{|\bar{x}|}{\sqrt{2\pi} \sigma _{\text{tot}}}
                \exp{\left(-\frac{\left(x-x^{\text{m}}\right) ^2}
                {2\sigma^2 _{\text{tot}}} \right)},
        """
        #Check inputs
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        # Get normalised likelihood
        L = self._compute_pointwise_ll(parameters, model, obs, obsErr)
        N = np.abs(np.nanmean(obs)) if n_observations>0 else 1

        return (L + np.log(N)) if N>0 else L

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\text{base}}}, \quad
            \frac{\partial L}{\partial \sigma _{\text{rel}}}.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 2
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        obsErr = np.asarray(observationErrors).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs, obsErr)


class GaussianErrorModelWithMeasuringErrors(ErrorModelWithMeasuringErrors):
    r"""
    See the documentation for MultiplicativeGaussianErrorModel.
    Here, it is expected that observationErrors are the measuring error (standard deviation) on observations.
    """
    def __init__(self, error_model):
        super(GaussianErrorModelWithMeasuringErrors, self).__init__(error_model)
        assert isinstance(error_model, GaussianErrorModel)

    @staticmethod
    def _compute_log_likelihood(parameters, model_output, observations, observationErrors):
        """
        Calculates the log-likelihood using numba speed up.
        """
        # Get parameters
        sigma_params = parameters[0]

        if sigma_params <= 0:
            # sigma is strictly positive
            return -np.inf

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot)) \
            - np.sum(((model_output - observations) / sigma_tot)**2) / 2

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(parameters, model_output, observations, observationErrors):
        """
        Calculates the pointwise log-likelihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_params = parameters[0]

        if sigma_params <= 0:
            # sigma is strictly positive
            n_obs = len(model_output)
            return np.full(n_obs, -np.inf)

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 \
            - np.log(sigma_tot) \
            - ((model_output - observations) / sigma_tot)**2 / 2

        return pointwise_ll

    @staticmethod
    def _compute_sensitivities(parameters, model_output, model_sensitivities, observations, observationErrors):
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        Shape observationErrors =  (n_obs, 1)
        """

        # Get parameters
        sigma_params = parameters[0]

        if sigma_params <= 0:
            # sigma is strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params   +   observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute error and squared error
        yobsMinusYest = (observations - model_output)**2
        sqDiffOverErr = (yobsMinusYest/sigma_tot)**2

        # Compute log-likelihood due to measuring errors
        n_obs = len(model_output)
        sumLogSigmaTot = np.sum(np.log(sigma_tot))
        measure_log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - sumLogSigmaTot \
            - np.sum(sqDiffOverErr**2, axis=0) / 2
        measure_log_likelihood = measure_log_likelihood[0]

        # Compute sensitivities
        dpsi = np.sum(yobsMinusYest/sigma_tot**2 * model_sensitivities, axis=0)
        dsigma = sigma_params*np.sum(yobsMinusYest**2 / sigma_tot_sq**2, axis=0) - np.sum(sigma_params/sigma_tot_sq, axis=0)
        measure_sensitivities = np.concatenate((dpsi, dsigma))

        return measure_log_likelihood, measure_sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma | x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma ) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`x^{\text{m}}`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        observationErrors
            An array-like object with the measuring errors of the observations.
        """
        if self._normalised_log_likelihood:
            return self.compute_normalised_log_likelihood(parameters, model_output, observations, observationErrors)

        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs, obsErr)

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        observationErrors
            An array-like object with the measuring errors of the observations.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs,obsErr)

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations. In this case,

            .. math::
                p(x | \psi , \sigma ) =
                \frac{|\bar{x}|}{\sqrt{2\pi} \sigma }
                \exp{\left(-\frac{\left(x-x^{\text{m}}\right) ^2}
                {2\sigma^2 } \right)}.
        """
        #Check inputs
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        # Get normalised likelihood
        L = self._compute_pointwise_ll(parameters, model, obs, obsErr)
        N = np.abs(np.nanmean(obs)) if n_observations>0 else 1

        return (L + np.log(N)) if N>0 else L

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma }.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        :param observationErrors:
            An array-like object with the measuring errors of the observations.
        :type observationErrors: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        obsErr = np.asarray(observationErrors).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs, obsErr)


class LogNormalErrorModelWithMeasuringErrors(ErrorModelWithMeasuringErrors):
    r"""
    See the documentation for LogNormalErrorModel. Here, it is expected that observationErrors are the measuring error
    (standard deviation) of the log of the observations. If only the measuring error is known, observationErrors might
    be approximated as measuringError/measuredValue.

    .. math::
        X(t, \psi , \theta _{\mathrm{log}}) =
        y \, \mathrm{e}^{\mu + \theta _{\mathrm{log}} \varepsilon },

    as defined in the documentation for LogNormalErrorModel, but where :math:`\theta _{\mathrm{log}}` is the measuring
    error of :math:`\log X`. Here, we assume that :math:`\theta _{\mathrm{log}} = \theta / X.`
    """
    def __init__(self, error_model):
        super(LogNormalErrorModelWithMeasuringErrors, self).__init__(error_model)
        assert isinstance(error_model, LogNormalErrorModel)

    @staticmethod
    def _compute_log_likelihood(parameters, model_output, observations, observationErrors):
        """
        Calculates the log-likelihood using numba speed up.
        """
        # Get parameters
        sigma_params = parameters[0]

        if (sigma_params <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            return -np.inf

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot))   -   np.nansum(np.log(observations)) \
            - np.nansum((
                    (np.log(model_output)   -   np.log(observations)   -   sigma_tot**2 / 2) / sigma_tot
                )**2
            ) / 2

        return log_likelihood
        
    @staticmethod
    def _compute_pointwise_ll(parameters, model_output, observations, observationErrors):
        """
        Calculates the pointwise log-lieklihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_params = parameters[0]

        if (sigma_params <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            return -np.inf

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 \
            - np.log(sigma_tot)   -   np.log(observations) \
            - ((
                    (np.log(model_output)   -   np.log(observations)   -   sigma_tot**2 / 2) / sigma_tot
                )**2
            ) / 2

        return pointwise_ll
        
    @staticmethod
    def _compute_sensitivities(parameters, model_output, model_sensitivities, observations, observationErrors):
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        Shape observationErrors =  (n_obs, 1)
        """
        # Get parameters
        sigma_params = parameters[0]

        if (sigma_params <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute error and squared error
        yobsMinusYestMinusMu = np.log(observations) - np.log(model_output) + sigma_tot**2 / 2
        yDiffOverErrSq = (yobsMinusYestMinusMu/sigma_tot)**2

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot))   -   np.nansum(np.log(observations)) \
            - np.nansum(yDiffOverErrSq, axis=0) / 2
        log_likelihood = log_likelihood[0]

        # Compute sensitivities
        dpsi = \
            np.sum(yobsMinusYestMinusMu / model_output * model_sensitivities, axis=0) \
            / sigma_tot**2
        dsigma = \
            - np.sum(sigma_params/sigma_tot_sq) \
            - sigma_params * np.sum(yobsMinusYestMinusMu/sigma_tot_sq) \
            + sigma_params * np.sum(yobsMinusYestMinusMu**2 / sigma_tot_sq**2 )
        sensitivities = np.concatenate((dpsi, dsigma))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`y` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\mathrm{log}} | x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\mathrm{log}} ) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`y`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        """
        if self._normalised_log_likelihood:
            return self.compute_normalised_log_likelihood(parameters, model_output, observations, observationErrors)

        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs, obsErr)

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\mathrm{log}} | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\mathrm{log}} ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs, obsErr)

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations. In this case,

            .. math::
                p(x | \psi , \sigma _{\mathrm{log}} ) =
                \frac{1}{\sqrt{2\pi} \sigma _{\mathrm{log}}}
                \exp{\left(-\frac{
                    \left(\log x - \log y + \sigma _{\mathrm{log}}^2/2\right) ^2}
                {2\sigma _{\mathrm{log}}^2 } \right)}.
        """
        #Check inputs
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        # Get normalised likelihood
        L = self._compute_pointwise_ll(parameters, model, obs, obsErr)

        return L + np.log(obs)

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\mathrm{log}} }.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        obsErr = np.asarray(observationErrors).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs, obsErr)


class MultiplicativeGaussianErrorModelWithMeasuringErrors(ErrorModelWithMeasuringErrors):
    r"""
    See the documentation for MultiplicativeGaussianErrorModel.
    Here, it is expected that observationErrors = observations*error_rel, which must be calculated externally.
    """
    def __init__(self, error_model):
        super(MultiplicativeGaussianErrorModelWithMeasuringErrors, self).__init__(error_model)
        assert isinstance(error_model, MultiplicativeGaussianErrorModel)

    @staticmethod
    def _compute_log_likelihood(parameters, model_output, observations, observationErrors):
        """
        Calculates the log-likelihood using numba speed up.
        """
        # Get parameters
        sigma_p_rel = parameters

        if sigma_p_rel <= 0:
            # sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation from parameters
        sigma_params = np.abs(sigma_p_rel * model_output)
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot)) \
            - np.sum((model_output - observations)**2 / sigma_tot**2) / 2

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(parameters, model_output, observations, observationErrors):
        """
        Calculates the pointwise log-likelihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_p_rel = parameters

        if sigma_p_rel <= 0:
            # sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation from parameters
        sigma_params = np.abs(sigma_p_rel * model_output)
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 \
            - np.log(sigma_tot) \
            - (model_output - observations)**2 / sigma_tot**2 / 2

        return pointwise_ll

    @staticmethod
    def _compute_sensitivities(parameters, model_output, model_sensitivities, observations, observationErrors):
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        Shape observationErrors =  (n_obs, 1)
        """
        # Get parameters
        sigma_p_rel = parameters

        if sigma_p_rel <= 0:
            # sigma_rel are strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        # Compute total standard deviation from parameters
        sigma_params = np.abs(sigma_p_rel * model_output)
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute error and squared error
        error = observations - model_output
        squared_error = error**2

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot)) \
            - np.sum(squared_error / sigma_tot_sq) / 2

        # Compute sensitivities
        dpsi = \
            np.sum(
                error / sigma_tot_sq * model_sensitivities, axis=0) \
            - sigma_p_rel * np.sum(model_sensitivities / sigma_tot, axis=0) \
            + sigma_p_rel * np.sum(
                squared_error / sigma_tot**3 * model_sensitivities, axis=0)
        dsigma_rel = \
            np.sum(squared_error * sigma_p_rel * model_output**2 / sigma_tot_sq**2, axis=0) \
            - np.sum(sigma_p_rel*model_output**2 / sigma_tot_sq, axis=0)
        sensitivities = np.concatenate((dpsi, dsigma_rel))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\text{rel}} |
            x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{rel}}) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`x^{\text{m}}`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        """
        if self._normalised_log_likelihood:
            return self.compute_normalised_log_likelihood(parameters, model_output, observations, observationErrors)

        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_obs = len(observations)
        if len(model) != n_obs:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs, obsErr)

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs, obsErr)

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations. In this case,

            .. math::
                p(x | \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
                \frac{|\bar{x}|}{\sqrt{2\pi} \sigma _{\text{tot}}}
                \exp{\left(-\frac{\left(x-x^{\text{m}}\right) ^2}
                {2\sigma^2 _{\text{tot}}} \right)},
        """
        #Check inputs
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        # Get normalised likelihood
        L = self._compute_pointwise_ll(parameters, model, obs, obsErr)
        N = np.abs(np.nanmean(obs)) if n_observations>0 else 1

        return (L + np.log(N)) if N>0 else L

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\text{rel}}}.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        obsErr = np.asarray(observationErrors).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs, obsErr)


class NaiveLogNormalErrorModelWithMeasuringErrors(ErrorModelWithMeasuringErrors):
    r"""
    See the documentation for NaiveLogNormalErrorModel. Here, it is expected that observationErrors are the measuring error
    (standard deviation) of the log of the observations. If only the measuring error is known, observationErrors might
    be approximated as measuringError/measuredValue.

    .. math::
        X(t, \psi , \theta _{\mathrm{log}}) =
        y \, \mathrm{e}^{\mu + \theta _{\mathrm{log}} \varepsilon },

    as defined in the documentation for NaiveLogNormalErrorModel, but where :math:`\theta _{\mathrm{log}}` is the measuring
    error of :math:`\log X`. Here, we assume that :math:`\theta _{\mathrm{log}} = \theta / X.`
    """
    def __init__(self, error_model):
        super(NaiveLogNormalErrorModelWithMeasuringErrors, self).__init__(error_model)
        assert isinstance(error_model, NaiveLogNormalErrorModel)

    @staticmethod
    def _compute_log_likelihood(parameters, model_output, observations, observationErrors):
        """
        Calculates the log-likelihood using numba speed up.
        """
        # Get parameters
        sigma_params = parameters[0]

        if (sigma_params <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            return -np.inf

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot))  \
            - np.nansum((
                    ( np.log(model_output)   -   np.log(observations) ) / sigma_tot
                )**2
            ) / 2

        return log_likelihood
        
    @staticmethod
    def _compute_pointwise_ll(parameters, model_output, observations, observationErrors):
        """
        Calculates the pointwise log-lieklihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_params = parameters[0]

        if (sigma_params <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            return -np.inf

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 - np.log(sigma_tot) \
            - ((
                    ( np.log(model_output)   -   np.log(observations) ) / sigma_tot
                )**2
            ) / 2

        return pointwise_ll
        
    @staticmethod
    def _compute_sensitivities(parameters, model_output, model_sensitivities, observations, observationErrors):
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        Shape observationErrors =  (n_obs, 1)
        """
        # Get parameters
        sigma_params = parameters[0]

        if (sigma_params <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        #Get total sigma
        sigma_tot_sq = sigma_params*sigma_params + observationErrors*observationErrors
        sigma_tot = np.sqrt(sigma_tot_sq)

        # Compute error and squared error
        yobsMinusYest = np.log(observations) - np.log(model_output)
        yDiffOverErrSq = (yobsMinusYest/sigma_tot)**2

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.nansum(np.log(sigma_tot))  \
            - np.nansum(yDiffOverErrSq, axis=0) / 2
        log_likelihood = log_likelihood[0]

        # Compute sensitivities
        dpsi = \
            np.sum(yobsMinusYest / model_output * model_sensitivities, axis=0) \
            / sigma_tot**2
        dsigma = \
            - np.sum(sigma_params/sigma_tot_sq) \
            + sigma_params * np.sum(yobsMinusYest**2 / sigma_tot_sq**2 )
        sensitivities = np.concatenate((dpsi, dsigma))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`y` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\mathrm{log}} | x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\mathrm{log}} ) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`y`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        """
        if self._normalised_log_likelihood:
            return self.compute_normalised_log_likelihood(parameters, model_output, observations, observationErrors)

        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs, obsErr)

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\mathrm{log}} | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\mathrm{log}} ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs, obsErr)

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations. In this case, nothing has units of observations, so the likelihood is unchanged.
        """
        #Check inputs
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        obsErr = np.asarray(observationErrors)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        # Get normalised likelihood
        return self._compute_pointwise_ll(parameters, model, obs, obsErr)

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\mathrm{log}} }.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        obsErr = np.asarray(observationErrors).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs, obsErr)


class ReducedErrorModelWithMeasuringErrors(ReducedErrorModel):
    """
    A class that can be used to permanently fix model parameters of an
    :class:`ErrorModelWithMeasuringErrors` instance.

    This may be useful to explore simplified versions of a model.

    Parameters
    ----------
    error_model
        An instance of a :class:`ErrorModelWithMeasuringErrors`.
    _setDefaults
        Initialises internal variables. Default True. This should not be set to False unless another method handles
        the internal variables.
    """
    def __init__(self, error_model):
        # super().__init__(error_model)

        # Check input
        if not isinstance(error_model, ErrorModelWithMeasuringErrors):
            raise ValueError(
                'The error model has to be an instance of a '
                'chi.ErrorModelWithMeasuringErrors')

        self._error_model = error_model

        # Set defaults
        self._fixed_params_mask = None
        self._fixed_params_values = None
        self._n_parameters = error_model.n_parameters()
        self._parameter_names = error_model.get_parameter_names()

    #Alternate constructor
    # @classmethod
    # def init_from_reduced_error_model(self, error_model):
    #     # Check input
    #     if not isinstance(error_model, ReducedErrorModel):
    #         raise ValueError(
    #             'The error model has to be an instance of a '
    #             'chi.ReducedErrorModel')

    #     #Copy the underlying error model, converting if necessary
    #     underlying_model = error_model._error_model
    #     self._error_model = \
    #         underlying_model if isinstance(underlying_model, ErrorModelWithMeasuringErrors) else \
    #         return_measuring_error_model_from_error_model(underlying_model)

    #     #Copy other arguments
    #     self._fixed_params_mask   = error_model._fixed_params_mask   #pylint: disable=protected-access
    #     self._fixed_params_values = error_model._fixed_params_values #pylint: disable=protected-access
    #     self._n_parameters        = error_model._n_parameters        #pylint: disable=protected-access
    #     self._parameter_names     = error_model._parameter_names     #pylint: disable=protected-access

    #     return self(self._error_model, _setDefaults=False)

    def compute_log_likelihood(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        observationErrors
            An array-like object with the measuring errors of the observations.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        score = self._error_model.compute_log_likelihood(
            parameters, model_output, observations, observationErrors)
        return score

    def compute_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value. :math:`\psi` and
        :math:`\sigma` are the parameters of the mechanistic model and the
        error model, respectively.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        observationErrors
            An array-like object with the measuring errors of the observations.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        pointwise_ll = self._error_model.compute_pointwise_ll(
            parameters, model_output, observations, observationErrors)
        return pointwise_ll

    def compute_normalised_pointwise_ll(self, parameters, model_output, observations, observationErrors):
        r"""
            Returns the log-likelihood of model parameters, L(\psi, \sigma | x^{\text{obs}}), with units of 1/observations multiplied out. All quantitaties with units of [observations], including model_output, are divided by the absolute value of the mean of the observations.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        score = self._error_model.compute_normalised_pointwise_ll(
            parameters, model_output, observations, observationErrors)
        return score

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations, observationErrors):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

        The sensitivities of the log-likelihood is defined as the partial
        derivative of the :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi} \quad \text{and} \quad
            \frac{\partial L}{\partial \sigma},

        where both :math:`\psi` and :math:`\sigma` should be interpreted
        as a collection of multiple parameters.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length p
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        :param observationErrors:
            An array-like object with the measuring errors of the observations.
        :type observationErrors: list, numpy.ndarray of length t
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        score, sensitivities = self._error_model.compute_sensitivities(
            parameters, model_output, model_sensitivities, observations, observationErrors)

        if self._fixed_params_mask is None:
            return score, sensitivities

        # Filter sensitivities for fixed parameters
        n_mechanistic = model_sensitivities.shape[1]
        mask = np.ones(n_mechanistic + self._n_parameters, dtype=bool)
        mask[-self._n_parameters:] = ~self._fixed_params_mask

        return score, sensitivities[mask]

def return_reduced_measuring_error_model_from_reduced_model(r_error_model):
    # Check input
    if not isinstance(r_error_model, ReducedErrorModel):
        raise ValueError(
            'The error model has to be an instance of a '
            'chi.ReducedErrorModel')

    #Copy the underlying error model, converting if necessary
    underlying_e_model = r_error_model._error_model
    underlying_me_model = \
        underlying_e_model if isinstance(underlying_e_model, ErrorModelWithMeasuringErrors) else \
        return_measuring_error_model_from_error_model(underlying_e_model)

    #make the reduced model
    r_me_model = ReducedErrorModelWithMeasuringErrors(underlying_me_model)

    #Copy other arguments
    r_me_model._fixed_params_mask   = r_error_model._fixed_params_mask   #pylint: disable=protected-access
    r_me_model._fixed_params_values = r_error_model._fixed_params_values #pylint: disable=protected-access
    r_me_model._n_parameters        = r_error_model._n_parameters        #pylint: disable=protected-access
    r_me_model._parameter_names     = r_error_model._parameter_names     #pylint: disable=protected-access

    return r_me_model

def return_measuring_error_model_from_error_model(error_model):
    """
    Given an error_model, returns a class that extends ErrorModelWithMeasuringErrors of the appropriate type. For
    example, if an instance of GaussianErrorModel is given, an instance of GaussianErrorModelWithMeasuringErrors is
    returned.
    """
    if isinstance(error_model, ErrorModelWithMeasuringErrors):
        return copy(error_model)
    if isinstance(error_model, ConstantAndMultiplicativeGaussianErrorModel):
        return ConstantAndMultiplicativeGaussianErrorModelWithMeasuringErrors(error_model)
    elif isinstance(error_model, GaussianErrorModel):
        return GaussianErrorModelWithMeasuringErrors(error_model)
    elif isinstance(error_model, LogNormalErrorModel):
        return LogNormalErrorModelWithMeasuringErrors(error_model)
    elif isinstance(error_model, MultiplicativeGaussianErrorModel):
        return MultiplicativeGaussianErrorModelWithMeasuringErrors(error_model)
    elif isinstance(error_model, NaiveLogNormalErrorModel):
        return NaiveLogNormalErrorModelWithMeasuringErrors(error_model)
    else:
        raise ValueError("Unknown error_model type %s"%(type(error_model)))
