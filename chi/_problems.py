#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#
# The InverseProblem class is based on the SingleOutputProblem and
# MultiOutputProblem classes of PINTS (https://github.com/pints-team/pints/),
# which is distributed under the BSD 3-clause license.
#

import copy
from warnings import warn

import myokit
import numpy as np
import pandas as pd
import pints

import chi


class InverseProblem(object):
    """
    Represents an inference problem where a model is fit to a
    one-dimensional or multi-dimensional time series, such as measured in a
    PKPD study.

    Parameters
    ----------
    model
        An instance of a :class:`MechanisticModel`.
    times
        A sequence of points in time. Must be non-negative and increasing.
    values
        A sequence of single- or multi-valued measurements. Must have shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of points in
        ``times`` and ``n_outputs`` is the number of outputs in the model. For
        ``n_outputs = 1``, the data can also have shape ``(n_times, )``.
    """

    def __init__(self, model, times, values):

        # Check model
        if not isinstance(model, chi.MechanisticModel):
            raise ValueError(
                'Model has to be an instance of a chi.Model.'
            )
        self._model = model

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be increasing.')

        # Check values, copy so that they can no longer be changed
        values = np.asarray(values)
        if values.ndim == 1:
            np.expand_dims(values, axis=1)
        self._values = pints.matrix2d(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        # Check for correct shape
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values as a NumPy array of shape ``(n_times, n_outputs)``.
        """
        output = self._model.simulate(parameters, self._times)

        # The chi.Model.simulate method returns the model output as
        # (n_outputs, n_times). We therefore need to transponse the result.
        return output.transpose()

    def evaluateS1(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        The returned data is a tuple of NumPy arrays ``(y, y')``, where ``y``
        has shape ``(n_times, n_outputs)``, while ``y'`` has shape
        ``(n_times, n_outputs, n_parameters)``.
        *This method only works for problems whose model implements the
        :class:`ForwardModelS1` interface.*
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

    def times(self):
        """
        Returns this problem's times.
        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.
        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values


class ProblemModellingController(object):
    """
    A problem modelling controller which simplifies the model building process
    of a pharmacokinetic and pharmacodynamic problem.

    The class is instantiated with an instance of a :class:`MechanisticModel`
    and one instance of an :class:`ErrorModel` for each mechanistic model
    output.

    :param mechanistic_model: A mechanistic model for the problem.
    :type mechanistic_model: MechanisticModel
    :param error_models: A list of error models. One error model has to be
        provided for each mechanistic model output.
    :type error_models: list[ErrorModel]
    :param outputs: A list of mechanistic model output names, which can be used
        to map the error models to mechanistic model outputs. If ``None``, the
        error models are assumed to be ordered in the same order as
        :meth:`MechanisticModel.outputs`.
    :type outputs: list[str], optional
    """

    def __init__(self, mechanistic_model, error_models, outputs=None):
        super(ProblemModellingController, self).__init__()

        # Check inputs
        if not isinstance(mechanistic_model, chi.MechanisticModel) and chi.MechanisticModel not in type(mechanistic_model).__mro__:
            raise TypeError(
                'The mechanistic model has to be an instance of a '
                'chi.MechanisticModel.')

        if not isinstance(error_models, list):
            error_models = [error_models]

        for error_model in error_models:
            if not isinstance(error_model, chi.ErrorModel):
                raise TypeError(
                    'Error models have to be instances of a '
                    'chi.ErrorModel.')

        # Copy mechanistic model
        mechanistic_model = copy.deepcopy(mechanistic_model)

        # Set outputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        # Get number of outputs
        n_outputs = mechanistic_model.n_outputs()

        if len(error_models) != n_outputs:
            raise ValueError(
                'Wrong number of error models. One error model has to be '
                'provided for each mechanistic error model.')

        # Copy error models
        error_models = [copy.copy(error_model) for error_model in error_models]

        # Remember models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Set defaults
        self._population_models = None
        self._log_prior = None
        self._data = None
        self._dataErr = None
        self._dosing_regimens = None
        self._individual_fixed_param_dict = None

        # Set error models to un-normalised by default
        self.set_normalised_error_models(False)

        # Set parameter names and number of parameters
        self._set_error_model_parameter_names()
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

    def _clean_data(self, data, dose_key, dose_duration_key):
        """
        Makes sure that the data is formated properly.

        1. ids are strings
        2. time are numerics or NaN
        3. biomarkers are strings
        4. measurements are numerics or NaN
        5. dose are numerics or NaN
        6. duration are numerics or NaN
        """
        # Create container for data
        columns = [
            self._id_key, self._time_key, self._biom_key, self._meas_key]
        if dose_key is not None:
            columns += [dose_key]
        if dose_duration_key is not None:
            columns += [dose_duration_key]
        cleanData = pd.DataFrame(columns=columns)

        # Convert IDs to strings
        cleanData[self._id_key] = data[self._id_key].astype(
            "string")

        # Convert times to numerics
        cleanData[self._time_key] = pd.to_numeric(data[self._time_key])

        # Convert biomarkers to strings
        cleanData[self._biom_key] = data[self._biom_key].astype(
            "string")

        # Convert measurements to numerics
        cleanData[self._meas_key] = pd.to_numeric(data[self._meas_key])

        # Convert dose to numerics
        if dose_key is not None:
            cleanData[dose_key] = pd.to_numeric(
                data[dose_key])

        # Convert duration to numerics
        if dose_duration_key is not None:
            cleanData[dose_duration_key] = pd.to_numeric(
                data[dose_duration_key])

        return cleanData

    def _create_log_likelihoods(self, individual):
        """
        Returns a list of log-likelihoods, one for each individual in the
        dataset.
        """
        # Get IDs
        ids = self._ids
        if individual is not None:
            ids = [individual]

        # Create a likelihood for each individual
        log_likelihoods = []
        for individual in ids:
            # Set dosing regimen
            try:
                self._mechanistic_model.simulator.set_protocol(
                    self._dosing_regimens[individual])
            except TypeError:
                # TypeError is raised when applied regimens is still None,
                # i.e. no doses were defined by the datasets.
                pass

            #Set individually fixed parameters
            if self._individual_fixed_param_dict is not None and len(self._individual_fixed_param_dict)>0:
                try:
                    mechanistic_model = self._mechanistic_model.copy()
                    mechanistic_model.fix_parameters(self._individual_fixed_param_dict[individual])
                except TypeError:
                    pass
            else:
                mechanistic_model = self._mechanistic_model

            log_likelihood = self._create_log_likelihood(individual, mechanistic_model)
            if log_likelihood is not None:
                # If data exists for this individual, append to log-likelihoods
                log_likelihoods.append(log_likelihood)

        return log_likelihoods

    def _create_log_likelihood(self, individual, mechanistic_model=None):
        """
        Gets the relevant data for the individual and returns the resulting
        chi.LogLikelihood.
        """
        #get mechanistic_model
        if mechanistic_model is None:
            mechanistic_model = self._mechanistic_model
        
        # Flag for considering errors, too
        haveErrors = self._dataErr is not None

        # Get individuals data
        times = []
        observations = []
        observationErrors = []
        mask = self._data[self._id_key] == individual
        data = self._data[mask][
            [self._time_key, self._biom_key, self._meas_key]]

        # Get individual measurement errors on the data
        if haveErrors:
            maskE = self._dataErr[self._id_key] == individual
            assert (np.asarray(mask)==np.asarray(maskE)).all()
            dataErr = self._dataErr[mask][
                [self._time_key, self._biom_key, self._meas_key]]

        for output in mechanistic_model.outputs():
            # Mask data for biomarker
            biomarker = self._output_biomarker_dict[output]
            mask  = data[self._biom_key] == biomarker
            temp_df = data[mask]
            if haveErrors:
                maskE = dataErr[self._biom_key] == biomarker
                assert (np.asarray(mask)==np.asarray(maskE)).all()
                temp_ef = dataErr[mask]

            # Filter observations for non-NaN entries
            mask  = temp_df[self._meas_key].notnull()
            temp_df = temp_df[[self._time_key, self._meas_key]][mask]
            if haveErrors:
                maskE = temp_ef[self._meas_key].notnull()
                assert (np.asarray(mask)==np.asarray(maskE)).all()
                temp_ef = temp_ef[[self._time_key, self._meas_key]][mask]
            
            # Filter times for non-NaN entries
            mask  = temp_df[self._time_key].notnull()
            temp_df = temp_df[mask]
            if haveErrors:
                maskE = temp_ef[self._time_key].notnull()
                assert (np.asarray(mask)==np.asarray(maskE)).all()
                temp_ef = temp_ef[mask]

            # Collect data for output
            times.append(temp_df[self._time_key].to_numpy())
            observations.append(temp_df[self._meas_key].to_numpy())
            if haveErrors:
                observationErrors.append(temp_ef[self._meas_key].to_numpy())

        # Count outputs that were measured
        # TODO: copy mechanistic model and update model outputs.
        # (Useful for e.g. control group and dose group training)
        n_measured_outputs = 0
        for output_measurements in observations:
            if len(output_measurements) > 0:
                n_measured_outputs += 1

        # If no outputs were measured, do not construct a likelihood
        if n_measured_outputs == 0:
            return None

        # Create log-likelihood and set ID to individual
        if haveErrors:
            for model_id, error_model in enumerate(self._error_models):
                isMeas = isinstance(error_model, (
                    chi.ErrorModelWithMeasuringErrors, chi.ReducedErrorModelWithMeasuringErrors))
                if not isMeas:
                    if isinstance(error_model, chi.ReducedErrorModel):
                        self._error_models[model_id] = \
                            chi.ReducedErrorModelWithMeasuringErrors.init_from_reduced_error_model(error_model)
                    else:
                        self._error_models[model_id] = chi.return_measuring_error_model_from_error_model(error_model)
                
            log_likelihood = chi.LogLikelihoodWithMeasuringErrors(
                mechanistic_model, self._error_models, observations, observationErrors, times)
        else:
            log_likelihood = chi.LogLikelihood(
                mechanistic_model, self._error_models, observations, times)
        log_likelihood.set_id(individual)

        return log_likelihood

    def _initialise_individual_fixed_params(self):
        """
        Initialises a dictionary to contain parameters that are fixed for each patient parameter.
        """
        fixedParams = dict()
        for label in self._ids:
            fixedParams = dict()

        return fixedParams

    def _extract_dosing_regimens(self, dose_key, duration_key):
        """
        Converts the dosing regimens defined by the pandas.DataFrame into
        myokit.Protocols, and returns them as a dictionary with individual
        IDs as keys, and regimens as values.

        For each dose entry in the dataframe a dose event is added
        to the myokit.Protocol. If the duration of the dose is not provided
        a bolus dose of duration 0.01 time units is assumed.
        """
        # Create duration column if it doesn't exist and set it to default
        # bolus duration of 0.01
        if duration_key is None:
            duration_key = 'Duration in base time unit'
            self._data[duration_key] = 0.01

        # Extract regimen from dataset
        regimens = dict()
        for label in self._ids:
            # Filter times and dose events for non-NaN entries
            mask = self._data[self._id_key] == label
            data = self._data[
                [self._time_key, dose_key, duration_key]][mask]
            mask = data[dose_key].notnull()
            data = data[mask]
            mask = data[self._time_key].notnull()
            data = data[mask]

            # Add dose events to dosing regimen
            regimen = myokit.Protocol()
            for _, row in data.iterrows():
                # Set duration
                duration = row[duration_key]
                if np.isnan(duration):
                    # If duration is not provided, we assume a bolus dose
                    # which we approximate by 0.01 time_units.
                    duration = 0.01

                # Compute dose rate and set regimen
                dose_rate = row[dose_key] / duration
                time = row[self._time_key]
                regimen.add(myokit.ProtocolEvent(dose_rate, time, duration))

            regimens[label] = regimen

        return regimens

    def _get_number_and_parameter_names(
            self, exclude_pop_model=False, exclude_bottom_level=False):
        """
        Returns the number and names of the log-likelihood.

        The parameters of the HierarchicalLogLikelihood depend on the
        data, and the population model. So unless both are set, the
        parameters will reflect the parameters of the individual
        log-likelihoods.
        """
        # Get mechanistic model parameters
        parameter_names = self._mechanistic_model.parameters()

        # Get error model parameters
        for error_model in self._error_models:
            parameter_names += error_model.get_parameter_names()

        # Stop here if population model is excluded or isn't set
        if (self._population_models is None) or (
                exclude_pop_model):
            # Get number of parameters
            n_parameters = len(parameter_names)

            return (n_parameters, parameter_names)

        # Set default number of individuals
        n_ids = 0
        if self._data is not None:
            n_ids = len(self._ids)

        # Construct population parameter names
        pop_parameter_names = []
        for param_id, pop_model in enumerate(self._population_models):
            # Get mechanistic/error model parameter name
            name = parameter_names[param_id]

            # Add names for individual parameters
            n_indiv, _ = pop_model.n_hierarchical_parameters(n_ids)
            if (n_indiv > 0):
                # If individual parameters are relevant for the hierarchical
                # model, append them
                indiv_names = ['ID %s: %s' % (n, name) for n in self._ids]
                pop_parameter_names += indiv_names

            # Add population-level parameters
            if pop_model.n_parameters() > 0:
                # pop_names = ["%s %s" % (name, pnam) for pnam in pop_model.get_parameter_names()]
                pop_parameter_names += pop_model.get_parameter_names()

        # Return only top-level parameters, if bottom is excluded
        if exclude_bottom_level:
            # Filter bottom-level
            start = 0
            parameter_names = []
            for param_id, pop_model in enumerate(self._population_models):
                n_indiv, n_pop = pop_model.n_hierarchical_parameters(n_ids)

                # If heterogenous or uniform population model,
                # individuals count as top-level
                if chi.is_heterogeneous_or_uniform_model(pop_model):
                    end = start + n_indiv + n_pop
                else:
                    # Otherwise, we skip individuals
                    start += n_indiv
                    end = start + n_pop
                # Add population parameters
                parameter_names += pop_parameter_names[start:end]
                # Shift start index
                start = end

            # Get number of parameters
            n_parameters = len(parameter_names)

            return (n_parameters, parameter_names)


        # Get number of parameters
        n_parameters = len(pop_parameter_names)

        return (n_parameters, pop_parameter_names)

    def _set_error_model_parameter_names(self):
        """
        Resets the error model parameter names and prepends the output name
        if more than one output exists.
        """
        # Reset error model parameter names to defaults
        for error_model in self._error_models:
            error_model.set_parameter_names(None)

        # Rename error model parameters, if more than one output
        n_outputs = self._mechanistic_model.n_outputs()
        if n_outputs > 1:
            # Get output names
            outputs = self._mechanistic_model.outputs()

            for output_id, error_model in enumerate(self._error_models):
                # Get original parameter names
                names = error_model.get_parameter_names()

                # Prepend output name
                output = outputs[output_id]
                names = [output + ' ' + name for name in names]

                # Set new parameter names
                error_model.set_parameter_names(names)

    def _set_population_model_parameter_names(self):
        """
        Resets the population model parameter names and appends the individual
        parameter names.
        """
        # Get individual parameter names
        parameter_names = self.get_parameter_names(exclude_pop_model=True)

        # Construct population parameter names
        for param_id, pop_model in enumerate(self._population_models):
            # Get mechanistic/error model parameter name
            name = parameter_names[param_id]

            # Create names for population-level parameters
            if pop_model.n_parameters() > 0:
                # Get original parameter names
                pop_model.set_parameter_names()
                pop_names = pop_model.get_parameter_names()

                # Append individual names and rename population model
                # parameters
                names = [
                    '%s %s' % (pop_prefix, name) for pop_prefix in pop_names]
                pop_model.set_parameter_names(names)

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        .. note::
            1. Fixing model parameters resets the log-prior to ``None``.
            2. Once a population model is set, only population model
               parameters can be fixed.

        :param name_value_dict: A dictionary with model parameters as keys, and
            the value to be fixed at as values.
        :type name_value_dict: dict
        """
        # Check type of dictionanry
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        #Find parameters that are fixed for individuals, rather than fixed with one value for all
        valuesWithLen = {k: v for k, v in name_value_dict.items() if hasattr(v, "__len__")}
        if len(valuesWithLen)>0:
            assert all([len(v)==len(self._ids) for v in valuesWithLen.values()])
            if self._individual_fixed_param_dict is None:
                self._individual_fixed_param_dict = self._initialise_individual_fixed_params()
            for i, _id in enumerate(self._ids):
                self._individual_fixed_param_dict[_id] = {k: v[i] for k, v in valuesWithLen.items()}
                
        # If a population model is set, fix only population parameters
        if self._population_models is not None:
            pop_models = self._population_models

            # Convert models to reduced models
            for model_id, pop_model in enumerate(pop_models):
                if not isinstance(pop_model, chi.ReducedPopulationModel):
                    pop_models[model_id] = chi.ReducedPopulationModel(
                        pop_model)

            # Fix parameters
            for pop_model in pop_models:
                pop_model.fix_parameters(name_value_dict)

            # If no parameters are fixed, get original model back
            for model_id, pop_model in enumerate(pop_models):
                if pop_model.n_fixed_parameters() == 0:
                    pop_model = pop_model.get_population_model()
                    pop_models[model_id] = pop_model

            # Safe reduced models and reset priors
            self._population_models = pop_models
            self._log_prior = None

            # Update names and number of parameters
            self._n_parameters, self._parameter_names = \
                self._get_number_and_parameter_names()

            # Stop here
            # (individual parameters cannot be fixed when pop model is set)
            return None

        # Get submodels
        mechanistic_model = self._mechanistic_model
        error_models = self._error_models

        # Convert models to reduced models
        if not isinstance(mechanistic_model, chi.ReducedMechanisticModel):
            mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)
        for model_id, error_model in enumerate(error_models):
            if not isinstance(error_model, chi.ReducedErrorModel):
                if isinstance(error_model, chi.ErrorModelWithMeasuringErrors):
                    error_models[model_id] = chi.ReducedErrorModelWithMeasuringErrors(error_model)
                else:
                    error_models[model_id] = chi.ReducedErrorModel(error_model)

        # Fix model parameters
        mechanistic_model.fix_parameters(name_value_dict)
        for error_model in error_models:
            error_model.fix_parameters(name_value_dict)

        # If no parameters are fixed, get original model back
        if mechanistic_model.n_fixed_parameters() == 0:
            mechanistic_model = mechanistic_model.mechanistic_model()
            self._individual_fixed_param_dict = self._initialise_individual_fixed_params()

        for model_id, error_model in enumerate(error_models):
            if error_model.n_fixed_parameters() == 0:
                error_model = error_model.get_error_model()
                error_models[model_id] = error_model

        # Save reduced models and reset priors
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models
        self._log_prior = None

        # Update names and number of parameters
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

    def get_dosing_regimens(self):
        """
        Returns a dictionary of dosing regimens in form of
        :class:`myokit.Protocol` instances.

        The dosing regimens are extracted from the dataset if a dose key is
        provided. If no dose key is provided ``None`` is returned.
        """
        return self._dosing_regimens

    def get_log_prior(self):
        """
        Returns the :class:`LogPrior` for the model parameters. If no
        log-prior is set, ``None`` is returned.
        """
        return self._log_prior

    def get_log_posterior(self, individual=None, prior_is_id_specific=False):
        r"""
        Returns the :class:`LogPosterior` defined by the observed biomarkers,
        the administered dosing regimen, the mechanistic model, the error
        model, the log-prior, and optionally the population model and the
        fixed model parameters.

        If measurements of multiple individuals exist in the dataset, the
        indiviudals ID can be passed to return the log-posterior associated
        to that individual. If no ID is selected and no population model
        has been set, a list of log-posteriors is returned correspodning to
        each of the individuals.

        This method raises an error if the data or the log-prior has not been
        set. See :meth:`set_data` and :meth:`set_log_prior`.

        .. note::
            When a population model has been set, individual log-posteriors
            can no longer be selected and ``individual`` is ignored.

        :param individual: The ID of an individual. If ``None`` the
            log-posteriors for all individuals is returned.
        :type individual: str | None, optional
        :param prior_is_id_specific: If True and this is a population model,
            then the resulting log_prior will be a list of priors for each ID.
        :type prior_is_id_specific: bool, optional
        """
        # Check prerequesites
        if self._log_prior is None:
            raise ValueError(
                'The log-prior has not been set.')

        # Make sure individual is None, when population model is set
        _id = individual if self._population_models is None else None

        # Check that individual is in ids
        if (_id is not None) and (_id not in self._ids):
            raise ValueError(
                'The individual cannot be found in the ID column of the '
                'dataset.')

        #Ignore prior_is_id_specific if this is not a population model
        if self._population_models is None:
            prior_is_id_specific = False
            
        # Create log-likelihoods
        log_likelihoods = self._create_log_likelihoods(_id)
        log_priors      = self._log_prior
        if self._population_models is not None:
            # Compose HierarchicalLogLikelihoods
            log_likelihoods = [chi.HierarchicalLogLikelihood(
                log_likelihoods, self._population_models,
                id_key=self._id_key, time_key=self._time_key, biom_key=self._biom_key, meas_key=self._meas_key
            )]
            if prior_is_id_specific:
                log_priors = chi.IDSpecificLogPrior([
                    log_priors for i in self._ids], self._population_models, self._ids)

        # Compose the log-posteriors
        log_posteriors = []
        for log_likelihood in log_likelihoods:
            # Create individual posterior
            if isinstance(log_likelihood, chi.LogLikelihood):
                log_posterior = chi.LogPosterior(
                    log_likelihood, log_priors)

            # Create hierarchical posterior
            elif isinstance(log_likelihood, chi.HierarchicalLogLikelihood):
                log_posterior = chi.HierarchicalLogPosterior(
                    log_likelihood, log_priors)

            # Append to list
            log_posteriors.append(log_posterior)

        # If only one log-posterior in list, unwrap the list
        if len(log_posteriors) == 1:
            return log_posteriors.pop()

        return log_posteriors

    def get_n_parameters(
            self, exclude_pop_model=False, exclude_bottom_level=False):
        """
        Returns the number of model parameters, i.e. the combined number of
        parameters from the mechanistic model, the error model and, if set,
        the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the number of model parameters.

        :param exclude_pop_model: A boolean flag which can be used to obtain
            the number of parameters as if the population model wasn't set.
        :type exclude_pop_model: bool, optional
        :param exclude_bottom_level: A boolean flag which can be used to
            exclude the bottom-level parameters. This only has an effect when
            a population model is set.
        :type exclude_bottom_level: bool, optional
        """
        if exclude_pop_model:
            n_parameters, _ = self._get_number_and_parameter_names(
                exclude_pop_model=True)
            return n_parameters

        if exclude_bottom_level:
            n_parameters, _ = self._get_number_and_parameter_names(
                exclude_bottom_level=True)
            return n_parameters

        return self._n_parameters

    def get_parameter_names(
            self, exclude_pop_model=False, exclude_bottom_level=False):
        """
        Returns the names of the model parameters, i.e. the parameter names
        of the mechanistic model, the error model and, if set, the
        population model.

        Any parameters that have been fixed to a constant value will not be
        included in the list of model parameters.

        :param exclude_pop_model: A boolean flag which can be used to obtain
            the parameter names as if the population model wasn't set.
        :type exclude_pop_model: bool, optional
        :param exclude_bottom_level: A boolean flag which can be used to
            exclude the bottom-level parameters. This only has an effect when
            a population model is set.
        :type exclude_bottom_level: bool, optional
        """
        if exclude_pop_model:
            _, parameter_names = self._get_number_and_parameter_names(
                exclude_pop_model=True)
            return copy.copy(parameter_names)

        if exclude_bottom_level:
            _, parameter_names = self._get_number_and_parameter_names(
                exclude_bottom_level=True)
            return parameter_names

        return copy.copy(self._parameter_names)

    def get_predictive_model(self, exclude_pop_model=False, individual=None):
        """
        Returns the :class:`PredictiveModel` defined by the mechanistic model,
        the error model, and optionally the population model and the
        fixed model parameters.

        :param exclude_pop_model: A boolean flag which can be used to obtain
            the predictive model as if the population model wasn't set.
        :type exclude_pop_model: bool, optional
        """
        #Check if no population model has been set, or is excluded
        no_population_model = (self._population_models is None) or (exclude_pop_model)

        #Check if we have individual-specific fixed parameters
        sifpd = self._individual_fixed_param_dict
        if sifpd is not None and len(sifpd)>0 and no_population_model:
            if individual is None:
                warn(UserWarning(
                    "No individual given for predictive model, but individual-specific fixed parameters exist."))
                mechanistic_model = self._mechanistic_model
            else:
                mechanistic_model = self._mechanistic_model.copy()
                mechanistic_model.fix_parameters(sifpd[individual])
        else:
            mechanistic_model = self._mechanistic_model

        # Create predictive model
        predictive_model = chi.PredictiveModel(
            mechanistic_model, self._error_models)

        # Return if no population model has been set, or is excluded
        if no_population_model:
            return predictive_model

        # Create predictive population model
        #TODO: Check that all of the _population_models have the right individual fixed parameters
        predictive_model = chi.PredictivePopulationModel(
            predictive_model, self._population_models, IDs=self._ids)

        return predictive_model

    def set_data(
            self, data, dataErr=None, output_biomarker_dict=None, id_key='ID',
            time_key='Time', biom_key='Biomarker', meas_key='Measurement',
            dose_key='Dose', dose_duration_key='Duration'):
        """
        Sets the data of the modelling problem.

        The data contains information about the measurement time points, the
        observed biomarker values, the type of biomarkers, IDs to
        identify the corresponding individuals, and optionally information
        on the administered dose amount and duration.

        The data is expected to be in form of a :class:`pandas.DataFrame`
        with the columns ID | Time | Biomarker | Measurement | Dose |
        Duration.

        If no dose or duration information exists, the corresponding column
        keys can be set to ``None``.

        :param data: A dataframe with an ID, time, biomarker,
            measurement and optionally a dose and duration column.
        :type data: pandas.DataFrame
        :param dataErr: A dataframe with entries labelled as in data,
            but whose measurement column gives the measuring error 
            of entries in data.
        :type dataErr: pandas.DataFrame
        :param output_biomarker_dict: A dictionary with mechanistic model
            output names as keys and dataframe biomarker names as values. If
            ``None`` the model outputs and biomarkers are assumed to have the
            same names.
        :type output_biomarker_dict: dict, optional
        :param id_key: The key of the ID column in the
            :class:`pandas.DataFrame`. Default is `'ID'`.
        :type id_key: str, optional
        :param time_key: The key of the time column in the
            :class:`pandas.DataFrame`. Default is `'ID'`.
        :type time_key: str, optional
        :param biom_key: The key of the biomarker column in the
            :class:`pandas.DataFrame`. Default is `'Biomarker'`.
        :type biom_key: str, optional
        :param meas_key: The key of the measurement column in the
            :class:`pandas.DataFrame`. Default is `'Measurement'`.
        :type meas_key: str, optional
        :param dose_key: The key of the dose column in the
            :class:`pandas.DataFrame`. Default is `'Dose'`.
        :type dose_key: str, optional
        :param dose_duration_key: The key of the duration column in the
            :class:`pandas.DataFrame`. Default is `'Duration'`.
        :type dose_duration_key: str, optional
        """
        # Check if we need to store data errors, too
        haveDataErrors = dataErr is not None

        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be a pandas.DataFrame.')
        if haveDataErrors and not isinstance(dataErr, pd.DataFrame):
            raise TypeError(
                'Data errors have to be a pandas.DataFrame.')

        # If model does not support dose administration, set dose keys to None
        mechanistic_model = self._mechanistic_model
        if isinstance(self._mechanistic_model, chi.ReducedMechanisticModel):
            mechanistic_model = self._mechanistic_model.mechanistic_model()
        if isinstance(mechanistic_model, chi.PharmacodynamicModel):
            dose_key = None
            dose_duration_key = None

        keys = [id_key, time_key, biom_key, meas_key]
        if dose_key is not None:
            keys += [dose_key]
        if dose_duration_key is not None:
            keys += [dose_duration_key]

        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')
            if haveDataErrors and key not in dataErr.keys():
                raise ValueError(
                    'DataErr does not have the key <' + str(key) + '>.')

        # Get default output-biomarker map
        outputs = self._mechanistic_model.outputs()
        biomarkers = data[biom_key].dropna().unique()
        if haveDataErrors:
            biomarkersE = dataErr[biom_key].dropna().unique()
            assert (biomarkers==biomarkersE).all()

        if output_biomarker_dict is None:
            if (len(outputs) == 1) and (len(biomarkers) == 1):
                # Create map of single output to single biomarker
                output_biomarker_dict = {outputs[0]: biomarkers[0]}
            else:
                # Assume trivial map
                output_biomarker_dict = {output: output for output in outputs}

        # Check that output-biomarker map is valid
        for output in outputs:
            if output not in list(output_biomarker_dict.keys()):
                raise ValueError(
                    'The output <' + str(output) + '> could not be identified '
                    'in the output-biomarker map.')

            biomarker = output_biomarker_dict[output]
            if biomarker not in biomarkers:
                raise ValueError(
                    'The biomarker <' + str(biomarker) + '> could not be '
                    'identified in the dataframe.')

        self._id_key, self._time_key, self._biom_key, self._meas_key = [
            id_key, time_key, biom_key, meas_key]
        self._data = data[keys]
        self._output_biomarker_dict = output_biomarker_dict

        # Make sure data is formatted correctly
        self._data = self._clean_data(self._data, dose_key, dose_duration_key)
        self._ids = self._data[self._id_key].unique()

        #Do the same thing to data errors
        if haveDataErrors:
            self._dataErr = dataErr[keys]
            self._dataErr = self._clean_data(self._dataErr, dose_key, dose_duration_key)
            err_ids = self._data[self._id_key].unique()
            assert (self._ids == err_ids).all()

        # Extract dosing regimens
        self._dosing_regimens = None
        if dose_key is not None:
            self._dosing_regimens = self._extract_dosing_regimens(
                dose_key, dose_duration_key)

        # Update number and names of parameters
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

    def set_log_prior(self, log_priors, parameter_names=None, prior_is_id_specific=False):
        """
        Sets the log-prior probability distribution of the model parameters.

        By default the log-priors are assumed to be ordered according to
        :meth:`get_parameter_names`. Alternatively, the mapping of the
        log-priors can be specified explicitly with the input argument
        ``param_names``.

        If a population model has not been set, the provided log-prior is used
        for all individuals.

        .. note::
            This method requires that the data has been set, since the number
            of parameters of an hierarchical log-posterior vary with the number
            of individuals in the dataset.

        :param log_priors: A list of :class:`pints.LogPrior` of the length
            :meth:`get_n_parameters`.
        :type log_priors: list[pints.LogPrior]
        :param parameter_names: A list of model parameter names, which is used
            to map the log-priors to the model parameters. If ``None`` the
            log-priors are assumed to be ordered according to
            :meth:`get_parameter_names`.
        :type parameter_names: list[str], optional
        :param prior_is_id_specific: If True and this is a population model,
            then the resulting log_prior will be a list of priors for each ID.
        :type prior_is_id_specific: bool, optional
        """
        # Check prerequesites
        if self._data is None:
            raise ValueError('The data has not been set.')

        # Check inputs
        for log_prior in log_priors:
            if not isinstance(log_prior, pints.LogPrior):
                raise ValueError(
                    'All marginal log-priors have to be instances of a '
                    'pints.LogPrior.')

        expected_n_parameters = self.get_n_parameters(
            exclude_pop_model=prior_is_id_specific, exclude_bottom_level=not prior_is_id_specific)
        if len(log_priors) != expected_n_parameters:
            raise ValueError(
                'One marginal log-prior has to be provided for each '
                'parameter.There are <' + str(expected_n_parameters) + '> model '
                'parameters.')

        n_parameters = 0
        for log_prior in log_priors:
            n_parameters += log_prior.n_parameters()

        if n_parameters != expected_n_parameters:
            raise ValueError(
                'The joint log-prior does not match the dimensionality of the '
                'problem. At least one of the marginal log-priors appears to '
                'be multivariate.')

        if parameter_names is not None:
            model_names = self.get_parameter_names(
                exclude_pop_model=prior_is_id_specific, exclude_bottom_level=not prior_is_id_specific)
            if sorted(list(parameter_names)) != sorted(model_names):
                raise ValueError(
                    'The specified parameter names do not match the model '
                    'parameter names.')

            # Sort log-priors according to parameter names
            ordered = []
            for name in model_names:
                index = parameter_names.index(name)
                ordered.append(log_priors[index])

            log_priors = ordered

        self._log_prior = pints.ComposedLogPrior(*log_priors)

    def set_normalised_error_models(self, value):
        """
        Makes all error functions divide likelihoods by the mean log observation before returning.

        :param value: A boolean.
        """
        self._normalised_error_models = value
        for em in self._error_models:
            em.set_normalised_log_likelihood(value)

    def set_population_model(self, pop_models, parameter_names=None):
        """
        Sets the population model of the modelling problem.

        A population model specifies how model parameters vary across
        individuals. The population model is defined by a list of
        :class:`PopulationModel` instances, one for each individual model
        parameter.

        .. note::
            Setting a population model resets the log-prior to ``None``.

        :param pop_models: A list of :class:`PopulationModel` instances of
            the same length as the number of individual model parameters, see
            :meth:`get_n_parameters` with ``exclude_pop_model=True``.
        :type pop_models: list[PopulationModel]
        :param parameter_names: A list of model parameter names, which can be
            used to map the population models to model parameters. If ``None``,
            the population models are assumed to be ordered in the same way as
            the model parameters, see
            :meth:`get_parameter_names` with ``exclude_pop_model=True``.
        :type parameter_names: list[str], optional
        """
        # Check inputs
        for pop_model in pop_models:
            if not isinstance(pop_model, chi.PopulationModel):
                raise TypeError(
                    'The population models have to be an instance of a '
                    'chi.PopulationModel.')

        # Get individual parameter names
        n_parameters, param_names = self._get_number_and_parameter_names(
            exclude_pop_model=True)

        # Make sure that each parameter is assigned to a population model
        if len(pop_models) != n_parameters:
            raise ValueError(
                'The number of population models does not match the number of '
                'model parameters. Exactly one population model has to be '
                'provided for each parameter. There are '
                '<' + str(n_parameters) + '> model parameters.')

        if (parameter_names is not None) and (
                sorted(parameter_names) != sorted(param_names)):
            raise ValueError(
                'The parameter names do not coincide with the model parameter '
                'names.')

        # Sort inputs according to `params`
        if parameter_names is not None:
            # Create default population model container
            ordered_pop_models = []

            # Map population models according to parameter names
            for name in param_names:
                index = parameter_names.index(name)
                ordered_pop_models.append(pop_models[index])

            pop_models = ordered_pop_models

        # Set data within each pop_model that needs it
        for pop_model in pop_models:
            if isinstance(pop_model, chi.KolmogorovSmirnovPopulationModel):
                pop_model.create_observation_CDF(
                    self._data, time_key=self._time_key, biom_key=self._biom_key, meas_key=self._meas_key)

        # Save individual parameter names and population models
        self._population_models = copy.copy(pop_models)

        # Update parameter names and number of parameters
        self._set_population_model_parameter_names()
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

        # Set prior to default
        self._log_prior = None
