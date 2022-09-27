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
        if not isinstance(mechanistic_model, chi.MechanisticModel):
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
        mechanistic_model = mechanistic_model.copy()

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
        self._population_model = None
        self._log_prior = None
        self._data = None
        self._dataErr = None
        self._covariates = None
        self._dosing_regimens = None
        self._individual_fixed_param_dict = None

        # Set error models to un-normalised by default
        self.set_normalised_error_models(False)

        # Set parameter names and number of parameters
        self._set_error_model_parameter_names()

    def _check_covariate_dict(
            self, covariate_dict, covariate_names, observables):
        """
        Makes sure that for each covariate name (from model) an observable
        (from dataframe) exists.
        """
        # Check that model needs covariates
        if len(covariate_names) == 0:
            return None

        # If no mapping is provided, construct default mapping
        if covariate_dict is None:
            # Assume trivial map
            covariate_dict = {cov: cov for cov in covariate_names}

        # Check that covariate name map is valid
        for cov in covariate_names:
            if cov not in list(covariate_dict.keys()):
                raise ValueError(
                    'The covariate <' + str(cov) + '> could not be identified '
                    'in the covariate name map.')

            mapped_name = covariate_dict[cov]
            if mapped_name not in observables:
                raise ValueError(
                    'The covariate <' + str(mapped_name) + '> could not be '
                    'identified in the dataframe.')

        return covariate_dict

    def _check_covariate_values(self, covariate_names):
        """
        Makes sure that covariates can be reshaped in to an array of shape
        (n, c).

        In other words, checks whether for each covariate_name there exists
        exactly one non-NaN value for each ID.
        """
        # Check that model needs covariates
        if len(covariate_names) == 0:
            return None

        for name in covariate_names:
            # Mask covariate values
            mask = self._data[self._obs_key] == self._covariate_dict[name]
            temp = self._data[mask]

            for _id in self._ids:
                # Mask values for individual
                mask = temp[self._id_key] == _id
                temp2 = temp.loc[mask, self._value_key].dropna()

                if len(temp2) != 1:
                    covariate = self._covariate_dict[name]
                    raise ValueError(
                        'There are either 0 or more than 1 value of the '
                        'covariate %s for ID %s. '
                        'Exactly one covariate value '
                        'has to be provided for each ID.' % (covariate, _id)
                    )

    def _check_output_observable_dict(
            self, output_observable_dict, outputs, observables):
        """
        Makes sure that the mechanistic model outputs are correctly mapped to
        the observables in the dataframe.
        """
        if output_observable_dict is None:
            if (len(outputs) == 1) and (len(observables) == 1):
                # Create map of single output to single observable
                output_observable_dict = {outputs[0]: observables[0]}
            else:
                # Assume trivial map
                output_observable_dict = {output: output for output in outputs}

        # Check that output-observable map is valid
        for output in outputs:
            if output not in list(output_observable_dict.keys()):
                raise ValueError(
                    'The output <' + str(output) + '> could not be identified '
                    'in the output-observable map.')

            observable = output_observable_dict[output]
            if observable not in observables:
                raise ValueError(
                    'The observable <' + str(observable) + '> could not be '
                    'identified in the dataframe.')

        return output_observable_dict

    def _clean_data(self, data, dose_key, dose_duration_key):
        """
        Makes sure that the data is formated properly.

        1. ids are strings
        2. time are numerics or NaN
        3. observables are strings
        4. values are numerics or NaN
        5. observable types are 'Modelled' or 'Covariate'
        6. dose are numerics or NaN
        7. duration are numerics or NaN
        """
        # Create container for data
        columns = [
            self._id_key, self._time_key, self._obs_key, self._value_key]
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

        # Convert observables to strings
        cleanData[self._obs_key] = data[self._obs_key].astype(
            "string")

        # Convert values to numerics
        cleanData[self._value_key] = pd.to_numeric(data[self._value_key])

        # Convert dose to numerics
        if dose_key is not None:
            cleanData[dose_key] = pd.to_numeric(
                data[dose_key])

        # Convert duration to numerics
        if dose_duration_key is not None:
            cleanData[dose_duration_key] = pd.to_numeric(
                data[dose_duration_key])

        return cleanData

    def _create_hierarchical_log_likelihood(self, log_likelihoods):
        """
        Returns an instance of a chi.HierarchicalLoglikelihood based on
        the provided list of log-likelihoods and the population models.
        """
        # Get covariates from the dataset if any are needed
        covariate_names = self.get_covariate_names()
        covariates = None
        if len(covariate_names) > 0:
            covariates = self._extract_covariates(covariate_names)

        log_likelihood = chi.HierarchicalLogLikelihood(
                log_likelihoods, self._population_model, covariates)

        return log_likelihood

    def _create_log_likelihoods(self, ids):
        """
        Returns a list of log-likelihoods, one for each individual in the
        dataset. If individual is not None, only the likelihood of this
        individual is returned.
        """
        # Create a likelihood for each individual
        log_likelihoods = []
        for individual in ids:
            # Set dosing regimen
            if self._dosing_regimens:
                self._mechanistic_model.set_dosing_regimen(
                    self._dosing_regimens[individual])

            # Get submodels
            mechanistic_model = self._mechanistic_model
            error_models = self._error_models

            #Set individually fixed parameters
            if self._individual_fixed_param_dict is not None and len(self._individual_fixed_param_dict)>0:
                try:
                    # Convert models to reduced models
                    if not isinstance(mechanistic_model, chi.ReducedMechanisticModel):
                        mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)
                    for model_id, error_model in enumerate(error_models):
                        if not isinstance(error_model, chi.ReducedErrorModel):
                            if isinstance(error_model, chi.ErrorModelWithMeasuringErrors):
                                error_models[model_id] = chi.ReducedErrorModelWithMeasuringErrors(error_model)
                            else:
                                error_models[model_id] = chi.ReducedErrorModel(error_model)

                    #Fix parameters
                    mechanistic_model.fix_parameters(self._individual_fixed_param_dict[individual])
                    for error_model in error_models:
                        error_model.fix_parameters(self._individual_fixed_param_dict[individual])
                except TypeError:
                    pass

            log_likelihood = self._create_log_likelihood(individual)
            if log_likelihood is not None:
                # If data exists for this individual, append to log-likelihoods
                log_likelihoods.append(log_likelihood)

        if len(ids) == 1:
            return log_likelihoods[0]

        return log_likelihoods

    def _create_log_likelihood(self, individual):
        """
        Gets the relevant data for the individual and returns the resulting
        chi.LogLikelihood.
        """
        # Flag for considering measuremnet errors
        haveErrors = self._dataErr is not None

        # Get individuals data
        times = []
        observations = []
        observationErrors = []
        mask = self._data[self._id_key] == individual
        data = self._data[mask][
            [self._time_key, self._obs_key, self._value_key]]
        # Get individual measurement errors on the data
        if haveErrors:
            maskE = self._dataErr[self._id_key] == individual
            assert (np.asarray(mask)==np.asarray(maskE)).all()
            dataErr = self._dataErr[mask][
                [self._time_key, self._obs_key, self._value_key]]

        for output in self._mechanistic_model.outputs():
            # Mask data for observable
            observable = self._output_observable_dict[output]
            mask = data[self._obs_key] == observable
            temp_df = data[mask]
            if haveErrors:
                maskE = dataErr[self._biom_key] == observable
                assert (np.asarray(mask)==np.asarray(maskE)).all()
                temp_ef = dataErr[mask]

            # Filter times and observations for non-NaN entries
            mask = temp_df[self._value_key].notnull()
            temp_df = temp_df[[self._time_key, self._value_key]][mask]
            mask = temp_df[self._time_key].notnull()
            temp_df = temp_df[mask]
            if haveErrors:
                maskE = temp_ef[self._value_key].notnull()
                temp_ef = temp_ef[[self._time_key, self._value_key]][mask]
                maskE = temp_ef[self._time_key].notnull()
                temp_ef = temp_ef[mask]
                assert (np.asarray(mask)==np.asarray(maskE)).all()

            # Collect data for output
            times.append(temp_df[self._time_key].to_numpy())
            observations.append(temp_df[self._value_key].to_numpy())
            if haveErrors:
                observationErrors.append(temp_ef[self._value_key].to_numpy())

        # Create log-likelihood and set ID to individual
        if haveErrors:
            error_models = self._error_models
            for model_id, error_model in enumerate(error_models):
                isMeas = isinstance(error_model, (
                    chi.ErrorModelWithMeasuringErrors, chi.ReducedErrorModelWithMeasuringErrors))
                if not isMeas:
                    if isinstance(error_model, chi.ReducedErrorModel):
                        error_models[model_id] = \
                            chi.return_reduced_measuring_error_model_from_reduced_model(error_model)
                    else:
                        error_models[model_id] = chi.return_measuring_error_model_from_error_model(error_model)
                
            log_likelihood = chi.LogLikelihoodWithMeasuringErrors(
                self._mechanistic_model, error_models, observations, observationErrors, times)
        else:
            log_likelihood = chi.LogLikelihood(
                self._mechanistic_model, self._error_models, observations, times)
        log_likelihood.set_id(individual)

        return log_likelihood

    def _initialise_individual_fixed_params(self):
        """
        Initialises a dictionary to contain parameters that are fixed for each patient parameter.
        """
        fixedParams = dict()
        # for label in self._ids:
        #     fixedParams = dict()

        return fixedParams

    def _extract_covariates(self, covariate_names):
        """
        Extracts covariates from the pandas.DataFrame and formats them
        as a np.ndarray of shape (n, c).
        """
        # Format covariates to array of shape (n, c)
        c = len(covariate_names)
        n = len(self._ids)
        covariates = np.empty(shape=(n, c))
        for idc, name in enumerate(covariate_names):
            mask = self._data[self._obs_key] == self._covariate_dict[name]
            temp = self._data[mask]
            for idn, _id in enumerate(self._ids):
                mask = temp[self._id_key] == _id
                covariates[idn, idc] = \
                    temp.loc[mask, self._value_key].dropna().values

        return covariates

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
        parameter from the model. Fixing the value of a parameter to ``None``,
        sets the parameter free again.

        .. note::
            1. Fixing model parameters resets the log-prior to ``None``.
            2. Once a population model is set, only population model
               parameters can be fixed.
            3. Setting data resets all population parameters as the number of
               parameters may change with the number of modelled individuals.

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
                if _id not in self._individual_fixed_param_dict:
                    self._individual_fixed_param_dict[_id] = {}
                for k, v in valuesWithLen.items():
                    self._individual_fixed_param_dict[_id][k] = v[i]
                
        # Convert models to reduced models
        models = []
        # If a population model is set, fix only population parameters
        if self._population_model is not None:
            population_model = self._population_model
            if not isinstance(population_model, chi.ReducedPopulationModel):
                population_model = chi.ReducedPopulationModel(
                    population_model)
            models.append(population_model)
        else:
            mechanistic_model = self._mechanistic_model
            error_models = self._error_models
            if not isinstance(mechanistic_model, chi.ReducedMechanisticModel):
                mechanistic_model = chi.ReducedMechanisticModel(
                    mechanistic_model)
            for idm, error_model in enumerate(error_models):
                if not isinstance(error_model, chi.ReducedErrorModel):
                    error_model = chi.ReducedErrorModel(error_model)
                error_models[idm] = error_model
            models += [mechanistic_model] + error_models

        # Fix parameters
        for model in models:
            model.fix_parameters(name_value_dict)

        #If a parameter has been globally unfixed, remove it from the individually fixed dictionary
        if self._individual_fixed_param_dict is not None and len(self._individual_fixed_param_dict)>0:
            for key, value in name_value_dict.items():
                if value is None:
                    for i, _id in enumerate(self._ids):
                        if key in self._individual_fixed_param_dict[_id]:
                            del self._individual_fixed_param_dict[_id][key]

        # Safe models
        # (if no parameters are fixed, convert back to non-reduced models)
        if self._population_model is not None:
            population_model = models[0]
            if population_model.n_fixed_parameters() == 0:
                population_model = population_model.get_population_model()
            self._population_model = population_model
        else:
            mechanistic_model = models[0]
            error_models = models[1:]
            if mechanistic_model.n_fixed_parameters() == 0:
                mechanistic_model = mechanistic_model.mechanistic_model()
            for idm, error_model in enumerate(error_models):
                if error_model.n_fixed_parameters() == 0:
                    error_models[idm] = error_model.get_error_model()
            self._mechanistic_model = mechanistic_model
            self._error_models = error_models

        # Reset priors
        self._log_prior = None

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

    def get_log_posterior(self, individual=None):
        r"""
        Returns the :class:`LogPosterior` defined by the measurements of the
        modelled observables,
        the administered dosing regimen, the mechanistic model, the error
        model, the log-prior, and optionally the population model, covariates
        and the fixed model parameters.

        If measurements of multiple individuals exist in the dataset, the
        indiviudals ID can be passed to return the log-posterior associated
        to that individual. If no ID is selected and no population model
        has been set, a list of log-posteriors is returned corresponding to
        each of the individuals.

        This method raises an error if the data or the log-prior have not been
        set. See :meth:`set_data` and :meth:`set_log_prior`.

        :param individual: The ID of an individual. If ``None`` the
            log-posteriors for all individuals is returned. Input is ignored if
            a population model is set.
        :type individual: str, optional
        """
        # Check prerequesites
        if self._data is None:
            raise ValueError(
                'The data has not been set.')
        if self._log_prior is None:
            raise ValueError(
                'The log-prior has not been set.')

        # Check that individual is in ids
        _id = individual if self._population_model is None else None
        if (_id is not None) and (_id not in self._ids):
            raise ValueError(
                'The individual cannot be found in the ID column of the '
                'dataset.')

        # Set ID to all IDs if population model is set, and to first ID
        # if not set but None
        if self._population_model is not None:
            ids = self._ids
        elif _id is None:
            ids = [self._ids[0]]
        else:
            ids = [_id]

        #Ignore prior_is_id_specific if this is not a population model
        if self._population_models is None:
            prior_is_id_specific = False
        else:
            prior_is_id_specific = self._prior_is_id_specific
            
        # Create log-likelihoods
        log_likelihood = self._create_log_likelihoods(ids)
        log_prior      = self._log_prior
        if self._population_model is not None:
            # Compose HierarchicalLogLikelihoods
            log_likelihood = self._create_hierarchical_log_likelihood(
                log_likelihood)
            if prior_is_id_specific:
                log_prior = chi.IDSpecificLogPrior([
                    log_prior for i in ids], self._population_models, ids)

        # Compose the log-posterior
        if isinstance(log_likelihood, chi.LogLikelihood):
            log_posterior = chi.LogPosterior(
                log_likelihood, log_prior)
        else:
            log_posterior = chi.HierarchicalLogPosterior(
                log_likelihood, log_prior)

        return log_posterior

    def get_n_parameters(self, exclude_pop_model=False, exclude_bottom_level=False):
        """
        Returns the number of the model parameters, i.e. the combined number of
        parameters from the mechanistic model and the error model when no
        population model has been set, or the number of population model
        parameters when a population model has been set.

        Any parameters that have been fixed to a constant value will not be
        included in the number of model parameters.

        :param exclude_pop_model: A boolean flag to indicate whether the
            population model should be ignored (if set).
        :type exclude_pop_model: bool, optional
        :param exclude_bottom_level: A boolean flag which can be used to
            exclude the bottom-level parameters. This only has an effect when
            a population model is set.
        :type exclude_bottom_level: bool, optional
        """
        if (self._population_model is None) or exclude_pop_model:
            n_parameters = self._mechanistic_model.n_parameters()
            for error_model in self._error_models:
                n_parameters += error_model.n_parameters()
            return n_parameters

        return self._population_model.n_parameters()
        # n_total, n_top = self._population_model.n_hierarchical_parameters(self._n_ids)
        # if exclude_bottom_level:
        #     # n_bottom = n_total - n_top
        #     return n_top
        # else:
        #     return n_total #also known as self._population_model.n_parameters()

    def get_covariate_names(self):
        """
        Returns the names of the covariates.
        """
        if self._population_model is None:
            return []

        return self._population_model.get_covariate_names()

    def get_parameter_names(self, exclude_pop_model=False, exclude_bottom_level=False):
        """
        Returns the names of the model parameters, i.e. the combined names of
        the mechanistic model parameters and the error model parameters when no
        population model has been set, or the names of population model
        parameters when a population model has been set.

        Any parameters that have been fixed to a constant value will not be
        included.

        :param exclude_pop_model: A boolean flag to indicate whether the
            population model should be ignored (if set).
        :type exclude_pop_model: bool, optional
        :param exclude_bottom_level: A boolean flag which can be used to
            exclude the bottom-level parameters. This only has an effect when
            a population model is set.
        :type exclude_bottom_level: bool, optional
        """
        if (self._population_model is None) or exclude_pop_model:
            names = self._mechanistic_model.parameters()
            for error_model in self._error_models:
                names += error_model.get_parameter_names()
            return names

        return self._population_model.get_parameter_names()

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

        # Get submodels
        mechanistic_model = self._mechanistic_model
        error_models = self._error_models

        #Check if we have individual-specific fixed parameters
        sifpd = self._individual_fixed_param_dict
        if sifpd is not None and len(sifpd)>0 and no_population_model:
            if individual is None:
                warn(UserWarning(
                    "No individual given for predictive model, but individual-specific fixed parameters exist."))
            else:
                # Convert models to reduced models
                if not isinstance(mechanistic_model, chi.ReducedMechanisticModel):
                    mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)

                # -- error models
                error_models = [None for e in self._error_models]
                for model_id, error_model in enumerate(self._error_models):
                    if not isinstance(error_model, chi.ReducedErrorModel):
                        if isinstance(error_model, chi.ErrorModelWithMeasuringErrors):
                            error_models[model_id] = chi.ReducedErrorModelWithMeasuringErrors(error_model)
                        else:
                            error_models[model_id] = chi.ReducedErrorModel(error_model)
                    else:
                        error_models[model_id] = copy.deepcopy(error_model)

                #Fix parameters
                mechanistic_model.fix_parameters(sifpd[individual])
                for error_model in error_models:
                    error_model.fix_parameters(sifpd[individual])

        # Create predictive model
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Return if no population model has been set, or is excluded
        if no_population_model:
            return predictive_model
        else:
            predictive_model = chi.PopulationPredictiveModel(
                predictive_model, self._population_model)
            return predictive_model

    def set_data(
            self, data, dataErr=None, output_observable_dict=None, covariate_dict=None,
            id_key='ID', time_key='Time', obs_key='Observable',
            value_key='Value', dose_key='Dose', dose_duration_key='Duration'):
        """
        Sets the data of the modelling problem.

        The data contains information about the measurement time points, the
        measured values of the observables, the observable name, IDs to
        identify the corresponding individuals, and optionally information
        on the administered dose amount and duration.

        The data is expected to be in form of a :class:`pandas.DataFrame`
        with the columns ID | Time | Observable | Value | Dose | Duration.

        If no information exists, the corresponding column
        keys can be set to ``None``.

        .. note::
            Time-dependent covariates are currently not supported. Thus, the
            Time column of observables that are used as covariates is ignored.

        :param data: A dataframe with an ID, time, observable,
            value and optionally a dose and duration column.
        :type data: pandas.DataFrame
        :param output_observable_dict: A dictionary with mechanistic model
            output names as keys and dataframe observable names as values. If
            ``None`` the model outputs and observables are assumed to have the
            same names.
        :type output_observable_dict: dict, optional
        :param covariate_dict: A dictionary with population model covariate
            names as keys and dataframe observables as values. If
            ``None`` the model covariates and observables are assumed to have
            the same names.
        :type covariate_dict: dict, optional
        :param id_key: The key of the ID column in the
            :class:`pandas.DataFrame`. Default is ``'ID'``.
        :type id_key: str, optional
        :param time_key: The key of the time column in the
            :class:`pandas.DataFrame`. Default is ``'Time'``.
        :type time_key: str, optional
        :param obs_key: The key of the observable column in the
            :class:`pandas.DataFrame`. Default is ``'Observable'``.
        :type obs_key: str, optional
        :param value_key: The key of the value column in the
            :class:`pandas.DataFrame`. Default is ``'Value'``.
        :type value_key: str, optional
        :param dose_key: The key of the dose column in the
            :class:`pandas.DataFrame`. Default is ``'Dose'``.
        :type dose_key: str, optional
        :param dose_duration_key: The key of the duration column in the
            :class:`pandas.DataFrame`. Default is ``'Duration'``.
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
        if not self._mechanistic_model.supports_dosing():
            dose_key = None
            dose_duration_key = None

        keys = [id_key, time_key, obs_key, value_key]
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

        # Check output observable map
        outputs = self._mechanistic_model.outputs()
        observables = data[obs_key].dropna().unique()
        output_observable_dict = self._check_output_observable_dict(
            output_observable_dict, outputs, observables)
        if haveDataErrors:
            observablesE = dataErr[obs_key].dropna().unique()
            assert (observables==observablesE).all()


        # Check covariate name map
        covariate_names = self.get_covariate_names()
        covariate_dict = self._check_covariate_dict(
            covariate_dict, covariate_names, observables)

        self._id_key, self._time_key, self._obs_key, self._value_key = [
            id_key, time_key, obs_key, value_key]
        self._data = data[keys]
        self._output_observable_dict = output_observable_dict
        self._covariate_dict = covariate_dict

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

        # Set number of modelled individuals of population model
        if isinstance(self._population_model, chi.ReducedPopulationModel):
            # Unfix model parameters
            self._population_model = \
                self._population_model.get_population_model()
        if self._population_model is not None:
            self._population_model.set_n_ids(len(self._ids))

        # Check that covariates can be reshaped into (n, c)
        self._check_covariate_values(covariate_names)

    def set_log_prior(self, log_prior, prior_is_id_specific=False):
        """
        Sets the prior distribution of the model parameters.

        The log-prior dimensions are assumed to be ordered according to
        :meth:`get_parameter_names`.

        .. note::
            This method requires that the data has been set, since the number
            of parameters of an hierarchical model may vary with the number
            of individuals in the dataset
            (see e.g. :class:`HeterogeneousModel`).

        :param log_prior: A :class:`pints.LogPrior` of the length
            :meth:`get_n_parameters`.
        :type log_priors: pints.LogPrior
        :param prior_is_id_specific: If True and this is a population model,
            then the resulting log_prior will be a list of priors for each ID.
        :type prior_is_id_specific: bool, optional
        """
        # Check prerequesites
        if self._data is None:
            raise ValueError('The data has not been set.')

        # Check inputs
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'The log-prior has to be an instance of pints.LogPrior.')
        if log_prior.n_parameters() != self.get_n_parameters():
            raise ValueError(
                'The dimension of the log-prior has to be the same as the '
                'number of parameters in the model. There are <'
                + str(self.get_n_parameters()) + '> model parameters.')

        self._log_prior = log_prior
        self._prior_is_id_specific = prior_is_id_specific

    def set_normalised_error_models(self, value=True):
        """
        Makes all error functions divide likelihoods by the mean log observation before returning.

        :param value: A boolean.
        """
        self._normalised_error_models = value
        for em in self._error_models:
            em.set_normalised_log_likelihood(value)

    def set_population_model(self, population_model):
        """
        Sets the population model.

        A population model specifies how model parameters vary across
        individuals. The dimension of the population model has to match the
        number of model parameters.

        .. note::
            Setting a population model resets the log-prior to ``None``,
            because it changes the top-level parameters of the model.

        :param population_model: A :class:`PopulationModel` whose dimension is
            the same as the number of bottom-level parameters.
        :type population_model: PopulationModel
        """
        # Check inputs
        if not isinstance(population_model, chi.PopulationModel):
            raise TypeError(
                'The population model has to be an instance of '
                'chi.PopulationModel.')

        # Make sure that dimension of population model is correct
        n_parameters = self.get_n_parameters(exclude_pop_model=True)
        if population_model.n_dim() != n_parameters:
            raise ValueError(
                'The dimension of the population model does not match the '
                'number of bottom-level parameters. There are '
                '<' + str(n_parameters) + '> bottom-level parameters.')

        # Remember population model
        self._population_model = population_model

        # Set number of modelled individuals, if data has been set already
        if self._data is not None:
            self._population_model.set_n_ids(len(self._ids))

        # Set dimension names to bottom-level parameters
        names = self.get_parameter_names(exclude_pop_model=True)
        self._population_model.set_dim_names(names)

        # Check that covariates can be found, if data has already been set
        if self._data is not None:
            try:
                # Get covariate names
                covariate_names = self.get_covariate_names()
                observables = self._data[self._obs_key].dropna().unique()
                self._covariate_dict = self._check_covariate_dict(
                    self._covariate_dict, covariate_names, observables)
            except ValueError:
                # New population model and data are not compatible, so reset
                # data
                self._data = None
                warn(
                    'The covariates of the new population model could not '
                    'automatically be matched to the observables in the '
                    'dataset. The data was therefore reset. Please set the '
                    'data again with the `set_data` method and specify the '
                    'covariate mapping.', UserWarning)

        # Set prior to default
        self._log_prior = None
