from copy import copy
import pandas as pd
import numpy as np
from chi import MechanisticModel

class MechanisticModelNoSBML(MechanisticModel):
    '''A function to define mechanistic models for use in Chi's hierarchical pints MCMC, but without requiring SBML files.
    
    :param func: a function of the form f = lambda times, ICs, params: model_output(*args).
    :param dataFrames: a list of pandas dataframes for each patient, containing the data. This is for error checking only and is not stored.
    :param paramNames: a list of the names of each parameter.
    :param ynorm: a scalar or numpy array that is used to divide model outputs, to ensure fair comparison across
        observables.
    :param replaceZero: a scalar, which is used to replace values less than or equal to 0, for use when returning
        estimates to log likelihood functions.
    '''
    def __init__(self, func, dataFrames, paramNames, ynorm=1, replaceZero=None):
        self.func = func

        #Parse data
        # self.dataFrame_list = dataFrames
        assert all([type(data) in [pd.DataFrame, pd.Series] for data in dataFrames])
        # self.dataValues_list = [data.values for data in dataFrames]
        # self.dataTimes_list = [data.index.values for data in dataFrames]

        #Sizes of data
        self.numTimes, self.numOutputs = dataFrames[0].shape
        # assert all([(data.shape == (self.numTimes, self.numOutputs)) for data in dataFrames]) --not true
        assert all([(data.shape[1] == self.numOutputs) for data in dataFrames])

        #parse ynorm
        self.ynorm = ynorm
        assert not hasattr(ynorm, "__len__") or len(ynorm) == self.numOutputs, "%g %d"%(
            len(ynorm) if hasattr(ynorm, "__len__") else ynorm, self.numOutputs)

        #parse replace zero
        self.replaceZero = replaceZero

        #parse ICs
        # numPatients = len(dataFrames)
        # self.ICs_list = ICs
        # assert len(ICs) == numPatients
        # assert len(ICs[0]) == self.numOutputs

        #Offset data, because pints doesn't allow negative times
        # if any([tarr[0]<0 for tarr in self.dataTimes_list]):
        #     self.data_t0_list = [tarr[0] for tarr in self.dataTimes_list]
        #     for i, t0 in enumerate(self.data_t0_list):
        #         self.dataTimes_list[i] -= t0
        #         self.dataFrame_list[i].index = self.dataTimes_list[i]
        # else:
        #     self.data_t0_list = [0 for tarr in self.dataTimes_list]

        #WHICH patient this is, which will be fed to copies of this class
        # self.patientIndex = None

        #other names and numbers
        self.outputNames = list(dataFrames[0].columns)
        assert all([(df.columns == self.outputNames).all() for df in dataFrames])
        self.paramNames = list(paramNames)
        self.numParams = len(paramNames)

        #SBML specific stuff
        self._model = None
        self.simulator = lambda: None
        self.simulator.set_protocol = lambda self, dosingRegimen: 1
        self._set_number_and_names()
        self._time_unit = self._get_time_unit()
        self._has_sensitivities = False

        super(MechanisticModel, self).__init__() #calls MechanisticModel.__init__.

    def _get_time_unit(self):
        """
        Gets the model's time unit.
        """
        return 1

    def _set_state_and_const(self, parameters, undoAlphabeticalOrder=True):
        parameters = np.array(parameters)
        if len(parameters)==self._n_parameters: #given ICs and actual parameters
            states, consts = parameters[:self._n_states], parameters[self._n_states:]
        elif len(parameters)==self._n_consts: #given actual parameters only
            states, consts = None, parameters
            raise NotImplementedError("Cannot handle not being given states (ICs) and constants (parameters)")
        else:
            raise ValueError("Unexpected length of parameters %d, compared to n_states %d and n_consts %d"%(
                len(parameters), self._n_states, self._n_consts
            ))
        
        if undoAlphabeticalOrder:
            consts = consts[self._const_original_order] 
            if states is not None:
                states = states[self._state_original_order]

        # #If initial conditions not given, because they're a fixed parameter
        # if states is None or (states==0).all():
        #     assert self.patientIndex is not None, \
        #         "If ICs are not given, the copies of the mechanistic model must be given a patient index." 
        #     states = self.ICs_list[self.patientIndex]

        self._set_state(states)
        self._set_const(consts)
        return states, consts

    def _set_const(self, consts):
        """
        Sets values of constant model parameters (real parameters).
        """
        self.paramValues = consts

    def _set_state(self, states):
        """
        Sets initial values of states (variables).
        """
        self.ICs = states

    def _set_number_and_names(self):
        """
        Sets the number of states, parameters and outputs, as well as their
        names. If the model is ``None`` the self._model is taken.
        """
        # Get the number of states and parameters
        self._n_states = self.numOutputs #non-constant variables (outputs)
        self._n_consts = self.numParams  #constant parameters (inputs)
        self._n_parameters = self.numParams + self.numOutputs #all "parameters" (inputs and outputs)

        # Get constant variable names and state names
        # names = self.outputNames + self.paramNames
        self._state_names = sorted(self.outputNames)
        self._const_names = sorted(self.paramNames)

        # Remember original order of state names for simulation
        self._state_order_after_sort = np.argsort(self.outputNames)
        self._state_original_order = np.argsort(self._state_order_after_sort)
        self._const_order_after_sort = np.argsort(self.paramNames)
        self._const_original_order = np.argsort(self._const_order_after_sort)
        #Note:
        # names_after_sort = np.array(names)[order_after_sort]
        # original_names = names_after_sort[original_order]
        # original_names == names

        # Set default parameter names
        self._parameter_names = self._state_names + self._const_names #all variables (outputs) and parameters (inputs)

        # Set default outputs
        self._output_names = self._state_names #all variables (outputs)
        self._n_outputs = self._n_states

        # Create references of displayed parameter and output names to
        # orginal myokit names (defaults to identity map)
        # (Key: myokit name, value: displayed name)
        self._parameter_name_map = dict(
            zip(self._parameter_names, self._parameter_names))
        self._output_name_map = dict(
            zip(self._output_names, self._output_names))

    def copy(self):
        """
        Returns a deep copy of the mechanistic model.
        .. note::
            Copying the model resets the sensitivity settings.
        """
        return copy(self)

    def enable_sensitivities(self, enabled, parameter_names=None):
        """
        Enables the computation of the model output sensitivities to the model
        parameters if set to ``True``.
        The sensitivities are computed using the forward sensitivities method,
        where an ODE for each sensitivity is derived. The sensitivities are
        returned together with the solution to the orginal system of ODEs when
        simulating the mechanistic model :meth:`simulate`.
        The optional parameter names argument can be used to set which
        sensitivities are computed. By default the sensitivities to all
        parameters are computed.
        :param enabled: A boolean flag which enables (``True``) / disables
            (``False``) the computation of sensitivities.
        :type enabled: bool
        :param parameter_names: A list of parameter names of the model. If
            ``None`` sensitivities for all parameters are computed.
        :type parameter_names: list[str], optional
        """
        raise NotImplementedError("No sensitivites enabled in the NoSBML Mechanistic Model.")

    def has_sensitivities(self):
        """
        Returns a boolean indicating whether sensitivities have been enabled.
        """
        return False

    def n_outputs(self):
        """
        Returns the number of output dimensions.
        By default this is the number of states.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters in the model.
        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._n_parameters

    def outputs(self):
        """
        Returns the output names of the model.
        """
        # Get user specified output names
        output_names = [
            self._output_name_map[name] for name in self._output_names]
        return output_names

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        # Get user specified parameter names
        parameter_names = [
            self._parameter_name_map[name] for name in self._parameter_names]

        return parameter_names

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.
        The outputs can be set to any quantifiable variable name of the
        :class:`myokit.Model`, e.g. `compartment.variable`.
        .. note::
            Setting outputs resets the sensitivity settings (by default
            sensitivities are disabled.)
        :param outputs:
            A list of output names.
        :type outputs: list[str]
        """
        return None

    def set_output_names(self, names):
        """
        Assigns names to the model outputs. By default the
        :class:`myokit.Model` names are assigned to the outputs.
        :param names: A dictionary that maps the current output names to new
            names.
        :type names: dict[str, str]
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current output names'
                'as keys and the new output names as values.')

        # Check that new output names are unique
        new_names = list(names.values())
        n_unique_new_names = len(set(names.values()))
        if len(new_names) != n_unique_new_names:
            raise ValueError(
                'The new output names have to be unique.')

        # Check that new output names do not exist already
        for new_name in new_names:
            if new_name in list(self._output_name_map.values()):
                raise ValueError(
                    'The output names cannot coincide with existing '
                    'output names. One output is already called '
                    '<' + str(new_name) + '>.')

        # Replace currently displayed names by new names
        for myokit_name in self._output_names:
            old_name = self._output_name_map[myokit_name]
            try:
                new_name = names[old_name]
                self._output_name_map[myokit_name] = str(new_name)
            except KeyError:
                # KeyError indicates that the current output is not being
                # renamed.
                pass

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.
        :param names: A dictionary that maps the current parameter names to new
            names.
        :type names: dict[str, str]
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current parameter names'
                'as keys and the new parameter names as values.')

        # Check that new parameter names are unique
        new_names = list(names.values())
        n_unique_new_names = len(set(names.values()))
        if len(new_names) != n_unique_new_names:
            raise ValueError(
                'The new parameter names have to be unique.')

        # Check that new parameter names do not exist already
        for new_name in new_names:
            if new_name in list(self._parameter_name_map.values()):
                raise ValueError(
                    'The parameter names cannot coincide with existing '
                    'parameter names. One parameter is already called '
                    '<' + str(new_name) + '>.')

        # Replace currently displayed names by new names
        for myokit_name in self._parameter_names:
            old_name = self._parameter_name_map[myokit_name]
            try:
                new_name = names[old_name]
                self._parameter_name_map[myokit_name] = str(new_name)
            except KeyError:
                # KeyError indicates that the current parameter is not being
                # renamed.
                pass

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs (and optionally
        the sensitivites) for the specified parameters and times.
        The model outputs are returned as a 2 dimensional NumPy array of shape
        (n_outputs, n_times). If sensitivities are enabled, a tuple is returned
        with the NumPy array of the model outputs and a NumPy array of the
        sensitivities of shape (n_times, n_outputs, n_parameters).
        :param parameters: An array-like object with values for the model
            parameters (initial conditions sorted into alphabetical order,
            then constant parameters sorted into alphabetical order).
        :type parameters: list, numpy.ndarray
        :param times: An array-like object with time points at which the output
            values are returned.
        :type times: list, numpy.ndarray
        """

        # Set initial conditions and constant model parameters
        ICs, params = self._set_state_and_const(parameters, undoAlphabeticalOrder=True)

        #Get model output
        yest = np.transpose(self.func(times, ICs, params)/self.ynorm)
        assert len(yest) == self._n_outputs
        yest = yest[self._state_order_after_sort]

        if self.replaceZero is not None:
            yest[yest<=0] = self.replaceZero
        return yest

    def time_unit(self):
        """
        Returns the model's unit of time.
        """
        return 1
