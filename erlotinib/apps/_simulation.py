#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import warnings

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd

import erlotinib as erlo
import erlotinib.apps as apps


class PDSimulationController(apps.BaseApp):
    """
    Creates an app which simulates a :class:`erlotinib.PharmacodynamicModel`.

    Parameter sliders can be used to adjust parameter values during
    the simulation.

    Extends :class:`BaseApp`.

    Example
    -------

    ::

        # Set up app with data and model
        app = PDSimulationController()
        app.add_model(model)
        app.add_data(data)

        # Define a simulation callback that updates the simulation according
        # to the sliders
        sliders = app.slider_ids()

        @app.app.callback(
            Output('fig', 'figure'),
            [Input(s, 'value') for s in sliders])
        def update_simulation(*args):
            parameters = args
            fig = app.update_simulation(parameters)

            return fig

        # Start the app
        app.start_application()
    """

    def __init__(self):
        super(PDSimulationController, self).__init__(
            name='PDSimulationController')

        # Instantiate figure and sliders
        self._fig = erlo.plots.PDTimeSeriesPlot(updatemenu=False)
        self._sliders = _SlidersComponent()

        # Create default layout
        self._set_layout()

        # Create defaults
        self._model = None
        self._times = np.linspace(start=0, stop=30)

    def _add_simulation(self):
        """
        Adds trace of simulation results to the figure.
        """
        # Make sure that parameters and sliders are ordered the same
        if self._model.parameters() != list(self._sliders.sliders().keys()):
            raise Warning('Model parameters do not align with slider.')

        # Get parameter values
        parameters = []
        for slider in self._sliders.sliders().values():
            value = slider.value
            parameters.append(value)

        # Add simulation to figure
        result = self._simulate(parameters)
        self._fig.add_simulation(result)

        # Remember index of model trace for update callback
        n_traces = len(self._fig._fig.data)
        self._model_trace = n_traces - 1

    def _create_figure_component(self):
        """
        Returns a figure component.
        """
        figure = dbc.Col(
            children=[dcc.Graph(
                figure=self._fig._fig,
                id='fig',
                style={'height': '67vh'})],
            md=9
        )

        return figure

    def _create_sliders(self):
        """
        Creates one slider for each parameter, and groups the slider by
        1. Pharmacokinetic input
        2. Initial values (of states)
        3. Parameters
        """
        parameters = self._model.parameters()
        # Add one slider for each parameter
        for parameter in parameters:
            self._sliders.add_slider(slider_id=parameter)

        # Split parameters into initial values, and parameters
        n_states = self._model._n_states
        states = parameters[:n_states]
        parameters = parameters[n_states:]

        # Group parameters:
        # Create PK input slider group
        pk_input = self._model.pk_input()
        if pk_input is not None:
            self._sliders.group_sliders(
                slider_ids=[pk_input], group_id='Pharmacokinetic input')

            # Make sure that pk input is not assigned to two sliders
            parameters.remove(pk_input)

        # Create initial values slider group
        self._sliders.group_sliders(
            slider_ids=states, group_id='Initial values')

        # Create parameters slider group
        self._sliders.group_sliders(
            slider_ids=parameters, group_id='Parameters')

    def _create_sliders_component(self):
        """
        Returns a slider component.
        """
        sliders = dbc.Col(
            children=self._sliders(),
            md=3,
            style={'marginTop': '5em'}
        )

        return sliders

    def _set_layout(self):
        """
        Sets the layout of the app.

        - Plot of simulation/data on the left.
        - Parameter sliders on the right.
        """
        self.app.layout = dbc.Container(
            children=[dbc.Row([
                self._create_figure_component(),
                self._create_sliders_component()])],
            style={'marginTop': '5em'})

    def _simulate(self, parameters):
        """
        Returns simulation of pharmacodynamic model in standard format, i.e.
        pandas.DataFrame with 'Time' and 'Biomarker' column.
        """
        # Solve the model
        result = self._model.simulate(parameters, self._times)

        # Rearrange results into a pandas.DataFrame
        result = pd.DataFrame({'Time': self._times, 'Biomarker': result[0, :]})

        return result

    def add_data(
            self, data, id_key='ID', time_key='Time', biom_key='Biomarker'):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time and a PD
        biomarker column, and adds a scatter plot of the biomarker time series
        to the figure. Each individual receives a unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        id_key
            Key label of the :class:`DataFrame` which specifies the ID column.
            The ID refers to the identity of an individual. Defaults to
            ``'ID'``.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        """
        # Add data to figure
        self._fig.add_data(data, id_key, time_key, biom_key)

        # Set axes labels to time_key and biom_key
        self._fig.set_axis_labels(xlabel=time_key, ylabel=biom_key)

    def add_model(self, model):
        """
        Adds a :class:`erlotinib.PharmacodynamicModel` to the application.

        One parameter slider is generated for each model parameter, and
        the solution for a default set of parameters is added to the figure.
        """
        if self._model is not None:
            # This is a temporary fix! In a future issue we will handle the
            # simulation of multiple models
            warnings.warn(
                'A model has been set previously. The passed model was '
                'therefore ignored.')

            return None

        if not isinstance(model, erlo.PharmacodynamicModel):
            raise TypeError(
                'Model has to be an instance of '
                'erlotinib.PharmacodynamicModel.')

        self._model = model

        # Add one slider for each parameter to the app
        self._create_sliders()

        # Add simulation of model to the figure
        self._add_simulation()

        # Update layout
        self._set_layout()

    def slider_ids(self):
        """
        Returns a list of the slider ids.
        """
        return list(self._sliders.sliders().keys())

    def update_simulation(self, parameters):
        """
        Simulates the model for the provided parameters and replaces the
        current simulation plot by the new one.
        """
        # Solve model
        result = self._model.simulate(parameters, self._times).flatten()

        # Replace simulation values in plotly.Figure
        self._fig._fig.data[self._model_trace].y = result

        return self._fig._fig


class PKSimulationController(apps.BaseApp):
    """
    Creates an app which simulates a :class:`erlotinib.PharmacokineticModel`.

    Parameter sliders can be used to adjust parameter values and the dosing
    regimen during the simulation.

    Extends :class:`BaseApp`.

    Example
    -------

    ::

        # Set up app with data and model
        app = PKSimulationController()
        app.add_model(model)
        app.add_data(data)

        # Define a simulation callback that updates the simulation according
        # to the sliders
        sliders = app.slider_ids()

        @app.app.callback(
            Output('fig', 'figure'),
            [Input(s, 'value') for s in sliders])
        def update_simulation(*args):
            parameters = args
            fig = app.update_simulation(parameters)

            return fig

        # Start the app
        app.start_application()
    """

    def __init__(self):
        super(PKSimulationController, self).__init__(
            name='PKSimulationController')

        # Instantiate figure and sliders
        self._fig = erlo.plots.PKTimeSeriesPlot(updatemenu=False)
        self._sliders = _SlidersComponent()

        # Create default layout
        self._set_layout()

        # Create defaults
        self._model = None
        self._times = np.linspace(start=0, stop=30, num=300)
        self._slider_min = 0
        self._slider_max = 10

    def _add_simulation(self):
        """
        Adds trace of simulation results to the figure.
        """
        # Make sure that parameters and sliders are ordered the same
        n_expected = sorted(
            ['Dose in mg', 'Dose duration in day'] + self._model.parameters())
        if n_expected != sorted(list(self._sliders.sliders().keys())):
            raise Warning('Model parameters do not align with slider.')

        # Get parameter values
        parameters = []
        for slider in self._sliders.sliders().values():
            value = slider.value
            parameters.append(value)

        # TODO: add dose info
        # Add simulation to figure
        result = self._simulate(parameters)
        self._fig.add_simulation(result)

        # Remember index of model trace for update callback
        n_traces = len(self._fig._fig.data)
        self._model_trace = n_traces - 1

    def _create_figure_component(self):
        """
        Returns a figure component.
        """
        figure = dbc.Col(
            children=[dcc.Graph(
                figure=self._fig._fig,
                id='fig',
                style={'height': '67vh'})],
            md=9
        )

        return figure

    def _create_sliders(self):
        """
        Creates one slider for each parameter, and the dosing regimen
        specifications. The sliders are grouped in
        1. Dosing regimen
        2. Initial values (of states)
        3. Parameters
        """
        # Add sliders for dosing regimen
        dose_sliders = [
            'Dose in mg',
            'Dose duration in day']
        self._n_dose_sliders = len(dose_sliders)
        for slider in dose_sliders:
            self._sliders.add_slider(
                slider_id=slider, min_value=0.001, max_value=1)

        parameters = self._model.parameters()
        # Add one slider for each parameter
        for parameter in parameters:
            self._sliders.add_slider(
                slider_id=parameter, min_value=self._slider_min, 
                max_value=self._slider_max)

        # Create dosing regimen slider group
        self._sliders.group_sliders(
            slider_ids=dose_sliders, group_id='Dosing regimen')

        # Split parameters into initial values, and parameters
        n_states = self._model._n_states
        states = parameters[:n_states]
        parameters = parameters[n_states:]

        # Group parameters:
        # Create initial values slider group
        self._sliders.group_sliders(
            slider_ids=states, group_id='Initial values')

        # Create parameters slider group
        self._sliders.group_sliders(
            slider_ids=parameters, group_id='Parameters')

    def _create_sliders_component(self):
        """
        Returns a slider component.
        """
        sliders = dbc.Col(
            children=self._sliders(),
            md=3,
            style={'marginTop': '5em'}
        )

        return sliders

    def _set_layout(self):
        """
        Sets the layout of the app.

        - Plot of simulation/data on the left.
        - Parameter sliders on the right.
        """
        self.app.layout = dbc.Container(
            children=[dbc.Row([
                self._create_figure_component(),
                self._create_sliders_component()])],
            style={'marginTop': '5em'})

    def _simulate(self, parameters):
        """
        Returns simulation of pharmacodynamic model in standard format, i.e.
        pandas.DataFrame with 'Time' and 'Biomarker' column.
        """
        # Solve the model
        result = self._model.simulate(parameters, self._times)

        # Rearrange results into a pandas.DataFrame
        result = pd.DataFrame({'Time': self._times, 'Biomarker': result[0, :]})

        return result

    def add_data(
            self, data, id_key='ID', time_key='Time', biom_key='Biomarker',
            dose_key='Dose'):
        """
        Adds pharmacokinetic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time and a PD
        biomarker column, and adds a scatter plot of the biomarker time series
        to the figure. Each individual receives a unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        id_key
            Key label of the :class:`DataFrame` which specifies the ID column.
            The ID refers to the identity of an individual. Defaults to
            ``'ID'``.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        dose_key
            Key label of the :class:`DataFrame` which specifies the dose
            column. Defaults to ``'Dose'``.
        """
        # Add data to figure
        self._fig.add_data(data, id_key, time_key, biom_key, dose_key)

        # Set axes labels to time_key and biom_key
        self._fig.set_axis_labels(time_key, biom_key, dose_key)

    def add_model(self, model):
        """
        Adds a :class:`erlotinib.PharmacokineticModel` to the application.

        One parameter slider is generated for each model parameter, and the
        dose amount as well the dose duration. The solution of the simulation
        for a default set of parameters is added to the figure.
        """
        if self._model is not None:
            # This is a temporary fix! In a future issue we will handle the
            # simulation of multiple models
            warnings.warn(
                'A model has been set previously. The passed model was '
                'therefore ignored.')

            return None

        if not isinstance(model, erlo.PharmacokineticModel):
            raise TypeError(
                'Model has to be an instance of '
                'erlotinib.PharmacokineticModel.')

        self._model = model

        # Add one slider for each parameter to the app
        self._create_sliders()

        # Add simulation of model to the figure
        self._add_simulation()

        # Update layout
        self._set_layout()

    def slider_ids(self):
        """
        Returns a list of the slider ids.
        """
        return list(self._sliders.sliders().keys())

    def update_simulation(self, parameters):
        """
        Simulates the model for the provided parameters and replaces the
        current simulation plot by the new one.
        """
        # Sort parameters in dosing regimen and model parameters
        dose, duration = parameters[:self._n_dose_sliders]
        parameters = parameters[self._n_dose_sliders:]

        # Solve model
        self._model.set_dosing_regimen(dose, 3, 1, duration)
        result = self._model.simulate(parameters, self._times).flatten()

        # Replace simulation values in plotly.Figure
        self._fig._fig.data[self._model_trace].y = result

        return self._fig._fig


class _SlidersComponent(object):
    """
    A helper class that helps to organise the sliders of the
    :class:`SimulationController`.

    The sliders are arranged horizontally. Sliders may be grouped by meaning.
    """

    def __init__(self):
        # Set defaults
        self._sliders = {}
        self._slider_groups = {}

    def __call__(self):
        # Returns the contents in form of a list of dash components.

        # If no sliders have been added, print a default message.
        if not self._sliders:
            default = [dbc.Alert(
                "No model has been chosen.", color="primary")]
            return default

        # If sliders have not been grouped, print a default message.
        if not self._sliders:
            default = [dbc.Alert(
                "Sliders have not been grouped.", color="primary")]
            return default

        # Group and label sliders
        contents = self._compose_contents()
        return contents

    def _compose_contents(self):
        """
        Returns the grouped sliders with labels as a list of dash components.
        """
        contents = []
        for group_id in self._slider_groups.keys():
            # Create label for group
            group_label = html.Label(group_id)

            # Group sliders
            group = self._slider_groups[group_id]
            container = []
            for slider_id in group:
                # Create label for slider
                label = html.Label(slider_id, style={'fontSize': '0.8rem'})
                slider = self._sliders[slider_id]

                # Add label and slider to group container
                container += [
                    dbc.Col(children=[label], width=12),
                    dbc.Col(children=[slider], width=12)]

            # Convert slider group to dash component
            group = dbc.Row(
                children=container, style={'marginBottom': '1em'})

            # Add label and group to contents
            contents += [group_label, group]

        return contents

    def add_slider(
            self, slider_id, value=0.5, min_value=0, max_value=2,
            step_size=0.01):
        """
        Adds a slider.

        Parameters
        ----------
        slider_id
            ID of the slider.
        value
            Default value of the slider.
        min_value
            Minimal value of slider.
        max_value
            Maximal value of slider.
        step_size
            Elementary step size of slider.
        """
        if '.' in slider_id:
            raise ValueError(
                'The parameter names are currently used as slider ids, and '
                'slider ids do not allow ".".')

        self._sliders[slider_id] = dcc.Slider(
            id=slider_id,
            value=value,
            min=min_value,
            max=max_value,
            step=step_size,
            marks={
                str(min_value): str(min_value),
                str(max_value): str(max_value)},
            updatemode='drag')

    def group_sliders(self, slider_ids, group_id):
        """
        Visually groups sliders. Group ID will be used as label.

        Each slider can only be in one group.
        """
        # Check that incoming sliders do not belong to any group already
        for index, existing_group in enumerate(self._slider_groups.values()):
            for slider in slider_ids:
                if slider in existing_group:
                    raise ValueError(
                        'Slider <' + str(slider) + '> exists already in group '
                        '<' + str(self._slider_groups.keys()[index]) + '>.')

        self._slider_groups[group_id] = slider_ids

    def sliders(self):
        """
        Returns a dictionary of slider objects with the slider ID as key and
        the slider object as value.
        """
        return self._sliders


# For simple debugging the app can be launched by executing the python file.
if __name__ == "__main__":

    from dash.dependencies import Input, Output

    'PD simulation example'
    # # Get data and model
    # data = erlo.DataLibrary().lung_cancer_control_group(True)
    # data = data.rename(columns={
    #     'Time': 'Time in day', 'Biomarker': 'Tumour volume in cm^3'})
    # path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
    # model = erlo.PharmacodynamicModel(path)
    # model.set_parameter_names(names={
    #     'myokit.drug_concentration': 'Drug concentration in mg/L',
    #     'myokit.tumour_volume': 'Tumour volume in cm^3',
    #     'myokit.kappa': 'Potency in L/mg/day',
    #     'myokit.lambda_0': 'Exponential growth rate in 1/day',
    #     'myokit.lambda_1': 'Linear growth rate in cm^3/day'})

    # # Set up demo app
    # app = PDSimulationController()
    # app.add_model(model)
    # app.add_data(
    #     data, time_key='Time in day', biom_key='Tumour volume in cm^3')

    # # Define a simulation callback
    # sliders = app.slider_ids()

    # @app.app.callback(
    #     Output('fig', 'figure'),
    #     [Input(s, 'value') for s in sliders])
    # def update_simulation(*args):
    #     """
    #     Simulates the model for the current slider values and updates the
    #     model plot in the figure.
    #     """
    #     parameters = args
    #     fig = app.update_simulation(parameters)

    #     return fig

    # app.start_application(debug=True)

    'PK simulation example'
    # Get data and model
    data = erlo.DataLibrary().lung_cancer_medium_erlotinib_dose_group()
    data = data.rename(columns={
        '#ID': 'ID',
        'TIME in day': 'Time in day',
        'DOSE in mg': 'Dose in mg',
        'PLASMA CONCENTRATION in mg/L': 'Plasma conc. in mg/L'})
    path = erlo.ModelLibrary().one_compartment_pk_model()
    model = erlo.PharmacokineticModel(path)
    model.set_administration(compartment='central', direct=False)
    model.set_dosing_regimen(dose=1, start=3, period=1)
    model.set_parameter_names(names={
        'central.drug_amount': 'Central drug amount in mg',
        'dose.drug_amount': 'Dose drug amount in mg',
        'central.size': 'Volume of distribution in L',
        'dose.absorption_rate': 'Absorption rate in 1/day',
        'myokit.elimination_rate': 'Elimination rate in 1/day'})

    # Set up demo app
    app = PKSimulationController()
    app.add_model(model)
    app.add_data(
        data, time_key='Time in day', biom_key='Plasma conc. in mg/L',
        dose_key='Dose in mg')

    # Define a simulation callback
    sliders = app.slider_ids()

    @app.app.callback(
        Output('fig', 'figure'),
        [Input(s, 'value') for s in sliders])
    def update_simulation(*args):
        """
        Simulates the model for the current slider values and updates the
        model plot in the figure.
        """
        parameters = args
        fig = app.update_simulation(parameters)

        return fig

    app.start_application(debug=True)
