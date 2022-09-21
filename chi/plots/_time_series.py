#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import matplotlib.colors as mplc
from seaborn import color_palette
from chi import plots


class PDPredictivePlot(plots.SingleFigure):
    """
    A figure class that visualises the predictions of a predictive
    pharmacodynamic model.

    Extends :class:`SingleFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PDPredictivePlot, self).__init__(updatemenu)

    def _add_data_trace(self, _id, times, measurements, color, measurementErrors=None):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        _measurementErrors = None if measurementErrors is None else \
            measurementErrors if isinstance(measurementErrors, go.scatter.ErrorY) else \
            go.scatter.ErrorY(array=measurementErrors) if len(measurementErrors)==len(measurements) else \
            go.scatter.ErrorY(
                    array=measurementErrors[1], arrayminus=measurementErrors[0]
                ) if len(measurementErrors)==2 else None
                
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=measurements,
                error_y=_measurementErrors,
                name="ID: %s" % str(_id),
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def _add_prediction_scatter_trace(self, times, samples, colourIndex=None, legendLabel=None, n_colors=10):
        """
        Adds scatter plot of samples from the predictive model.
        """
        # Get colour (light blueish)
        colours = [mplc.rgb2hex(c) for c in color_palette("pastel", n_colors=n_colors)]
        colour = colours[0 if colourIndex is None else colourIndex%n_colors]
        # colour = plotly.colors.qualitative.Pastel2[1 if colourIndex is None else colourIndex]

        # Add trace
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=samples,
                name="Predicted samples" if legendLabel is None else legendLabel,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=colour,
                    opacity=0.7 if colourIndex is None else 0.5,
                    line=dict(color='black', width=1))))

    def _add_prediction_bulk_prob_trace(self, data, colourIndex=None, legendLabel=None, n_colors=10):
        """
        Adds the bulk probabilities as two line plots (one for upper and lower
        limit) and shaded area to the figure.
        """
        # Construct times that go from min to max and back to min
        # (Important for shading with 'toself')
        times = data['Time'].unique()
        times = np.hstack([times, times[::-1]])

        # Get unique bulk probabilities and sort in descending order
        bulk_probs = data['Bulk probability'].unique()
        bulk_probs[::-1].sort()

        # Get colors (shift start a little bit, because 0th level is too light)
        n_traces = len(bulk_probs)
        if colourIndex is None:
            shift = 2
            colors = plotly.colors.sequential.Blues[shift:shift+n_traces]
        else:
            # colors = plotly.colors.qualitative.Pastel2
            colors = [mplc.rgb2hex(c) for c in color_palette("pastel", n_colors=n_colors)]

        # Add traces
        for trace_id, bulk_prob in enumerate(bulk_probs):
            # Get relevant upper and lower percentiles
            mask = data['Bulk probability'] == bulk_prob
            reduced_data = data[mask]

            upper = reduced_data['Upper'].to_numpy()
            lower = reduced_data['Lower'].to_numpy()
            values = np.hstack([upper, lower[::-1]])

            # Add trace
            self._fig.add_trace(go.Scatter(
                x=times,
                y=values,
                line=dict(width=1, color=colors[trace_id if colourIndex is None else colourIndex]),
                fill='toself',
                legendgroup='Model prediction',
                name='Predictive model' if legendLabel is None else legendLabel,
                text="%s Bulk" % bulk_prob,
                hoverinfo='text',
                showlegend=True if trace_id == n_traces-1 else False))

    def _compute_bulk_probs(self, data, bulk_probs, time_key, sample_key):
        """
        Computes the upper and lower percentiles from the predictive model
        samples, corresponding to the provided bulk probabilities.
        """
        # Create container for perecentiles
        container = pd.DataFrame(columns=[
            'Time', 'Upper', 'Lower', 'Bulk probability'])

        # Translate bulk probabilities into percentiles
        percentiles = []
        for bulk_prob in bulk_probs:
            lower = 0.5 - bulk_prob / 2
            upper = 0.5 + bulk_prob / 2

            percentiles.append([bulk_prob, lower, upper])

        # Get unique times
        unique_times = data[time_key].unique()

        # Fill container with percentiles for each time
        for time in unique_times:
            # Mask relevant data
            mask = data[time_key] == time
            reduced_data = data[mask]

            # Get percentiles
            percentile_df = reduced_data[sample_key].rank(
                pct=True)
            for item in percentiles:
                bulk_prob, lower, upper = item

                # Get biomarker value corresponding to percentiles
                mask = percentile_df <= lower
                if sum(mask)>0:
                    biom_lower = reduced_data[mask][sample_key].max()
                else:
                    #More than the required percentile of the data has the same, minimum value
                    # e.g., more than 5% of the data is equal to 0
                    biom_lower = np.min(reduced_data[sample_key])

                mask = percentile_df >= upper
                biom_upper = reduced_data[mask][sample_key].min() if sum(mask)>0 else np.max(reduced_data[sample_key])

                # Append percentiles to container
                container = container.append(pd.DataFrame({
                    'Time': [time],
                    'Lower': [biom_lower],
                    'Upper': [biom_upper],
                    'Bulk probability': [str(bulk_prob)]}))

        return container

    def add_data(
            self, data, measurementErrors=None, biomarker=None, id_key='ID', time_key='Time',
            biom_key='Biomarker', meas_key='Measurement', n_colors=10, dataErrs=None):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time, a PD
        biomarker and a measurement column, and adds a scatter plot of the
        measurement time series to the figure. Each individual receives a
        unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        biomarker
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
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
        meas_key
            Key label of the :class:`DataFrame` which specifies the column of
            the measured PD biomarker. Defaults to ``'Measurement'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key, meas_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask]
        dataErrs_given_as_pm = dataErrs is not None and isinstance(dataErrs, list) and len(dataErrs)==2
        dataErrs = None if dataErrs is None else \
            [d[mask] for d in dataErrs] if dataErrs_given_as_pm else \
            dataErrs[mask]

        # Get a colour scheme
        colors = [mplc.rgb2hex(c) for c in color_palette("bright", n_colors=n_colors)]

        # Fill figure with scatter plots of individual data
        ids = data[id_key].unique()
        for index, _id in enumerate(ids):
            # Get individual data
            mask = data[id_key] == _id
            times = data[time_key][mask]
            measurements = data[meas_key][mask]
            measurementErrors = None if dataErrs is None else \
                [d[meas_key][mask] for d in dataErrs] if dataErrs_given_as_pm else \
                dataErrs[meas_key][mask]
            
            color = colors[index % n_colors]

            # Create Scatter plot
            self._add_data_trace(_id, times, measurements, color, measurementErrors=measurementErrors)

    def add_prediction(
            self, data, biomarker=None, bulk_probs=[0.9], time_key='Time',
            biom_key='Biomarker', sample_key='Sample', colourIndex=None, legendLabel=None, n_colors=10):
        r"""
        Adds the prediction for the observable pharmacodynamic biomarker values
        to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a PD biomarker and a
        sample column. The time column determines the time of the biomarker
        measurement and the sample column the corresponding biomarker
        measurement. The biomarker column determines the biomarker type.

        A list of bulk probabilities ``bulk_probs`` can be specified, which are
        then added as area to the figure. The corresponding upper and lower
        percentiles are estimated from the ranks of the provided
        samples.

        .. warning::
            For low sample sizes the illustrated bulk probabilities may deviate
            significantly from the theoretical bulk probabilities. The upper
            and lower limit are determined from the rank of the samples for
            each time point.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        biomarker
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
        bulk_probs
            A list of bulk probabilities that are illustrated in the
            figure. If ``None`` the samples are illustrated as a scatter plot.
        time_key
            Key label of the :class:`pandas.DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`pandas.DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        sample_key
            Key label of the :class:`pandas.DataFrame` which specifies the
            sample column. Defaults to ``'Sample'``.
        colourIndex
            Optional. Which colour to select, to draw the prediction with.
        legendLabel
            Optional. Label to write on the legend for this prediction.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, biom_key, sample_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask]

        # Add samples as scatter plot if no bulk probabilites are provided, and
        # terminate method
        if bulk_probs is None:
            times = data[time_key]
            samples = data[sample_key]
            self._add_prediction_scatter_trace(times, samples, colourIndex, legendLabel, n_colors=n_colors)

            return None

        # Not more than 7 bulk probabilities are allowed (Purely aesthetic
        # criterion)
        if len(bulk_probs) > 7:
            raise ValueError(
                'At most 7 different bulk probabilities can be illustrated at '
                'the same time.')

        # Make sure that bulk probabilities are between 0 and 1
        bulk_probs = [float(probability) for probability in bulk_probs]
        for probability in bulk_probs:
            if (probability < 0) or (probability > 1):
                raise ValueError(
                    'The provided bulk probabilities have to between 0 and 1.')

        # Add bulk probabilities to figure
        percentile_df = self._compute_bulk_probs(
            data, bulk_probs, time_key, sample_key)
        self._add_prediction_bulk_prob_trace(percentile_df, colourIndex, legendLabel, n_colors=n_colors)


class PKPredictivePlot(plots.SingleSubplotFigure):
    """
    A figure class that visualises the predictions of a predictive
    pharmacokinetic model.

    Extends :class:`SingleSubplotFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PKPredictivePlot, self).__init__()

        self._create_template_figure(
            rows=2, cols=1, shared_x=True, row_heights=[0.2, 0.8])

        # Define legend name of prediction
        self._prediction_name = 'Predictive model'

        if updatemenu:
            self._add_updatemenu()

    def _add_dose_trace(
            self, _id, times, doses, durations, color,
            is_prediction=False):
        """
        Adds scatter plot of dose events to figure.
        """
        # Convert durations to strings
        durations = [
            'Dose duration: ' + str(duration) for duration in durations]

        name = "ID: %s" % str(_id)
        if is_prediction:
            name = 'Predictive model'

        # Add scatter plot of dose events
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=doses,
                name=name,
                legendgroup=name,
                showlegend=False,
                mode="markers",
                text=durations,
                hoverinfo='text',
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=1,
            col=1)

    def _add_biom_trace(self, _id, times, measurements, color, measurementErrors=None):
        """
        Adds scatter plot of an indiviudals pharamcokinetics to figure.
        """
        _measurementErrors = None if measurementErrors is None else \
            measurementErrors if isinstance(measurementErrors, go.scatter.ErrorY) else \
            go.scatter.ErrorY(array=measurementErrors) if len(measurementErrors)==len(measurements) else \
            go.scatter.ErrorY(
                    array=measurementErrors[1], arrayminus=measurementErrors[0]
                ) if len(measurementErrors)==2 else None
            
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=measurements,
                error_y=_measurementErrors,
                name="ID: %s" % str(_id),
                legendgroup="ID: %s" % str(_id),
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=2,
            col=1)

    def _add_updatemenu(self):
        """
        Adds a button to the figure that switches the biomarker scale from
        linear to logarithmic.
        """
        self._fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"yaxis2.type": "linear"}],
                            label="Linear y-scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis2.type": "log"}],
                            label="Log y-scale",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 0, "t": -10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

    def _add_prediction_scatter_trace(self, times, samples, colourIndex=None, legendLabel=None, n_colors=10):
        """
        Adds scatter plot of samples from the predictive model.
        """
        # Get colour (light blueish)
        # color = plotly.colors.qualitative.Pastel2[1 if colourIndex is None else colourIndex]
        # if colourIndex is None:
        #     colour = plotly.colors.qualitative.Pastel2[1]
        # else:
        #     colors = plotly.colors.qualitative.Plotly
        #     colourIndex = colourIndex%len(colors)
        #     colour = colors[colourIndex]
        colours = [mplc.rgb2hex(c) for c in color_palette("pastel", n_colors=n_colors)]
        colour = colours[0 if colourIndex is None else colourIndex%n_colors]

        # Add trace
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=samples,
                name="Predicted samples" if legendLabel is None else legendLabel,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=colour,
                    opacity=0.7 if colourIndex is None else 0.5,
                    line=dict(color='black', width=1))))

    def _add_prediction_bulk_prob_trace(self, data, colors, colourIndex=None, legendLabel=None, n_colors=10):
        """
        Adds the bulk probabilities as two line plots (one for upper and lower
        limit) and shaded area to the figure.
        """
        # Construct times that go from min to max and back to min
        # (Important for shading with 'toself')
        times = data['Time'].unique()
        times = np.hstack([times, times[::-1]])

        # Get unique bulk probabilities and sort in descending order
        bulk_probs = data['Bulk probability'].unique()
        bulk_probs[::-1].sort()

        # Get colors (shift start a little bit, because 0th level is too light)
        n_traces = len(bulk_probs)
        if colourIndex is None:
            shift = 2
            colors = plotly.colors.sequential.Blues[shift:shift+n_traces]
        else:
            # colors = plotly.colors.qualitative.Pastel2
            colors = [mplc.rgb2hex(c) for c in color_palette("pastel", n_colors=n_colors)]
            colourIndex = colourIndex%n_colors

        # Add traces
        n_traces = len(bulk_probs)
        for trace_id, bulk_prob in enumerate(bulk_probs):
            # Get relevant upper and lower percentiles
            mask = data['Bulk probability'] == bulk_prob
            reduced_data = data[mask]

            upper = reduced_data['Upper'].to_numpy()
            lower = reduced_data['Lower'].to_numpy()
            values = np.hstack([upper, lower[::-1]])

            # Add trace
            self._fig.add_trace(go.Scatter(
                x=times,
                y=values,
                line=dict(width=1, color=colors[trace_id if colourIndex is None else colourIndex]),
                fill='toself',
                legendgroup=self._prediction_name,
                name=self._prediction_name if legendLabel is None else legendLabel,
                text="%s Bulk" % bulk_prob,
                hoverinfo='text',
                showlegend=True if trace_id == n_traces-1 else False),
                row=2,
                col=1)

    def _compute_bulk_probs(self, data, bulk_probs, time_key, sample_key):
        """
        Computes the upper and lower percentiles from the predictive model
        samples, corresponding to the provided bulk probabilities.
        """
        # Create container for perecentiles
        container = pd.DataFrame(columns=[
            'Time', 'Upper', 'Lower', 'Bulk probability'])

        # Translate bulk probabilities into percentiles
        percentiles = []
        for bulk_prob in bulk_probs:
            lower = 0.5 - bulk_prob / 2
            upper = 0.5 + bulk_prob / 2

            percentiles.append([bulk_prob, lower, upper])

        # Get unique times
        unique_times = data[time_key].unique()

        # Fill container with percentiles for each time
        for time in unique_times:
            # Mask relevant data
            mask = data[time_key] == time
            reduced_data = data[mask]

            # Get percentiles
            percentile_df = reduced_data[sample_key].rank(
                pct=True)
            for item in percentiles:
                bulk_prob, lower, upper = item

                # Get biomarker value corresponding to percentiles
                mask = percentile_df <= lower
                if sum(mask)>0:
                    biom_lower = reduced_data[mask][sample_key].max()
                else:
                    #More than the required percentile of the data has the same, minimum value
                    # e.g., more than 5% of the data is equal to 0
                    biom_lower = np.min(reduced_data[sample_key])

                mask = percentile_df >= upper
                biom_upper = reduced_data[mask][sample_key].min() if sum(mask)>0 else np.max(reduced_data[sample_key])
                
                # Append percentiles to container
                container = container.append(pd.DataFrame({
                    'Time': [time],
                    'Lower': [biom_lower],
                    'Upper': [biom_upper],
                    'Bulk probability': [str(bulk_prob)]}))

        return container

    def add_data(
            self, data, biomarker=None, id_key='ID', time_key='Time',
            biom_key='Biomarker', meas_key='Measurement', dose_key='Dose',
            dose_duration_key='Duration', n_colors=10, dataErrs=None):
        """
        Adds pharmacokinetic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time, a PK
        biomarker and measurement column, and adds a scatter plot of the
        measurement time series to the figure. The dataframe is also expected
        to have information about the administered dose via a dose and a
        dose duration column. Each individual receives a unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        biomarker
            The measured bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
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
        meas_key
            Key label of the :class:`DataFrame` which specifies the column of
            the measured PD biomarker. Defaults to ``'Measurement'``.
        dose_key
            Key label of the :class:`DataFrame` which specifies the dose
            column. Defaults to ``'Dose'``.
        dose_duration_key
            Key label of the :class:`DataFrame` which specifies the dose
            duration column. Defaults to ``'Duration'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        keys = [
            id_key, time_key, biom_key, meas_key, dose_key, dose_duration_key]
        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Get dose information
        mask = data[dose_key].notnull()
        dose_data = data[mask][[id_key, time_key, dose_key, dose_duration_key]]

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask][[id_key, time_key, meas_key]]
        dataErrs_given_as_pm = dataErrs is not None and isinstance(dataErrs, list) and len(dataErrs)==2
        dataErrs = None if dataErrs is None else \
            [d[mask][[id_key, time_key, meas_key]] for d in dataErrs] if dataErrs_given_as_pm else \
            dataErrs[mask][[id_key, time_key, meas_key]]

        # Set axis labels to dataframe keys
        self.set_axis_labels(time_key, biom_key, dose_key)

        # Get a colour scheme
        colors = [mplc.rgb2hex(c) for c in color_palette("bright", n_colors=n_colors)]

        # Fill figure with scatter plots of individual data
        ids = data[id_key].unique()
        for index, _id in enumerate(ids):
            # Get doses applied to individual
            mask = dose_data[id_key] == _id
            dose_times = dose_data[time_key][mask]
            doses = dose_data[dose_key][mask]
            durations = dose_data[dose_duration_key][mask]

            # Get biomarker measurements
            mask = data[id_key] == _id
            times = data[time_key][mask]
            measurements = data[meas_key][mask]
            measurementErrors = None if dataErrs is None else \
                [d[meas_key][mask] for d in dataErrs] if dataErrs_given_as_pm else \
                dataErrs[meas_key][mask]

            # Get a color for the individual
            color = colors[index % n_colors]

            # Create scatter plot of dose events
            self._add_dose_trace(_id, dose_times, doses, durations, color)

            # Create Scatter plot
            self._add_biom_trace(_id, times, measurements, color, measurementErrors=measurementErrors)

    def add_prediction(
            self, data, biomarker=None, bulk_probs=[0.9], time_key='Time',
            biom_key='Biomarker', sample_key='Sample', dose_key='Dose',
            dose_duration_key='Duration', colourIndex=None, legendLabel=None, n_colors=10):
        r"""
        Adds the prediction for the observable pharmacokinetic biomarker values
        to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a PD biomarker and a
        sample column. The time column determines the time of the biomarker
        measurement and the sample column the corresponding biomarker
        measurement. The biomarker column determines the biomarker type. The
        dataframe is also expected to have information about the administered
        dose via a dose and a dose duration column.

        A list of bulk probabilities ``bulk_probs`` can be specified, which are
        then added as area to the figure. The corresponding upper and lower
        percentiles are estimated from the ranks of the provided
        samples.

        .. warning::
            For low sample sizes the illustrated bulk probabilities may deviate
            significantly from the theoretical bulk probabilities. The upper
            and lower limit are determined from the rank of the samples for
            each time point.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        biomarker
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
        bulk_probs
            A list of bulk probabilities that are illustrated in the
            figure. If ``None`` the samples are illustrated as a scatter plot.
        time_key
            Key label of the :class:`pandas.DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`pandas.DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        sample_key
            Key label of the :class:`pandas.DataFrame` which specifies the
            sample column. Defaults to ``'Sample'``.
        dose_key
            Key label of the :class:`DataFrame` which specifies the dose
            column. Defaults to ``'Dose'``.
        dose_duration_key
            Key label of the :class:`DataFrame` which specifies the dose
            duration column. Defaults to ``'Duration'``.
        colourIndex
            Optional. Which colour to select, to draw the prediction with.
        legendLabel
            Optional. Label to write on the legend for this prediction.        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        keys = [
            time_key, biom_key, sample_key, dose_key, dose_duration_key]
        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Get dose information
        mask = data[dose_key].notnull()
        dose_data = data[mask][[time_key, dose_key, dose_duration_key]]

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask][[time_key, sample_key]]

        # Set axis labels to dataframe keys
        self.set_axis_labels(time_key, biom_key, dose_key)

        # Add samples as scatter plot if no bulk probabilites are provided, and
        # terminate method
        if bulk_probs is None:
            times = data[time_key]
            samples = data[sample_key]
            self._add_prediction_scatter_trace(times, samples, colourIndex, legendLabel, n_colors=n_colors)

            return None

        # Not more than 7 bulk probabilities are allowed (Purely aesthetic
        # criterion)
        if len(bulk_probs) > 7:
            raise ValueError(
                'At most 7 different bulk probabilities can be illustrated at '
                'the same time.')

        # Make sure that bulk probabilities are between 0 and 1
        bulk_probs = [float(probability) for probability in bulk_probs]
        for probability in bulk_probs:
            if (probability < 0) or (probability > 1):
                raise ValueError(
                    'The provided bulk probabilities have to between 0 and 1.')

        # Define colour scheme
        shift = 2
        colors = plotly.colors.sequential.Blues[shift:]

        # Create scatter plot of dose events
        self._add_dose_trace(
            _id=None,
            times=dose_data[time_key],
            doses=dose_data[dose_key],
            durations=dose_data[dose_duration_key],
            color=colors[0],
            is_prediction=True)

        # Add bulk probabilities to figure
        percentile_df = self._compute_bulk_probs(
            data, bulk_probs, time_key, sample_key)
        self._add_prediction_bulk_prob_trace(percentile_df, colors, colourIndex, legendLabel, n_colors=n_colors)

    def set_axis_labels(self, time_label, biom_label, dose_label):
        """
        Sets the label of the time axis, the biomarker axis, and the dose axis.
        """
        self._fig.update_xaxes(title=time_label, row=2)
        self._fig.update_yaxes(title=dose_label, row=1)
        self._fig.update_yaxes(title=biom_label, row=2)


class PDTimeSeriesPlot(plots.SingleFigure):
    """
    A figure class that visualises measurements of a pharmacodynamic biomarker
    across multiple individuals.

    Measurements of a pharmacodynamic biomarker over time are visualised as a
    scatter plot.

    Extends :class:`SingleFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PDTimeSeriesPlot, self).__init__(updatemenu)

    def _add_data_trace(self, _id, times, measurements, color, measurementErrors=None):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        _measurementErrors = None if measurementErrors is None else \
            measurementErrors if isinstance(measurementErrors, go.scatter.ErrorY) else \
            go.scatter.ErrorY(array=measurementErrors) if len(measurementErrors)==len(measurements) else \
            go.scatter.ErrorY(
                    array=measurementErrors[1], arrayminus=measurementErrors[0]
                ) if len(measurementErrors)==2 else None
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=measurements,
                error_y=_measurementErrors,
                name="ID: %s" % str(_id),
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def _add_simulation_trace(self, times, biomarker):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=biomarker,
                name="Model",
                showlegend=True,
                mode="lines",
                line=dict(color='black')))

    def add_data(
            self, data, biomarker=None, id_key='ID', time_key='Time',
            biom_key='Biomarker', meas_key='Measurement', n_colors=10, dataErrs=None):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time, a PD
        biomarker and a measurement column, and adds a scatter plot of the
        measurement time series to the figure. Each individual receives a
        unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        biomarker
            The measured bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
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
        meas_key
            Key label of the :class:`DataFrame` which specifies the column of
            the measured PD biomarker. Defaults to ``'Measurement'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key, meas_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask]
        dataErrs_given_as_pm = dataErrs is not None and isinstance(dataErrs, list) and len(dataErrs)==2
        dataErrs = None if dataErrs is None else \
            [d[mask] for d in dataErrs] if dataErrs_given_as_pm else \
            dataErrs[mask]
        # Get a colour scheme
        # colors = plotly.colors.qualitative.Plotly
        colors = [mplc.rgb2hex(c) for c in color_palette("bright", n_colors=n_colors)]

        # Fill figure with scatter plots of individual data
        ids = data[id_key].unique()
        for index, _id in enumerate(ids):
            # Get individual data
            mask = data[id_key] == _id
            times = data[time_key][mask]
            measurements = data[meas_key][mask]
            measurementErrors = None if dataErrs is None else \
                [d[meas_key][mask] for d in dataErrs] if dataErrs_given_as_pm else \
                dataErrs[meas_key][mask]
            
            color = colors[index % n_colors]

            # Create Scatter plot
            self._add_data_trace(_id, times, measurements, color, measurementErrors=measurementErrors)

    def add_simulation(self, data, time_key='Time', biom_key='Biomarker'):
        """
        Adds a pharmacodynamic time series simulation to the figure.

        Expects a :class:`pandas.DataFrame` with a time and a PD biomarker
        column, and adds a line plot of the biomarker time series to the
        figure.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, biom_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        times = data[time_key]
        biomarker = data[biom_key]

        self._add_simulation_trace(times, biomarker)


class PKTimeSeriesPlot(plots.SingleSubplotFigure):
    """
    A figure class that visualises measurements of a pharmacokinetic biomarker
    across multiple individuals.

    Measurements of a pharmacokinetic biomarker over time are visualised as a
    scatter plot.

    Extends :class:`SingleSubplotFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PKTimeSeriesPlot, self).__init__()

        self._create_template_figure(
            rows=2, cols=1, shared_x=True, row_heights=[0.2, 0.8])

        if updatemenu:
            self._add_updatemenu()

    def _add_dose_trace(self, _id, times, doses, durations, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        # Convert durations to strings
        durations = [
            'Dose duration: ' + str(duration) for duration in durations]

        # Add scatter plot of dose events
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=doses,
                name="ID: %s" % str(_id),
                legendgroup="ID: %s" % str(_id),
                showlegend=False,
                mode="markers",
                text=durations,
                hoverinfo='text',
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=1,
            col=1)

    def _add_biom_trace(self, _id, times, measurements, color, measurementErrors=None):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        _measurementErrors = None if measurementErrors is None else \
            measurementErrors if isinstance(measurementErrors, go.scatter.ErrorY) else \
            go.scatter.ErrorY(array=measurementErrors) if len(measurementErrors)==len(measurements) else \
            go.scatter.ErrorY(
                    array=measurementErrors[1], arrayminus=measurementErrors[0]
                ) if len(measurementErrors)==2 else None

        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=measurements,
                error_y=_measurementErrors,
                name="ID: %s" % str(_id),
                legendgroup="ID: %s" % str(_id),
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=2,
            col=1)

    def _add_updatemenu(self):
        """
        Adds a button to the figure that switches the biomarker scale from
        linear to logarithmic.
        """
        self._fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"yaxis2.type": "linear"}],
                            label="Linear y-scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis2.type": "log"}],
                            label="Log y-scale",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 0, "t": -10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

    def add_data(
            self, data, biomarker=None, id_key='ID', time_key='Time',
            biom_key='Biomarker', meas_key='Measurement', dose_key='Dose',
            dose_duration_key='Duration', n_colors=10, dataErrs=None):
        """
        Adds pharmacokinetic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time, a PK
        biomarker and measurement column, and adds a scatter plot of the
        measurement time series to the figure. The dataframe is also expected
        to have information about the administered dose via a dose and a
        dose duration column. Each individual receives a unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        biomarker
            The measured bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
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
        meas_key
            Key label of the :class:`DataFrame` which specifies the column of
            the measured PD biomarker. Defaults to ``'Measurement'``.
        dose_key
            Key label of the :class:`DataFrame` which specifies the dose
            column. Defaults to ``'Dose'``.
        dose_duration_key
            Key label of the :class:`DataFrame` which specifies the dose
            duration column. Defaults to ``'Duration'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        keys = [
            id_key, time_key, biom_key, meas_key, dose_key, dose_duration_key]
        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Get dose information
        mask = data[dose_key].notnull()
        dose_data = data[mask][[id_key, time_key, dose_key, dose_duration_key]]

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask][[id_key, time_key, meas_key]]
        dataErrs_given_as_pm = dataErrs is not None and isinstance(dataErrs, list) and len(dataErrs)==2
        dataErrs = None if dataErrs is None else \
            [d[mask][[id_key, time_key, meas_key]] for d in dataErrs] if dataErrs_given_as_pm else \
            dataErrs[mask][[id_key, time_key, meas_key]]

        # Set axis labels to dataframe keys
        self.set_axis_labels(time_key, biom_key, dose_key)

        # Get a colour scheme
        # colors = plotly.colors.qualitative.Plotly
        colors = [mplc.rgb2hex(c) for c in color_palette("bright", n_colors=n_colors)]


        # Fill figure with scatter plots of individual data
        ids = data[id_key].unique()
        for index, _id in enumerate(ids):
            # Get doses applied to individual
            mask = dose_data[id_key] == _id
            dose_times = dose_data[time_key][mask]
            doses = dose_data[dose_key][mask]
            durations = dose_data[dose_duration_key][mask]

            # Get biomarker measurements
            mask = data[id_key] == _id
            times = data[time_key][mask]
            measurements = data[meas_key][mask]
            measurementErrors = None if dataErrs is None else \
                [d[meas_key][mask] for d in dataErrs] if dataErrs_given_as_pm else \
                dataErrs[meas_key][mask]
                
            # Get a color for the individual
            color = colors[index % n_colors]

            # Create scatter plot of dose events
            self._add_dose_trace(_id, dose_times, doses, durations, color)

            # Create Scatter plot
            self._add_biom_trace(_id, times, measurements, color, measurementErrors=measurementErrors)

    def add_simulation(
            self, data, time_key='Time', biom_key='Biomarker',
            dose_key='Dose'):
        """
        Adds a pharmacokinetic time series simulation to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a PK biomarker,
        and a dose column. A line plot of the biomarker time series, as well
        as the dosing regimen is added to the figure.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        """
        raise NotImplementedError

    def set_axis_labels(self, time_label, biom_label, dose_label):
        """
        Sets the label of the time axis, the biomarker axis, and the dose axis.
        """
        self._fig.update_xaxes(title=time_label, row=2)
        self._fig.update_yaxes(title=dose_label, row=1)
        self._fig.update_yaxes(title=biom_label, row=2)


class PDPredictiveSubPlots(plots.SingleSubplotFigure):
    """
    A figure class that visualises the predictions of a predictive
    pharmacodynamic model.

    Extends :class:`SingleSubplotFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    nrows
        Integer number of rows to pass to plotly for forming subplots.
    ncols
        Integer number of columns to pass to plotly for forming subplots.
    shared_x
        Boolean flag to pass to plotly if subplots should share x-axes.
    shared_y
        Boolean flag to pass to plotly if subplots should share y-axes.
    """

    def __init__(self, updatemenu=True, nrows=1, ncols=1, shared_x=False, shared_y=False):
        super(PDPredictiveSubPlots, self).__init__()

        self._create_template_figure(
            rows=nrows, cols=ncols, shared_x=shared_x, shared_y=shared_y)

        if updatemenu:
            self._add_updatemenu()

        self.nrows, self.ncols = nrows, ncols

    def _add_data_trace(self, _id, times, measurements, color, row, col, measurementErrors=None):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        _measurementErrors = None if measurementErrors is None else \
            measurementErrors if isinstance(measurementErrors, go.scatter.ErrorY) else \
            go.scatter.ErrorY(array=measurementErrors) if len(measurementErrors)==len(measurements) else \
            go.scatter.ErrorY(
                    array=measurementErrors[1], arrayminus=measurementErrors[0]
                ) if len(measurementErrors)==2 else None
                
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=measurements,
                error_y=_measurementErrors,
                name="ID: %s" % str(_id),
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
                row=row,
                col=col)

    def _add_prediction_scatter_trace(self, times, samples, row, col, colourIndex=None, legendLabel=None, n_colors=10):
        """
        Adds scatter plot of samples from the predictive model.
        """
        # Get colour (light blueish)
        colours = [mplc.rgb2hex(c) for c in color_palette("pastel", n_colors=n_colors)]
        colour = colours[0 if colourIndex is None else colourIndex%n_colors]
        # colour = plotly.colors.qualitative.Pastel2[1 if colourIndex is None else colourIndex]

        # Add trace
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=samples,
                name="Predicted samples" if legendLabel is None else legendLabel,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=colour,
                    opacity=0.7 if colourIndex is None else 0.5,
                    line=dict(color='black', width=1))),
                row=row,
                col=col)

    def _add_prediction_bulk_prob_trace(self, data, row, col, colourIndex=None, legendLabel=None, n_colors=10):
        """
        Adds the bulk probabilities as two line plots (one for upper and lower
        limit) and shaded area to the figure.
        """
        # Construct times that go from min to max and back to min
        # (Important for shading with 'toself')
        times = data['Time'].unique()
        times = np.hstack([times, times[::-1]])

        # Get unique bulk probabilities and sort in descending order
        bulk_probs = data['Bulk probability'].unique()
        bulk_probs[::-1].sort()

        # Get colors (shift start a little bit, because 0th level is too light)
        n_traces = len(bulk_probs)
        if colourIndex is None:
            shift = 2
            colors = plotly.colors.sequential.Blues[shift:shift+n_traces]
        else:
            # colors = plotly.colors.qualitative.Pastel2
            colors = [mplc.rgb2hex(c) for c in color_palette("pastel", n_colors=n_colors)]

        # Add traces
        for trace_id, bulk_prob in enumerate(bulk_probs):
            # Get relevant upper and lower percentiles
            mask = data['Bulk probability'] == bulk_prob
            reduced_data = data[mask]

            upper = reduced_data['Upper'].to_numpy()
            lower = reduced_data['Lower'].to_numpy()
            values = np.hstack([upper, lower[::-1]])

            # Add trace
            self._fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    line=dict(width=1, color=colors[trace_id if colourIndex is None else colourIndex]),
                    fill='toself',
                    legendgroup='Model prediction',
                    name='Predictive model' if legendLabel is None else legendLabel,
                    text="%s Bulk" % bulk_prob,
                    hoverinfo='text',
                    showlegend=True if trace_id == n_traces-1 else False),
                row=row,
                col=col)

    def _compute_bulk_probs(self, data, bulk_probs, time_key, sample_key):
        """
        Computes the upper and lower percentiles from the predictive model
        samples, corresponding to the provided bulk probabilities.
        """
        # Create container for perecentiles
        container = pd.DataFrame(columns=[
            'Time', 'Upper', 'Lower', 'Bulk probability'])

        # Translate bulk probabilities into percentiles
        percentiles = []
        for bulk_prob in bulk_probs:
            lower = 0.5 - bulk_prob / 2
            upper = 0.5 + bulk_prob / 2

            percentiles.append([bulk_prob, lower, upper])

        # Get unique times
        unique_times = data[time_key].unique()

        # Fill container with percentiles for each time
        for time in unique_times:
            # Mask relevant data
            mask = data[time_key] == time
            reduced_data = data[mask]

            # Get percentiles
            percentile_df = reduced_data[sample_key].rank(
                pct=True)
            for item in percentiles:
                bulk_prob, lower, upper = item

                # Get biomarker value corresponding to percentiles
                mask = percentile_df <= lower
                if sum(mask)>0:
                    biom_lower = reduced_data[mask][sample_key].max()
                else:
                    #More than the required percentile of the data has the same, minimum value
                    # e.g., more than 5% of the data is equal to 0
                    biom_lower = np.min(reduced_data[sample_key])

                mask = percentile_df >= upper
                biom_upper = reduced_data[mask][sample_key].min() if sum(mask)>0 else np.max(reduced_data[sample_key])

                # Append percentiles to container
                container = container.append(pd.DataFrame({
                    'Time': [time],
                    'Lower': [biom_lower],
                    'Upper': [biom_upper],
                    'Bulk probability': [str(bulk_prob)]}))

        return container

    def add_data(
            self, data, row, col, measurementErrors=None, biomarker=None, id_key='ID', time_key='Time',
            biom_key='Biomarker', meas_key='Measurement', n_colors=10, dataErrs=None):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time, a PD
        biomarker and a measurement column, and adds a scatter plot of the
        measurement time series to the figure. Each individual receives a
        unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        biomarker
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
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
        meas_key
            Key label of the :class:`DataFrame` which specifies the column of
            the measured PD biomarker. Defaults to ``'Measurement'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key, meas_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask]
        dataErrs_given_as_pm = dataErrs is not None and isinstance(dataErrs, list) and len(dataErrs)==2
        dataErrs = None if dataErrs is None else \
            [d[mask] for d in dataErrs] if dataErrs_given_as_pm else \
            dataErrs[mask]

        # Get a colour scheme
        colors = [mplc.rgb2hex(c) for c in color_palette("bright", n_colors=n_colors)]

        # Fill figure with scatter plots of individual data
        ids = data[id_key].unique()
        for index, _id in enumerate(ids):
            # Get individual data
            mask = data[id_key] == _id
            times = data[time_key][mask]
            measurements = data[meas_key][mask]
            measurementErrors = None if dataErrs is None else \
                [d[meas_key][mask] for d in dataErrs] if dataErrs_given_as_pm else \
                dataErrs[meas_key][mask]
            
            color = colors[index % n_colors]

            # Create Scatter plot
            self._add_data_trace(_id, times, measurements, color, row, col, measurementErrors)

    def add_prediction(
            self, data, row, col, biomarker=None, bulk_probs=[0.9], time_key='Time',
            biom_key='Biomarker', sample_key='Sample', colourIndex=None, legendLabel=None, n_colors=10):
        r"""
        Adds the prediction for the observable pharmacodynamic biomarker values
        to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a PD biomarker and a
        sample column. The time column determines the time of the biomarker
        measurement and the sample column the corresponding biomarker
        measurement. The biomarker column determines the biomarker type.

        A list of bulk probabilities ``bulk_probs`` can be specified, which are
        then added as area to the figure. The corresponding upper and lower
        percentiles are estimated from the ranks of the provided
        samples.

        .. warning::
            For low sample sizes the illustrated bulk probabilities may deviate
            significantly from the theoretical bulk probabilities. The upper
            and lower limit are determined from the rank of the samples for
            each time point.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        biomarker
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
        bulk_probs
            A list of bulk probabilities that are illustrated in the
            figure. If ``None`` the samples are illustrated as a scatter plot.
        time_key
            Key label of the :class:`pandas.DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`pandas.DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        sample_key
            Key label of the :class:`pandas.DataFrame` which specifies the
            sample column. Defaults to ``'Sample'``.
        colourIndex
            Optional. Which colour to select, to draw the prediction with.
        legendLabel
            Optional. Label to write on the legend for this prediction.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, biom_key, sample_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask]

        # Add samples as scatter plot if no bulk probabilites are provided, and
        # terminate method
        if bulk_probs is None:
            times = data[time_key]
            samples = data[sample_key]
            self._add_prediction_scatter_trace(times, samples, colourIndex, legendLabel, n_colors=n_colors)

            return None

        # Not more than 7 bulk probabilities are allowed (Purely aesthetic
        # criterion)
        if len(bulk_probs) > 7:
            raise ValueError(
                'At most 7 different bulk probabilities can be illustrated at '
                'the same time.')

        # Make sure that bulk probabilities are between 0 and 1
        bulk_probs = [float(probability) for probability in bulk_probs]
        for probability in bulk_probs:
            if (probability < 0) or (probability > 1):
                raise ValueError(
                    'The provided bulk probabilities have to between 0 and 1.')

        # Add bulk probabilities to figure
        percentile_df = self._compute_bulk_probs(
            data, bulk_probs, time_key, sample_key)
        self._add_prediction_bulk_prob_trace(percentile_df, row, col, colourIndex, legendLabel, n_colors=n_colors)

    def _add_updatemenu(self):
        """
        Adds a button to the figure that switches the biomarker scale from
        linear to logarithmic.
        """
        self._fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"yaxis2.type": "linear"}],
                            label="Linear y-scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis2.type": "log"}],
                            label="Log y-scale",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 0, "t": -10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

    def set_axis_labels(self, xlabels=None, ylabels=None, row=None, col=None):
        """
        Sets the labels of each axis along the given row or col.
        """
        #Make sure rows and cols match in length
        multipleLabels = False
        if row is None and col is not None:
            if hasattr(col, "__len__"):
                row = [row for c in col]
                multipleLabels = True
        elif row is not None and col is None:
            if hasattr(row, "__len__"):
                col = [col for r in row]
                multipleLabels = True
        elif row is not None and col is not None:
            if hasattr(row, "__len__"):
                assert len(col) == len(row)
                multipleLabels = True
        
        #Make sure xlabels match in length
        if xlabels is not None:
            if ylabels is not None and isinstance(xlabels, (list, tuple, np.ndarray)):
                assert len(xlabels) == len(ylabels)
        elif ylabels is not None:
            if xlabels is not None and isinstance(ylabels, (list, tuple, np.ndarray)):
                assert len(xlabels) == len(ylabels)

        #Set xlabels
        if xlabels is not None:
            if multipleLabels:
                _xlabels = [xlabels]*len(row) if isinstance(xlabels, str) else xlabels
                for xlabel, r, c in zip(_xlabels, row, col):
                    self._fig.update_xaxes(title=xlabel, row=r, col=c)
            else:
                self._fig.update_xaxes(title=xlabels, row=row, col=col)
        #Set ylabels
        if ylabels is not None:
            if multipleLabels:
                _ylabels = [ylabels]*len(row) if isinstance(ylabels, str) else ylabels
                for ylabel, r, c in zip(_ylabels, row, col):
                    self._fig.update_yaxes(title=ylabel, row=r, col=c)
            else:
                self._fig.update_yaxes(title=ylabels, row=row, col=col)


class PDTimeSeriesSubPlots(plots.SingleSubplotFigure):
    """
    A figure class that visualises measurements of a pharmacodynamic biomarker
    across multiple individuals.

    Measurements of a pharmacodynamic biomarker over time are visualised as a
    scatter plot.

    Extends :class:`SingleSubplotFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    nrows
        Integer number of rows to pass to plotly for forming subplots.
    ncols
        Integer number of columns to pass to plotly for forming subplots.
    shared_x
        Boolean flag to pass to plotly if subplots should share x-axes.
    shared_y
        Boolean flag to pass to plotly if subplots should share y-axes.
    """

    def __init__(self, updatemenu=True, nrows=1, ncols=1, shared_x=False, shared_y=False):
        super(PDTimeSeriesSubPlots, self).__init__()

        self._create_template_figure(
            rows=nrows, cols=ncols, shared_x=shared_x, shared_y=shared_y)

        if updatemenu:
            self._add_updatemenu()

        self.nrows, self.ncols = nrows, ncols

    def _add_data_trace(self, _id, times, measurements, color, row, col, measurementErrors=None):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        _measurementErrors = None if measurementErrors is None else \
            measurementErrors if isinstance(measurementErrors, go.scatter.ErrorY) else \
            go.scatter.ErrorY(array=measurementErrors) if len(measurementErrors)==len(measurements) else \
            go.scatter.ErrorY(
                    array=measurementErrors[1], arrayminus=measurementErrors[0]
                ) if len(measurementErrors)==2 else None
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=measurements,
                error_y=_measurementErrors,
                name="ID: %s" % str(_id),
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
                row=row,
                col=col)

    def _add_simulation_trace(self, times, biomarker, row, col):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=biomarker,
                name="Model",
                showlegend=True,
                mode="lines",
                line=dict(color='black')),
                row=row,
                col=col)

    def add_data(
            self, data, row, col, biomarker=None, id_key='ID', time_key='Time',
            biom_key='Biomarker', meas_key='Measurement', n_colors=10, dataErrs=None):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time, a PD
        biomarker and a measurement column, and adds a scatter plot of the
        measurement time series to the figure. Each individual receives a
        unique colour.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD data in form of
            an ID, time, and biomarker column.
        biomarker
            The measured bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
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
        meas_key
            Key label of the :class:`DataFrame` which specifies the column of
            the measured PD biomarker. Defaults to ``'Measurement'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key, meas_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biomarker
        data = data[mask]
        dataErrs_given_as_pm = dataErrs is not None and isinstance(dataErrs, list) and len(dataErrs)==2
        dataErrs = None if dataErrs is None else \
            [d[mask] for d in dataErrs] if dataErrs_given_as_pm else \
            dataErrs[mask]
        # Get a colour scheme
        # colors = plotly.colors.qualitative.Plotly
        colors = [mplc.rgb2hex(c) for c in color_palette("bright", n_colors=n_colors)]

        # Fill figure with scatter plots of individual data
        ids = data[id_key].unique()
        for index, _id in enumerate(ids):
            # Get individual data
            mask = data[id_key] == _id
            times = data[time_key][mask]
            measurements = data[meas_key][mask]
            measurementErrors = None if dataErrs is None else \
                [d[meas_key][mask] for d in dataErrs] if dataErrs_given_as_pm else \
                dataErrs[meas_key][mask]
            
            color = colors[index % n_colors]

            # Create Scatter plot
            self._add_data_trace(_id, times, measurements, color, row, col, measurementErrors=measurementErrors)

    def add_simulation(self, data, row, col, time_key='Time', biom_key='Biomarker'):
        """
        Adds a pharmacodynamic time series simulation to the figure.

        Expects a :class:`pandas.DataFrame` with a time and a PD biomarker
        column, and adds a line plot of the biomarker time series to the
        figure.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, biom_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        times = data[time_key]
        biomarker = data[biom_key]

        self._add_simulation_trace(times, biomarker, row, col)

    def _add_updatemenu(self):
        """
        Adds a button to the figure that switches the biomarker scale from
        linear to logarithmic.
        """
        self._fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"yaxis2.type": "linear"}],
                            label="Linear y-scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis2.type": "log"}],
                            label="Log y-scale",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 0, "t": -10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

    def set_axis_labels(self, xlabels=None, ylabels=None, row=None, col=None):
        """
        Sets the labels of each axis along the given row or col.
        """
        #Make sure rows and cols match in length
        multipleLabels = False
        if row is None and col is not None:
            if hasattr(col, "__len__"):
                row = [row for c in col]
                multipleLabels = True
        elif row is not None and col is None:
            if hasattr(row, "__len__"):
                col = [col for r in row]
                multipleLabels = True
        elif row is not None and col is not None:
            if hasattr(row, "__len__"):
                assert len(col) == len(row)
                multipleLabels = True
        
        #Make sure xlabels match in length
        if xlabels is not None:
            if ylabels is not None and isinstance(xlabels, (list, tuple, np.ndarray)):
                assert len(xlabels) == len(ylabels)
        elif ylabels is not None:
            if xlabels is not None and isinstance(ylabels, (list, tuple, np.ndarray)):
                assert len(xlabels) == len(ylabels)

        #Set xlabels
        if xlabels is not None:
            if multipleLabels:
                _xlabels = [xlabels]*len(row) if isinstance(xlabels, str) else xlabels
                for xlabel, r, c in zip(_xlabels, row, col):
                    self._fig.update_xaxes(title=xlabel, row=r, col=c)
            else:
                self._fig.update_xaxes(title=xlabels, row=row, col=col)
        #Set ylabels
        if ylabels is not None:
            if multipleLabels:
                _ylabels = [ylabels]*len(row) if isinstance(ylabels, str) else ylabels
                for ylabel, r, c in zip(_ylabels, row, col):
                    self._fig.update_yaxes(title=ylabel, row=r, col=c)
            else:
                self._fig.update_yaxes(title=ylabels, row=row, col=col)
