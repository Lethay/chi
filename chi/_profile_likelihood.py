from chi import InferenceController
import numpy as np
import pandas as pd
import pints
from tqdm.notebook import tqdm
import warnings

#plc = ProfileLikelihoodController(log_posteriors)
#plc.set_initial_parameters(map_estimates)
#plc.set_transform(get_transformFunc(chiNames, measurementNames, error_models))
class ProfileLikelihoodController(InferenceController):
    """
        Find the profile likelihood of each parameter, from a given solution.
        parameters:
        - problem:                  a chi.ProblemModellingController.
        - optimisation_controller:  a chi.OptimisationController.
        - bounds:                   a container of size (n_parameters, 2) between which the profile_likelihood will be
                                    calculated.
        - initValues:               initial values from which the profile should be calculated. If None, this is
                                    calculated from map_estimates instead.
        - map_estimates:            results of optimising the problem with optimisation_controller. Used to calculate
                                    initValues, if it is None.
    """
    def __init__(self, log_posteriors, method=pints.CMAES):
        super(ProfileLikelihoodController, self).__init__(log_posteriors)

        # Set default optimiser
        self._optimiser = method

        # Set default transformation
        transform = pints.ComposedTransformation(
            *[pints.IdentityTransformation(1) for p in log_posteriors[0].get_parameter_names()]
        )
        self.set_transform(transform)

    def _get_id_parameter_pairs(self, log_posterior):
        """
        Returns a zipped list of ID (pop_prefix), and parameter name pairs.

        Posteriors that are not derived from a HierarchicalLoglikelihood carry
        typically only a single ID (the ID of the individual they are
        modelling). In that case all parameters are assigned with the same ID.

        For posteriors that are derived from a HierarchicalLoglikelihood it
        often makes sense to label the parameters with different IDs. These ID
        parameter name pairs are reconstructed here.
        """
        # Get IDs and parameter names
        ids = log_posterior.get_id()
        parameters = log_posterior.get_parameter_names()

        # If IDs is only one ID, expand to list of length n_parameters
        if not isinstance(ids, list):
            n_parameters = len(parameters)
            ids = [ids] * n_parameters

        return zip(ids, parameters)

    def profile_likelihood(self, problem, bounds=None, n_max_iterations=1000, n_max_unchanged_iterations=100, n_runs=1,
            n_values_per_dimension=11, show_param_progress_bar=False, show_param_value_progress_bar=False,
            show_id_progress_bar=False, show_run_progress_bar=False, log_to_screen=True,
            withoutOptimisation=False):
        #TODO: Make this work for withPopulationModel
        #Check we've been given initial parameters
        assert self._initial_params is not None, "set_initial_parameters must be run before profile_likelihood."

        #Check that the problem has the correct shape
        assert problem.get_n_parameters() == self._n_parameters, "Problem required n_parameters: %d. Observed: %d."%(
            self._n_parameters, problem.get_n_parameters()
        )

        #If we aren't given bounds, generate some simple ones from the initial values
        if bounds is None:
            pMin = np.min(self._initial_params, axis=(0,1))
            pMax = np.max(self._initial_params, axis=(0,1))
            bMin = self._transform.to_model(0.5*self._transform.to_search(pMin))
            bMax = self._transform.to_model(2.0*self._transform.to_search(pMax))
            bounds = np.transpose([bMin, bMax])

        #If we were given some, check they have the correct shape
        else:
            bounds = np.array(bounds)
            assert bounds.shape == (self._n_parameters, 2), "Bounds required shape: (%d, 2). Observed: (%d, %d)."%(
                self._n_parameters, *bounds.shape)

        #Transform boundaries
        if self._transform is not None:
            tr_bounds = np.transpose([self._transform.to_search(b) for b in np.transpose(bounds)])

        #Get parameter names
        chiNames = problem.get_parameter_names()

        #Get prior and log-likelihoods
        priors      = [p._log_prior      for p in self._log_posteriors]
        # likelihoods = [p._log_likelihood for p in self._log_posteriors]

        # Initialise result dataframe
        result = pd.DataFrame(
            columns=['ID', 'Fixed parameter', 'Fixed value', 'Parameter', 'Estimate', 'Score', 'Run'])

        # Initialise intermediate container for individual runs
        run_result = pd.DataFrame(
            columns=['ID', 'Fixed parameter', 'Fixed value', 'Parameter', 'Estimate', 'Score', 'Run'])

        #For each parameter:
        for param_id in tqdm(
                range(self._n_parameters), disable=not show_param_progress_bar):
            if not show_param_progress_bar:
                print("Param %d/%d"%(param_id, self._n_parameters))

            #Get variables
            fixedParamName = chiNames[param_id]
            tr_bound = tr_bounds[param_id]

            #Find values of parameters we're not getting the profile of
            unfixedIndices    = np.delete(np.arange(self._n_parameters), param_id)
            unfixedInitValues = self._initial_params[:, 0, unfixedIndices]
            unfixedPriors     = [[p._priors[i] for i in unfixedIndices] for p in priors]

            #Get transform and bound functions of unfixed values. TODO: bounds
            trs = self._transform._transformations
            fixedTransform   = trs[param_id]
            unfixedTransform = pints.ComposedTransformation(*[trs[i] for i in unfixedIndices])
            unfixedBounds    = self._bounds 

            #Prepare results container
            run_result['Parameter']       = np.array(chiNames)[unfixedIndices]
            run_result['Fixed parameter'] = fixedParamName

            #Loop over parameter values
            fixedParamValues = fixedTransform.to_model(
                np.linspace(tr_bound[0], tr_bound[1], n_values_per_dimension))
            for param_value_id in tqdm(
                    range(n_values_per_dimension), disable=not show_param_value_progress_bar):
                if not show_param_value_progress_bar:
                    print("Param %d/%d, value %d/%d"%(
                        param_id, self._n_parameters, param_value_id, n_values_per_dimension))
                fixedValue = fixedParamValues[param_value_id]

                #Fix parameters
                problem.fix_parameters({fixedParamName: fixedValue})
                
                #reset prior and likelihood
                problem.set_log_prior(unfixedPriors[0])
                fixed_log_posteriors = problem.get_log_posterior()

                #set prior ID-specific again
                for p, pr in enumerate(unfixedPriors):
                    fixed_log_posteriors[p]._log_prior = pints.ComposedLogPrior(*pr)

                #Prepare results container
                run_result['Fixed value']     = fixedValue

                #Loop for each individual:
                for posterior_id, log_posterior in enumerate(tqdm(
                        fixed_log_posteriors, disable=not show_id_progress_bar)):
                    if not show_id_progress_bar:
                        print("Param %d/%d, value %d/%d, ID %d/%d"%(
                            param_id, self._n_parameters, param_value_id, n_values_per_dimension,
                            posterior_id+1, len(fixed_log_posteriors)))

                    individual_result = pd.DataFrame(
                        columns=['ID', 'Fixed parameter', 'Fixed value', 'Parameter', 'Estimate', 'Score', 'Run'])

                    # Set ID of individual (or IDs of parameters, if hierarchical)
                    run_result['ID'] = log_posterior.get_id()

                    # Run optimisation multiple times
                    for run_id in tqdm(
                            range(n_runs), disable=not show_run_progress_bar):
                        if not show_run_progress_bar:
                            print("Param %d/%d, value %d/%d, ID %d/%d, run %d/%d"%(
                                param_id, self._n_parameters, param_value_id, n_values_per_dimension,
                                posterior_id+1, len(fixed_log_posteriors), run_id+1, n_runs))

                        if withoutOptimisation:
                            estimates = unfixedInitValues[posterior_id]
                            score = log_posterior(unfixedInitValues[posterior_id])
                        else:
                            opt = pints.OptimisationController(
                                function=log_posterior,
                                x0=unfixedInitValues[posterior_id],
                                method=self._optimiser,
                                transformation=unfixedTransform,
                                boundaries=unfixedBounds)

                            # Configure optimisation routine
                            opt.set_log_to_screen(log_to_screen)
                            opt.set_max_iterations(iterations=n_max_iterations)
                            opt.set_max_unchanged_iterations(iterations=n_max_unchanged_iterations)
                            opt.set_parallel(self._parallel_evaluation)

                            # Find optimal parameters
                            try:
                                estimates, score = opt.run()
                            except Exception:
                                # If inference breaks fill estimates with nan
                                estimates = [np.nan] * self._n_parameters
                                score = np.nan

                        # Save estimates and score of runs
                        run_result['Estimate'] = estimates
                        run_result['Score'] = score
                        run_result['Run'] = run_id + 1
                        individual_result = individual_result.append(run_result)
                    #End of run-loop

                    # Save runs for individual
                    result = result.append(individual_result)
                #End of individual-loop
    
                #Unfix parameter
                problem.fix_parameters({fixedParamName: None})
            #End of parameter-value-loop
        #End of parameter-loop

        # #Dumping data
        # import pickle
        # outfile = open(title + "_plh.pickle", "wb")
        # pickle.dump(result, outfile)
        # outfile.close()     

        # #Simple differences
        # for nam in chiNames:
        #     maxScore = np.max(result[result.loc[:, "Fixed parameter"]==nam]).loc["Score"]
        #     minScore = np.min(result[result.loc[:, "Fixed parameter"]==nam]).loc["Score"]
        #     print(nam, maxScore, minScore)

        # #Load data
        # infile = open(title + "_plh.pickle", "rb")
        # result = pickle.load(infile)
        # infile.close()
        # #Plot
        # import seaborn as sb
        # import matplotlib.pyplot as plt
        # chiNames = list(np.unique(result.loc[:, "Fixed parameter"]))
        # for nam in chiNames:
        #     plt.figure()
        #     sb.lineplot(data=result[result.loc[:, "Fixed parameter"]==nam], x="Fixed value", y="Score")
        #     plt.title(nam)
        # plt.show(block=False)

        return result

    def set_initial_parameters(
            self, data, id_key='ID', param_key='Parameter', est_key='Estimate',
            score_key='Score', run_key='Run'):
        """
        Sets the initial parameter values of the MCMC runs to the parameter set
        with the maximal a posteriori probability across a number of parameter
        sets.

        This method is intended to be used in conjunction with the results of
        the :class:`OptimisationController`.

        It expects a :class:`pandas.DataFrame` with the columns 'ID',
        'Parameter', 'Estimate', 'Score' and 'Run'. The maximum a posteriori
        probability values across all estimates is determined and used as
        initial point for the MCMC runs.

        If multiple parameter sets assume the maximal a posteriori probability
        value, a parameter set is drawn randomly from them.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the parameter estimates in form of
            a parameter, estimate and score column.
        id_key
            Key label of the :class:`DataFrame` which specifies the individual
            ID column. Defaults to ``'ID'``.
        param_key
            Key label of the :class:`DataFrame` which specifies the parameter
            name column. Defaults to ``'Parameter'``.
        est_key
            Key label of the :class:`DataFrame` which specifies the parameter
            estimate column. Defaults to ``'Estimate'``.
        score_key
            Key label of the :class:`DataFrame` which specifies the score
            estimate column. The score refers to the maximum a posteriori
            probability associated with the estimate. Defaults to ``'Score'``.
        run_key
            Key label of the :class:`DataFrame` which specifies the
            optimisation run column. Defaults to ``'Run'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, param_key, est_key, score_key, run_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Convert dataframe IDs and parameter names to strings
        data = data.astype({id_key: str, param_key: str})

        # Get posterior IDs (one posterior may have multiple IDs, one
        # for each parameter)
        for index, log_posterior in enumerate(self._log_posteriors):

            # Get MAP for each parameter of log_posterior
            for prefix, parameter in self._get_id_parameter_pairs(
                    log_posterior):

                # Get estimates for ID (prefix)
                if prefix is None:
                    #Pandas treats None as np.nan, so we'd get None != None
                    mask = data[id_key].apply(lambda x: x==None or x=='None')
                else:
                    mask = data[id_key] == prefix
                individual_data = data[mask]

                # If ID (prefix) doesn't exist, move on to next iteration
                if individual_data.empty:
                    warnings.warn(
                        'The log-posterior ID <' + str(prefix) + '> could not'
                        ' be identified in the dataset for parameter ' + str(parameter)+
                        ', and was therefore not set to a specific value.')
                    
                    continue

                # Among estimates for this ID (prefix), get the relevant
                # parameter
                mask = individual_data[param_key] == parameter
                individual_data = individual_data[mask]

                # If parameter with this ID (prefix) doesn't exist, move on to
                # next iteration
                if individual_data.empty:
                    warnings.warn(
                        'The parameter <' + str(parameter) + '> with ID '
                        '<' + str(prefix) + '> could not be identified in the '
                        'dataset, and was therefore not set to a specific '
                        'value.')

                    continue

                #Find out how many runs are in the data we were given
                runs = individual_data[run_key].unique()

                # Get estimates with maximum a posteriori probability
                max_prob = individual_data[score_key].max()
                mask = individual_data[score_key] == max_prob
                individual_data = individual_data[mask]
                runs = individual_data[run_key].unique()

                # Choose a random value if we have multiple best runs, then set map_estimate
                selected_param_set = np.random.choice(runs)
                mask = individual_data[run_key] == selected_param_set
                individual_data = individual_data[mask]
                map_estimate = individual_data[est_key].to_numpy()

                # Create mask for parameter position in log-posterior
                ids = log_posterior.get_id()
                if not isinstance(ids, list):
                    n_parameters = len(self._parameters)
                    ids = [ids] * n_parameters
                id_mask = np.array(ids) == prefix
                param_mask = np.array(self._parameters) == parameter
                mask = id_mask & param_mask

                # Set initial parameters across runs to map estimate
                self._initial_params[index, :, mask] = map_estimate

    def set_optimiser(self, optimiser):
        """
        Sets method that is used to find the maximum a posteiori probability
        estimates.
        """
        if not issubclass(optimiser, pints.Optimiser):
            raise ValueError(
                'Optimiser has to be a `pints.Optimiser`.')
        self._optimiser = optimiser

