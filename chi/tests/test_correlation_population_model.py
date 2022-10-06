import warnings
import chi
import numpy as np
from scipy.stats import pearsonr

popModels = [
    chi.LogNormalModelRelativeSigma(),
    chi.TruncatedGaussianModelRelativeSigma(),
    chi.TruncatedGaussianModelRelativeSigma(),
    chi.TruncatedGaussianModelRelativeSigma(),
    chi.PooledModel()
]

popModel     = chi.ComposedPopulationModel(popModels)
corPopModel  = chi.ComposedCorrelationPopulationModel(popModels, np.eye(len(popModels)))
topP0 = [79788.30976970856, 1.8324680380216178, 0.10898055097178613, 1.3500957631115962, 0.9353172045851563,
    0.8898984969855425, 4.990576447449756, 0.736495445386129, 0.7596904254524167]
self = corPopModel; parameters=topP0; n_samples=2; seed=None; covariates=None 

# ─── TEST FOR GAUSS ───────────────────────────────────────────────────────────
n_samp = 100000
cor = np.eye(len(popModels))
i, j = 1, 2
cor[i, j] = cor[j, i] = 0.9
corPopModel2 = chi.ComposedCorrelationPopulationModel(popModels, cor)
samps = corPopModel2.sample(topP0, n_samp)
cov = np.cov(np.transpose(samps))
std = np.std(samps, axis=0)
cov_div_var = cov / std / std[:, np.newaxis]
for row in cov_div_var:
    for col in row:
        print(("%.2f"%col).ljust(6), end=" ")
    print("")
print(pearsonr(samps[:, i], samps[:, j]))

# ─── TEST FOR LOG NORMAL ──────────────────────────────────────────────────────
i, j = 0, 2
cor = np.eye(len(popModels))
cor[i, j] = cor[j, i] = 0.9
corPopModel2 = chi.ComposedCorrelationPopulationModel(popModels, cor)
correlatedSamples = corPopModel2.sample(topP0, n_samp)
print(pearsonr(correlatedSamples[:, i], correlatedSamples[:, j])) #not quite perfect for log-normal dist, as it's the RANDOM EFFECTS that we correlated

# ─── CHECK SAMPLING ───────────────────────────────────────────────────────────
uncorrelatedSamples    = popModel.sample(topP0, n_samp)
corPopUncorrelatedSamples = corPopModel.sample(topP0, n_samp)
print(np.mean(uncorrelatedSamples, axis=0), np.mean(corPopUncorrelatedSamples,axis=0), sep="\n")

# ─── TAKE TWO SAMPLES AND CHECK LIKELIHOOD ────────────────────────────────────
print("Checking likelihoods over two samples:")
samp = uncorrelatedSamples[:2]
# self = corPopModel; parameters=topP0; observations=samp; covariates=None 
popLike    = popModel.compute_log_likelihood(topP0, samp)
corPopLike = corPopModel.compute_log_likelihood(topP0, samp)
corPop2Like = corPopModel2.compute_log_likelihood(topP0, samp)
print("Standard pop likelihood:", popLike)
print("Correlation pop likelihood with no correlations:", corPopLike,  "diff", corPopLike -popLike, "(%.2f%%)"%(100*(corPopLike -popLike)/np.abs(popLike)))
print("Correlation pop likelihood with correlations:",    corPop2Like, "diff", corPop2Like-popLike, "(%.2f%%)"%(100*(corPop2Like-popLike)/np.abs(popLike)))

# ─── MANY SAMPLES ─────────────────────────────────────────────────────────────
n_sub_samp = 1000
print("Over %d samples drawn from a no-correlation model:"%n_sub_samp)
popLike    = popModel.compute_log_likelihood(topP0, uncorrelatedSamples[:n_sub_samp])
corPopLike = corPopModel.compute_log_likelihood(topP0, uncorrelatedSamples[:n_sub_samp])
corPop2Like = corPopModel2.compute_log_likelihood(topP0, uncorrelatedSamples[:n_sub_samp])
print("Standard pop likelihood:", popLike)
print("Correlation pop likelihood with no correlations:", corPopLike,  "diff", corPopLike -popLike, "(%.2f%%)"%(100*(corPopLike -popLike)/np.abs(popLike)))
print("Correlation pop likelihood with correlations:",    corPop2Like, "diff", corPop2Like-popLike, "(%.2f%%)"%(100*(corPop2Like-popLike)/np.abs(popLike)))

print("Over %d samples drawn from a correlation model:"%n_sub_samp)
popLike    = popModel.compute_log_likelihood(topP0, correlatedSamples[:n_sub_samp])
corPopLike = corPopModel.compute_log_likelihood(topP0, correlatedSamples[:n_sub_samp])
corPop2Like = corPopModel2.compute_log_likelihood(topP0, correlatedSamples[:n_sub_samp])
print("Standard pop likelihood:", popLike)
print("Correlation pop likelihood with no correlations:", corPopLike,  "diff", corPopLike -popLike, "(%.2f%%)"%(100*(corPopLike -popLike)/np.abs(popLike)))
print("Correlation pop likelihood with correlations:",    corPop2Like, "diff", corPop2Like-popLike, "(%.2f%%)"%(100*(corPop2Like-popLike)/np.abs(popLike)))

# ─── CATCH WARNINGS ───────────────────────────────────────────────────────────
for s, samp in enumerate(correlatedSamples[:n_sub_samp]):
    with warnings.catch_warnings():
        try:
            popLike    = popModel.compute_log_likelihood(topP0, [samp])
            corPopLike = corPopModel.compute_log_likelihood(topP0, [samp])
            corPop2Like = corPopModel2.compute_log_likelihood(topP0, [samp])
        except RuntimeWarning as e:
            print(s, samp)
            raise e

# ─── TIMING TEST ──────────────────────────────────────────────────────────────
# samp = uncorrelatedSamples[:2]
# print("composed population model")
# %timeit popModel.compute_log_likelihood(topP0, samp)
# print("composed population population model, without correlations")
# %timeit corPopModel.compute_log_likelihood(topP0, samp)
# print("composed population population model, with correlations")
# %timeit corPopModel2.compute_log_likelihood(topP0, samp)


# ─── MANUAL LOOPING THROUGH LOG-LIKELIHOOD ────────────────────────────────────
# score = 0
# cdfs = np.zeros(observations.shape)
# cov = None
# current_dim = 0
# current_param = 0
# current_cov = 0
# p=0
# # Get covariates
# pop_model = self._population_models[p]
# if self._n_covariates > 0:
#     end_cov = current_cov + pop_model.n_covariates()
#     cov = covariates[:, current_cov:end_cov]
#     current_cov = end_cov

# end_dim = current_dim + pop_model.n_dim()
# end_param = current_param + pop_model.n_parameters()
# score += pop_model.compute_log_likelihood(
#     parameters=parameters[current_param:end_param],
#     observations=observations[:, current_dim:end_dim],
#     covariates=cov
# )
# if chi.is_heterogeneous_model(pop_model) or chi.is_pooled_model(pop_model):
#     cdfs[:, current_dim:end_dim] = 0.5
# else:
#     cdfs[:, current_dim:end_dim] = pop_model.compute_cdf(
#         parameters=parameters[current_param:end_param],
#         observations=observations[:, current_dim:end_dim]
#     )
# print(p, current_param, end_param, current_dim, end_dim, np.array(cdfs[:, current_dim:end_dim]).flatten())
# p+=1
# current_dim = end_dim
# current_param = end_param
