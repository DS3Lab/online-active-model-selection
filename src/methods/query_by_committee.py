import numpy as np
import scipy.stats as stats

def query_by_committee(data, idx_budget, streaming_data_indices, tuning_par):

    # Set vals, params
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]

    # Initialize
    prior = np.ones(data._num_models) / data._num_models
    posterior = prior
    z_i_log = np.zeros(data._num_instances, dtype=int)
    z_t_budget = np.zeros(data._num_instances, dtype=int)

    # If the strategy is adaptive,
    for i in range(data._num_instances):

        # Measure the normalized entropy of the incoming data
        hist, bin_edges = np.histogram(predictions[i, :], bins=data._num_classes)
        prob_i = hist/np.sum(hist)
        entropy_i = stats.entropy(prob_i, base=2) / np.log2(data._num_classes) * tuning_par

        # Check if the normalized entropy is greater than 1
        if entropy_i > 1:
            entropy_i = 1
        if entropy_i < 0:
            entropy_i = 0
        # Randomly decide whether to query z_i or not
        z_i = np.random.binomial(size=1, n=1, p=entropy_i)
        # Log the value
        z_i_log[i] = z_i

        # Terminate if budget is exceeded
        if np.sum(z_i_log) <= budget:
            z_t_budget[i] = z_i_log[i]


    # Labelling decisions as 0's and 1's
    labelled_instances = z_i_log
    ct_log = np.ones(data._num_instances, dtype=int)


    return (labelled_instances, ct_log, z_t_budget)